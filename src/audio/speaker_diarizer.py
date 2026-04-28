from __future__ import annotations

import os
import shutil
import pathlib
import platform
from typing import Dict, List
from collections import Counter

import numpy as np
import torch
import torchaudio

# Windows platformunda SpeechBrain'in symlink hatalarını (WinError 1314) önlemek için yama
if platform.system() == "Windows":
    _orig_symlink_to = pathlib.Path.symlink_to
    def _safe_symlink_to(self, target, target_is_directory=False):
        try:
            _orig_symlink_to(self, target, target_is_directory)
        except OSError:
            if self.exists():
                if self.is_dir():
                    shutil.rmtree(self)
                else:
                    self.unlink()
            if target_is_directory:
                shutil.copytree(target, self)
            else:
                shutil.copy(target, self)
    pathlib.Path.symlink_to = _safe_symlink_to

from silero_vad import load_silero_vad, get_speech_timestamps
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score # Amaç: Dinamik cluster sayısı seçmek için clustering ve kalite metriği kullanmak.
from speechbrain.inference.classifiers import EncoderClassifier


class SpeakerDiarizer:
    """
    CPU uyumlu hafif diarization:
      1) Silero VAD ile konuşma bölgelerini çıkarır
      2) SpeechBrain ECAPA ile speaker embedding üretir
      3) Agglomerative clustering ile Speaker_0 / Speaker_1 atar
    """

    def __init__(self):
        self.sample_rate = int(os.getenv("SENSIFYHR_DIAR_SR", "16000"))
        self.min_speakers = int(os.getenv("SENSIFYHR_MIN_SPEAKERS", "2")) # Panel mülakat senaryosu için speaker sayısını sabit 2 yerine üst sınırla yönetmek.
        self.max_speakers = int(os.getenv("SENSIFYHR_MAX_SPEAKERS", "5"))
        self.min_speech_ms = int(os.getenv("SENSIFYHR_VAD_MIN_SPEECH_MS", "500"))
        self.min_silence_ms = int(os.getenv("SENSIFYHR_VAD_MIN_SILENCE_MS", "250"))
        self.max_window_sec = float(os.getenv("SENSIFYHR_EMBED_MAX_WINDOW_SEC", "3.0"))
        self.min_window_sec = float(os.getenv("SENSIFYHR_EMBED_MIN_WINDOW_SEC", "0.8"))
        self.window_hop_sec = float(os.getenv("SENSIFYHR_EMBED_WINDOW_HOP_SEC", "1.5"))

        self.vad_model = load_silero_vad()
        self.embedder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"},
            savedir=os.getenv("SENSIFYHR_SB_CACHE", "./models/speechbrain_ecapa"),
        )
        # Amaç:
        # Son diarization çalışmasının kalite tanı bilgilerini tutmak.
        self.last_diarization_diagnostics: Dict = {}

    def _load_waveform(self, audio_path: str):
        waveform, sr = torchaudio.load(audio_path)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
            sr = self.sample_rate
        return waveform, sr

    def _get_vad_regions(self, waveform: torch.Tensor) -> List[Dict[str, float]]:
        speech = get_speech_timestamps(
            waveform.squeeze(0),
            self.vad_model,
            sampling_rate=self.sample_rate,
            min_speech_duration_ms=self.min_speech_ms,
            min_silence_duration_ms=self.min_silence_ms,
            return_seconds=True,
        )
        return [{"start": float(x["start"]), "end": float(x["end"])} for x in speech]

    def _split_region_into_windows(self, start: float, end: float) -> List[Dict[str, float]]:
        duration = max(0.0, end - start)
        if duration < self.min_window_sec:
            return []
        if duration <= self.max_window_sec:
            return [{"start": start, "end": end}]

        windows = []
        cur = start
        while cur < end:
            nxt = min(cur + self.max_window_sec, end)
            if (nxt - cur) >= self.min_window_sec:
                windows.append({"start": cur, "end": nxt})
            if nxt >= end:
                break
            cur += self.window_hop_sec
        return windows

    def _build_embedding_windows(self, regions: List[Dict[str, float]]) -> List[Dict[str, float]]:
        windows = []
        for region in regions:
            windows.extend(self._split_region_into_windows(region["start"], region["end"]))
        return windows

    def _embed_window(self, waveform: torch.Tensor, sr: int, start: float, end: float) -> np.ndarray:
        s = int(start * sr)
        e = int(end * sr)
        chunk = waveform[:, s:e]
        if chunk.numel() == 0:
            raise ValueError("Empty audio chunk for embedding.")
        with torch.no_grad():
            emb = self.embedder.encode_batch(chunk)
        vec = emb.squeeze().detach().cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(vec) + 1e-9
        return vec / norm
    
    # Amaç:
    # En iyi cluster sayısını silhouette score ile seçmek.
    # k=2..max_speakers aralığında dener ve:
    #   1) seçilen k
    #   2) en iyi quality score
    #   3) denenen k skorları
    # döndürür.
    def _select_best_k(self, X: np.ndarray):
        n_samples = len(X)

        # Amaç:
        # Çok az örnek varsa güvenli fallback dön.
        if n_samples < 3:
            return 2, None, {}

        upper_k = min(self.max_speakers, n_samples - 1)
        lower_k = min(self.min_speakers, upper_k)

        best_k = lower_k
        best_score = -1.0
        k_scores: Dict[int, float] = {}

        for k in range(lower_k, upper_k + 1):
            try:
                labels = AgglomerativeClustering(
                    n_clusters=k,
                    metric="cosine",
                    linkage="average",
                ).fit_predict(X)

                # Amaç:
                # Silhouette yalnızca 2..n_samples-1 label durumunda anlamlıdır.
                unique_labels = len(set(labels))
                if unique_labels < 2 or unique_labels >= n_samples:
                    continue

                score = float(silhouette_score(X, labels, metric="cosine"))
                k_scores[k] = round(score, 4)

                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception:
                continue

        final_score = round(best_score, 4) if best_score >= 0 else None
        return best_k, final_score, k_scores

    def diarize(self, audio_path: str) -> List[Dict[str, float | str]]:
        waveform, sr = self._load_waveform(audio_path)
        vad_regions = self._get_vad_regions(waveform)
        embed_windows = self._build_embedding_windows(vad_regions)

        if not embed_windows:
            self.last_diarization_diagnostics = {
                "selected_cluster_k": 0,
                "cluster_quality_score": None,
                "k_search_scores": {},
                "vad_region_count": len(vad_regions),
                "embedding_window_count": 0,
                "speaker_window_distribution": {},
            }
            return []

        embeddings = []
        valid_windows = []
        for w in embed_windows:
            try:
                vec = self._embed_window(waveform, sr, w["start"], w["end"])
            except Exception:
                continue
            embeddings.append(vec)
            valid_windows.append(w)

        if not valid_windows:
            self.last_diarization_diagnostics = {
                "selected_cluster_k": 0,
                "cluster_quality_score": None,
                "k_search_scores": {},
                "vad_region_count": len(vad_regions),
                "embedding_window_count": 0,
                "speaker_window_distribution": {},
            }
            return []

        # Amaç:
        # Pencere sayısı çok azsa mevcut pencereleri tek tek speaker gibi işaretleyip dönmek.
        if len(valid_windows) < 2:
            for idx, w in enumerate(valid_windows):
                w["speaker_id"] = f"Speaker_{idx}"

            self.last_diarization_diagnostics = {
                "selected_cluster_k": len(valid_windows),
                "cluster_quality_score": None,
                "k_search_scores": {},
                "vad_region_count": len(vad_regions),
                "embedding_window_count": len(valid_windows),
                "speaker_window_distribution": {f"Speaker_{idx}": 1 for idx in range(len(valid_windows))},
            }

            return valid_windows

        # Amaç:
        # Embedding matrisini oluşturmak.
        X = np.vstack(embeddings)

        # Amaç:
        # En uygun speaker sayısını dinamik seçmek.
        best_k, best_score, k_scores = self._select_best_k(X)

        # Amaç:
        # Seçilen k ile son clustering'i yapmak.
        clustering = AgglomerativeClustering(
            n_clusters=best_k,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(X)

        # Amaç:
        # Seçilen cluster kalitesini ve speaker dağılımını rapora taşımak.
        speaker_hist = Counter(labels.tolist())

        self.last_diarization_diagnostics = {
            "selected_cluster_k": int(best_k),
            "cluster_quality_score": best_score,
            "k_search_scores": {str(k): v for k, v in k_scores.items()},
            "vad_region_count": len(vad_regions),
            "embedding_window_count": len(valid_windows),
            "speaker_window_distribution": {f"Speaker_{int(k)}": int(v) for k, v in speaker_hist.items()},
        }

        diarized = []
        for w, label in zip(valid_windows, labels):
            diarized.append(
                {
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                    "speaker_id": f"Speaker_{int(label)}",
                }
            )
        return diarized


# Amaç:
# İki zaman aralığı arasındaki çakışma süresini hesaplamak.
def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


# Amaç:
# Overlap zayıf kaldığında, segmentin orta noktasına en yakın diarization penceresini bulmak.
def _nearest_window_speaker(seg_start: float, seg_end: float, diarized_windows: List[Dict]) -> str:
    if not diarized_windows:
        return "Speaker_0"

    seg_mid = (seg_start + seg_end) / 2.0

    best_speaker = "Speaker_0"
    best_distance = float("inf")

    for win in diarized_windows:
        win_mid = (float(win["start"]) + float(win["end"])) / 2.0
        dist = abs(seg_mid - win_mid)
        if dist < best_distance:
            best_distance = dist
            best_speaker = str(win["speaker_id"])

    return best_speaker


# Amaç:
# Tek başına araya giren kısa speaker sıçramalarını yumuşatmak.
# Örn: A - B - A gibi kısa bir sapmayı tekrar A yapar.
def _smooth_speaker_labels(segments: List[Dict], bridge_gap_sec: float = 1.2) -> List[Dict]:
    if len(segments) < 3:
        return segments

    out = [dict(seg) for seg in segments]

    for i in range(1, len(out) - 1):
        prev_seg = out[i - 1]
        curr_seg = out[i]
        next_seg = out[i + 1]

        prev_spk = prev_seg.get("speaker_id")
        curr_spk = curr_seg.get("speaker_id")
        next_spk = next_seg.get("speaker_id")

        prev_gap = max(0.0, float(curr_seg["start"]) - float(prev_seg["end"]))
        next_gap = max(0.0, float(next_seg["start"]) - float(curr_seg["end"]))

        if (
            prev_spk == next_spk
            and curr_spk != prev_spk
            and prev_gap <= bridge_gap_sec
            and next_gap <= bridge_gap_sec
        ):
            out[i]["speaker_id"] = prev_spk

    return out

# Amaç:
# Aynı speaker'a ait komşu diarization pencerelerini gerçek konuşma turn'lerine birleştirmek.
def _merge_speaker_windows_to_turns(
    diarized_windows: List[Dict],
    max_gap_sec: float = 0.35,
) -> List[Dict]:
    if not diarized_windows:
        return []

    windows = sorted(
        [
            {
                "start": float(w["start"]),
                "end": float(w["end"]),
                "speaker_id": str(w["speaker_id"]),
            }
            for w in diarized_windows
        ],
        key=lambda x: x["start"],
    )

    merged = [dict(windows[0])]

    for w in windows[1:]:
        prev = merged[-1]

        same_speaker = prev["speaker_id"] == w["speaker_id"]
        gap = w["start"] - prev["end"]

        # Amaç:
        # Aynı speaker ve aradaki boşluk küçükse aynı turn kabul et.
        if same_speaker and gap <= max_gap_sec:
            prev["end"] = max(prev["end"], w["end"])
        else:
            merged.append(dict(w))

    return merged

def assign_speakers_to_segments(
    text_segments: List[Dict],
    diarized_windows: List[Dict],
    min_overlap_ratio: float = 0.20,
    min_overlap_sec: float = 0.15,
) -> List[Dict]:
    """
    faster-whisper segmentlerini diarization pencerelerine bağlar.

    Mantık:
      1) Önce en yüksek overlap'i bul
      2) Overlap yeterliyse o speaker'ı ata
      3) Overlap zayıfsa en yakın pencerenin speaker'ını ata
      4) Son aşamada speaker sıçramalarını yumuşat
    """
    if not text_segments:
        return []

    if not diarized_windows:
        return [dict(seg, speaker_id="Speaker_0") for seg in text_segments]
    
    # Amaç:
    # Ham speaker window'ları önce gerçek turn bloklarına birleştir.
    diarized_windows = _merge_speaker_windows_to_turns(diarized_windows)

    enriched = []

    for seg in text_segments:
        seg_start = float(seg.get("start", 0.0) or 0.0)
        seg_end = float(seg.get("end", seg_start) or seg_start)
        seg_duration = max(0.0, seg_end - seg_start)

        best_speaker = None
        best_overlap = 0.0

        for win in diarized_windows:
            ov = _overlap(seg_start, seg_end, float(win["start"]), float(win["end"]))
            if ov > best_overlap:
                best_overlap = ov
                best_speaker = str(win["speaker_id"])

        # Amaç:
        # Overlap'in gerçekten anlamlı sayılması için minimum eşik belirlemek.
        required_overlap = max(min_overlap_sec, seg_duration * min_overlap_ratio)

        if best_speaker is None or best_overlap < required_overlap:
            best_speaker = _nearest_window_speaker(seg_start, seg_end, diarized_windows)

        item = dict(seg)
        item["speaker_id"] = best_speaker
        enriched.append(item)

    # Amaç:
    # İzole speaker sıçramalarını düzeltmek.
    enriched = _smooth_speaker_labels(enriched)

    return enriched