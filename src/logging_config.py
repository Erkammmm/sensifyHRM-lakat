# Amaç:
# Proje genelinde ortak logging ayarı sağlamak.
# LOG_LEVEL env ile seviye kontrol edilebilir. Varsayılan INFO.

from __future__ import annotations

import logging
import warnings
import os
from typing import Optional


_LOGGING_CONFIGURED = False


def configure_logging(level: Optional[str] = None) -> None:
    global _LOGGING_CONFIGURED

    if _LOGGING_CONFIGURED:
        return

    log_level_name = (level or os.getenv("LOG_LEVEL", "INFO")).upper().strip()
    log_level = getattr(logging, log_level_name, logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    clean_terminal = os.getenv("SENSIFYHR_CLEAN_TERMINAL", "1").strip() == "1"

    if clean_terminal:
        # Proje içi detay modüller: terminali doldurmasın
        logging.getLogger("src.audio.text_analyzer").setLevel(logging.WARNING)
        logging.getLogger("src.audio.voice_analyzer").setLevel(logging.WARNING)
        logging.getLogger("src.vision.face_analyzer").setLevel(logging.WARNING)
        logging.getLogger("src.audio.audio_signal_fusion").setLevel(logging.WARNING)

        # Üçüncü parti loglar
        logging.getLogger("faster_whisper").setLevel(logging.WARNING)
        logging.getLogger("speechbrain").setLevel(logging.ERROR)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("google_genai").setLevel(logging.WARNING)

        # Warning temizliği
        warnings.filterwarnings(
            "ignore",
            message=r".*speechbrain\.pretrained.*deprecated.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*estimate.*is deprecated since version 0\.26.*",
            category=FutureWarning,
        )

    _LOGGING_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)