# Amaç:
# Proje genelinde ortak logging ayarı sağlamak.
# LOG_LEVEL env ile seviye kontrol edilebilir. Varsayılan INFO.

from __future__ import annotations

import logging
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

    _LOGGING_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)