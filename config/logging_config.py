"""
Logging configuration using loguru.
Provides structured, colored, file-rotating logs.
"""

import sys
from pathlib import Path
from loguru import logger
from config.settings import settings


def setup_logging() -> None:
    """Configure loguru with console + file handlers.

    THE single logging setup: entrypoints must call this and must not
    install their own handlers (two divergent configs is how the live
    entrypoint previously bypassed the project logging standard).

    - Console: human-readable, UTC timestamps (explicit ``Z`` suffix).
    - File: JSON records (``serialize=True``) so the audit trail is
      machine-parseable, rotating, 30-day retention.
    """
    # Remove default handler
    logger.remove()

    # ─── Console handler (colored, UTC) ───────────────────────────────────────
    logger.add(
        sys.stdout,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss!UTC}Z</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # ─── File handler (JSON, rotating) ────────────────────────────────────────
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_path,
        level=settings.log_level,
        serialize=True,        # structured JSON records (tz-aware ISO 8601)
        rotation="10 MB",      # Rotate after 10MB
        retention="30 days",   # Keep 30 days of logs
        compression="zip",     # Compress old logs
        enqueue=True,          # Thread-safe async logging
    )

    logger.info("Logging configured — level: {}", settings.log_level)
