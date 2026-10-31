"""Shared defaults for the web application, CLI and Python facade."""

import os
from pathlib import Path
import sys

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else PROJECT_ROOT
load_dotenv(DATA_ROOT / ".env")


def env_int(name, default, minimum=1):
    value = int(os.getenv(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


DEFAULT_CLOUD_MODEL = "gemini-3-flash-preview"
DEFAULT_MODEL = os.getenv("SOLVER_MODEL", "qwen3.8")
DEFAULT_LOCAL = os.getenv("SOLVER_LOCAL", "true").lower() == "true"
DEFAULT_THINK = os.getenv("SOLVER_THINK", "false").lower() == "true"
MAX_BATCH_FILES = env_int("SOLVER_MAX_BATCH_FILES", 10)
MAX_FILE_SIZE = env_int("SOLVER_MAX_FILE_MB", 10) * 1024 * 1024
MAX_PENDING_JOBS = env_int("SOLVER_MAX_PENDING_JOBS", 4)
JOB_TTL_SECONDS = env_int("SOLVER_JOB_TTL_SECONDS", 3600, 60)
REQUEST_TIMEOUT_SECONDS = env_int("SOLVER_REQUEST_TIMEOUT_SECONDS", 180)
OLLAMA_MAX_SECONDS = env_int("SOLVER_OLLAMA_MAX_SECONDS", 600)


def solver_defaults():
    return {
        "llm_model_name": DEFAULT_MODEL,
        "local": DEFAULT_LOCAL,
        "think": DEFAULT_THINK,
    }
