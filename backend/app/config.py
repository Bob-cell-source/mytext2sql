import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    label: str
    provider_mode: str
    base_url: str
    model_name: str
    api_key_env: str


def _load_dotenv() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env_path = repo_root / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv()


def _required_env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "closed-qwen": ModelConfig(
        model_id="closed-qwen",
        label="闭源模型 / Qwen Compatible",
        provider_mode="closed",
        base_url=_required_env("CLOSED_MODEL_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        model_name=_required_env("CLOSED_MODEL_NAME", "qwen-plus"),
        api_key_env="CLOSED_MODEL_API_KEY",
    ),
    "open-vllm": ModelConfig(
        model_id="open-vllm",
        label="开源模型 / vLLM Local",
        provider_mode="open",
        base_url=_required_env("OPEN_MODEL_BASE_URL", "http://127.0.0.1:8000/v1"),
        model_name=_required_env("OPEN_MODEL_NAME", "grpo-local"),
        api_key_env="OPEN_MODEL_API_KEY",
    ),
}


APP_NAME = "Text2SQL Playground API"
APP_VERSION = "0.1.0"
USE_FINAL_CODE_FRAMEWORK = os.getenv("USE_FINAL_CODE_FRAMEWORK", "true").lower() == "true"
DEFAULT_TEMPERATURE = float(os.getenv("TEXT2SQL_TEMPERATURE", "0.0"))
DEFAULT_MAX_TOKENS = int(os.getenv("TEXT2SQL_MAX_TOKENS", "1024"))

DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "")
