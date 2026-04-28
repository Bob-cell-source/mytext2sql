import os
import time
import uuid
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

from .config import ModelConfig


_LOCK = Lock()
_RESOURCES: Optional[Dict[str, Any]] = None


def _candidate_file(*relative_paths: str) -> Optional[Path]:
    repo_root = Path(__file__).resolve().parents[2]
    for rel in relative_paths:
        path = repo_root / rel
        if path.exists():
            return path
    return None


def _load_resources() -> Dict[str, Any]:
    global _RESOURCES
    if _RESOURCES is not None:
        return _RESOURCES

    with _LOCK:
        if _RESOURCES is not None:
            return _RESOURCES

        import final_code as fc  # pylint: disable=import-outside-toplevel

        fc.ENABLE_QUESTION_REWRITE = False
        fc.REQUIRE_USER_CONFIRMATION = False

        schema_path = _candidate_file("schema.json", "data/schema_with.json")
        if schema_path is None:
            raise FileNotFoundError("No schema file found for final_code adapter.")
        schema_index = fc.build_schema_index(fc.load_json(str(schema_path)))

        golden_path = _candidate_file("golden_sql_marked.json", "final_dataset.json")
        golden_examples = fc.load_json(str(golden_path)) if golden_path else []

        vector_db = None
        vector_path = _candidate_file("vectordatabase.pkl", "vector_database.pkl")
        if vector_path is not None:
            try:
                vector_db = fc.SQLVectorDatabase()
                vector_db.load_database(str(vector_path))
            except Exception:
                vector_db = None

        _RESOURCES = {
            "fc": fc,
            "schema_index": schema_index,
            "golden_examples": golden_examples,
            "vector_db": vector_db,
        }
        return _RESOURCES


def _normalize_status(status: str) -> str:
    if status == "success_non_empty":
        return "success"
    if status == "success_empty":
        return "empty"
    return "error"


def run_with_final_code(
    question: str,
    model: ModelConfig,
    knowledge: str = "",
    table_list: Optional[list[str]] = None,
) -> Dict[str, Any]:
    resources = _load_resources()
    fc = resources["fc"]

    api_key = (
        os.getenv(model.api_key_env)
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
        or ""
    )
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    os.environ["DASHSCOPE_BASE_URL"] = model.base_url
    os.environ["LLM_MODEL"] = model.model_name

    item = {
        "sql_id": f"web_{uuid.uuid4().hex[:8]}",
        "question": question,
        "knowledge": knowledge or "",
        "table_list": table_list or [],
    }

    total_started = time.perf_counter()
    result = fc.generate_sql_for_item(
        item=item,
        schema_index=resources["schema_index"],
        golden_examples=resources["golden_examples"],
        vector_db=resources["vector_db"],
        prompt_log_dir=None,
    )
    total_elapsed_ms = int((time.perf_counter() - total_started) * 1000)

    return {
        "raw_output": result.get("sql", ""),
        "reasoning": (
            "当前结果由 final_code.py 框架生成。"
            "该链路会构建 schema/knowledge 上下文、加入 few-shot、"
            "尝试诊断 SQL 并基于执行反馈重试，而不是直接让模型自由假设表结构。"
        ),
        "action_type": "sql" if result.get("sql") else None,
        "sql": result.get("sql", ""),
        "execution": {
            "status": _normalize_status(result.get("status", "failed")),
            "row_count": result.get("row_count", 0) or 0,
            "columns": list(result["data"][0].keys()) if result.get("data") else [],
            "rows": (result.get("data") or [])[:100],
            "error_message": result.get("error", "") or "",
            "elapsed_ms": 0,
        },
        "total_elapsed_ms": total_elapsed_ms,
        "model_elapsed_ms": total_elapsed_ms,
    }
