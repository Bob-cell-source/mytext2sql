import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict

from sql_exe import execute_sql_with_pymysql

from agent_rl.result_matcher import build_result_fingerprint
from agent_rl.schemas import Observation


READ_ONLY_PATTERN = re.compile(r"^\s*(select|with)\b", re.IGNORECASE | re.DOTALL)
FORBIDDEN_PATTERN = re.compile(r"\b(insert|update|delete|drop|alter|truncate|create|grant|revoke)\b", re.IGNORECASE)


class SQLEnvironment:
    def __init__(self, work_dir: str = "./tmp_agent_rl", max_preview_rows: int = 5):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.max_preview_rows = max_preview_rows
        self.executor = execute_sql_with_pymysql()

    @staticmethod
    def _clean_sql(sql_text: str) -> str:
        sql_text = sql_text.strip()
        sql_text = re.sub(r"^```[sS][qQ][lL]\s*\n?", "", sql_text)
        sql_text = re.sub(r"^```\s*\n?", "", sql_text)
        sql_text = re.sub(r"\n?```\s*$", "", sql_text)
        return sql_text.strip()

    @staticmethod
    def _is_read_only_query(sql_text: str) -> bool:
        if FORBIDDEN_PATTERN.search(sql_text):
            return False
        return bool(READ_ONLY_PATTERN.match(sql_text))

    @staticmethod
    def _db_config() -> Dict[str, Any]:
        return {
            "host": os.getenv("DB_HOST", "127.0.0.1"),
            "user": os.getenv("DB_USER", "root"),
            "password": os.getenv("DB_PASSWORD", ""),
            "db": os.getenv("DB_NAME", "TGAC"),
            "port": int(os.getenv("DB_PORT", "9030")),
        }

    def _execute_internal(self, sql_text: str, turns_left: int, sql_id: str = "") -> tuple[Observation, list[dict]]:
        sql_text = self._clean_sql(sql_text)
        if not sql_text.endswith(";"):
            sql_text += ";"

        if not self._is_read_only_query(sql_text):
            return {
                "status": "error",
                "error_message": "Only a single read-only SELECT/WITH query is allowed.",
                "columns": [],
                "sample_rows": [],
                "row_count": 0,
                "fingerprint": "",
                "turns_left": turns_left,
            }, []

        request_id = sql_id or f"agent_{uuid.uuid4().hex[:8]}"
        tmp_in = self.work_dir / f"{request_id}_in.json"
        tmp_out = self.work_dir / f"{request_id}_out.json"

        payload = [{"sql_id": request_id, "sql": sql_text}]
        tmp_in.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        try:
            self.executor.execute_sql_with_pymysql(str(tmp_in), str(tmp_out), db_config=self._db_config())
            if not tmp_out.exists():
                return {
                    "status": "error",
                    "error_message": "SQL executor did not produce an output file.",
                    "columns": [],
                    "sample_rows": [],
                    "row_count": 0,
                    "fingerprint": "",
                    "turns_left": turns_left,
                }, []
            raw = json.loads(tmp_out.read_text(encoding="utf-8"))
        finally:
            try:
                tmp_in.unlink(missing_ok=True)
            except Exception:
                pass
            try:
                tmp_out.unlink(missing_ok=True)
            except Exception:
                pass

        record = raw[0] if raw else {}
        status = record.get("status", "error")
        result_rows = record.get("result", []) if status in ("success", "empty", "placeholder") else []
        columns = list(result_rows[0].keys()) if result_rows and isinstance(result_rows[0], dict) else []
        normalized_status = "success" if result_rows else "empty"
        if status == "error":
            normalized_status = "error"
        sample_rows = result_rows[: self.max_preview_rows]
        fingerprint = build_result_fingerprint(result_rows)

        observation = {
            "status": normalized_status,
            "error_message": record.get("error_message", ""),
            "columns": columns,
            "sample_rows": sample_rows,
            "row_count": len(result_rows),
            "fingerprint": fingerprint,
            "turns_left": turns_left,
        }
        return observation, result_rows

    def execute(self, sql_text: str, turns_left: int, sql_id: str = "") -> Observation:
        observation, _ = self._execute_internal(sql_text, turns_left=turns_left, sql_id=sql_id)
        return observation

    def execute_with_rows(self, sql_text: str, turns_left: int, sql_id: str = "") -> tuple[Observation, list[dict]]:
        return self._execute_internal(sql_text, turns_left=turns_left, sql_id=sql_id)
