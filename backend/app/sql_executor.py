from datetime import date, datetime
from decimal import Decimal
import time
from typing import Any, Dict, List

import pymysql

from .config import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def execute_sql(sql_text: str) -> Dict[str, Any]:
    conn = None
    started = time.perf_counter()
    try:
        conn = pymysql.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            cursorclass=pymysql.cursors.DictCursor,
        )
        with conn.cursor() as cursor:
            cursor.execute(sql_text)
            rows: List[Dict[str, Any]] = cursor.fetchall()
            normalized_rows = [{k: _jsonable(v) for k, v in row.items()} for row in rows]
            return {
                "status": "success" if normalized_rows else "empty",
                "row_count": len(normalized_rows),
                "columns": list(normalized_rows[0].keys()) if normalized_rows else [],
                "rows": normalized_rows[:100],
                "error_message": "",
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
            }
    except Exception as exc:
        message = str(exc)
        if "Access denied for user 'root'@'localhost'" in message:
            message += "；这是 MySQL 账号鉴权失败，通常是 root 账号未开放密码登录、鉴权插件不匹配，或 .env 中 DB_USER / DB_PASSWORD / DB_HOST 配置不对。"
        return {
            "status": "error",
            "row_count": 0,
            "columns": [],
            "rows": [],
            "error_message": message,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }
    finally:
        if conn is not None:
            conn.close()
