import os
import re
from typing import Optional

from openai import OpenAI
from openai import APIError

from .config import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, ModelConfig


REASONING_RE = re.compile(r"<reasoning>\s*(.*?)\s*</reasoning>", re.IGNORECASE | re.DOTALL)
SQL_RE = re.compile(r"<sql>\s*(.*?)\s*</sql>", re.IGNORECASE | re.DOTALL)
SOLUTION_RE = re.compile(r"<solution>\s*(.*?)\s*</solution>", re.IGNORECASE | re.DOTALL)


SYSTEM_PROMPT = """你是一个专业的 StarRocks / MySQL 数据分析助手。
请按下面协议输出：
1. 必须先输出 <reasoning>...</reasoning>
2. 如果需要执行 SQL，输出 <sql>...</sql>
3. 如果已经可以直接回答，输出 <solution>...</solution>
4. 不要输出协议之外的内容
"""


def call_model(question: str, model: ModelConfig) -> str:
    api_key = (
        os.getenv(model.api_key_env)
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
        or "dummy"
    )
    print(
        "Calling model:",
        {
            "model_id": model.model_id,
            "provider_mode": model.provider_mode,
            "base_url": model.base_url,
            "model_name": model.model_name,
        },
    )
    client = OpenAI(api_key=api_key, base_url=model.base_url)
    try:
        response = client.chat.completions.create(
            model=model.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question.strip()},
            ],
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
            stream=False,
        )
        return response.choices[0].message.content.strip()
    except APIError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Model call failed for model_id={model.model_id}, "
            f"base_url={model.base_url}, model_name={model.model_name}: {exc}"
        ) from exc


def extract_reasoning(text: str) -> str:
    match = REASONING_RE.search(text or "")
    return match.group(1).strip() if match else ""


def extract_action_type(text: str) -> Optional[str]:
    if SQL_RE.search(text or ""):
        return "sql"
    if SOLUTION_RE.search(text or ""):
        return "solution"
    return None


def extract_sql(text: str) -> str:
    sql_match = SQL_RE.search(text or "")
    if sql_match:
        return sql_match.group(1).strip()
    solution_match = SOLUTION_RE.search(text or "")
    if solution_match:
        return solution_match.group(1).strip()
    return ""
