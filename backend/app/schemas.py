from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    model_id: str
    mode: str = "intelligent"
    execute_sql: bool = True
    knowledge: str = ""
    table_list: List[str] = []


class ModelOption(BaseModel):
    model_id: str
    label: str
    provider_mode: str
    base_url: Optional[str] = None
    model_name: Optional[str] = None


class ExecutionPayload(BaseModel):
    status: str
    row_count: int = 0
    columns: List[str] = []
    rows: List[Dict[str, Any]] = []
    error_message: str = ""
    elapsed_ms: int = 0


class AskResponse(BaseModel):
    model_id: str
    provider_mode: str
    raw_output: str
    reasoning: str
    action_type: Optional[str]
    sql: str
    execution: Optional[ExecutionPayload] = None
    total_elapsed_ms: int = 0
    model_elapsed_ms: int = 0


class ErrorResponse(BaseModel):
    error: str
    model_id: Optional[str] = None
    provider_mode: Optional[str] = None
    base_url: Optional[str] = None
    model_name: Optional[str] = None
