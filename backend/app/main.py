from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import APIError
import time

from .config import APP_NAME, APP_VERSION, USE_FINAL_CODE_FRAMEWORK
from .final_code_adapter import run_with_final_code
from .llm_client import call_model, extract_action_type, extract_reasoning, extract_sql
from .model_router import get_model_config, list_model_options
from .schemas import AskRequest, AskResponse, ErrorResponse, ExecutionPayload, ModelOption
from .sql_executor import execute_sql


app = FastAPI(title=APP_NAME, version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/models", response_model=list[ModelOption])
def list_models() -> list[ModelOption]:
    return list_model_options()


@app.post("/ask", response_model=AskResponse, responses={502: {"model": ErrorResponse}})
def ask(request: AskRequest) -> AskResponse:
    total_started = time.perf_counter()
    try:
        model = get_model_config(request.model_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    use_final_code = request.mode == "intelligent" or (
        request.mode not in {"native", "intelligent"} and USE_FINAL_CODE_FRAMEWORK
    )

    if use_final_code:
        try:
            result = run_with_final_code(
                question=request.question,
                model=model,
                knowledge=request.knowledge,
                table_list=request.table_list,
            )
            execution = None
            if result.get("execution"):
                execution = ExecutionPayload(**result["execution"])
            return AskResponse(
                model_id=model.model_id,
                provider_mode=model.provider_mode,
                raw_output=result["raw_output"],
                reasoning=result["reasoning"],
                action_type=result["action_type"],
                sql=result["sql"],
                execution=execution,
                total_elapsed_ms=result["total_elapsed_ms"],
                model_elapsed_ms=result["model_elapsed_ms"],
            )
        except Exception as exc:
            detail = {
                "error": f"final_code framework failed: {exc}",
                "model_id": model.model_id,
                "provider_mode": model.provider_mode,
                "base_url": model.base_url,
                "model_name": model.model_name,
            }
            raise HTTPException(status_code=502, detail=detail) from exc

    try:
        model_started = time.perf_counter()
        raw_output = call_model(request.question, model)
        model_elapsed_ms = int((time.perf_counter() - model_started) * 1000)
    except APIError as exc:
        detail = {
            "error": f"Model API request failed: {exc}",
            "model_id": model.model_id,
            "provider_mode": model.provider_mode,
            "base_url": model.base_url,
            "model_name": model.model_name,
        }
        raise HTTPException(status_code=502, detail=detail) from exc
    except Exception as exc:
        detail = {
            "error": str(exc),
            "model_id": model.model_id,
            "provider_mode": model.provider_mode,
            "base_url": model.base_url,
            "model_name": model.model_name,
        }
        raise HTTPException(status_code=502, detail=detail) from exc
    reasoning = extract_reasoning(raw_output)
    action_type = extract_action_type(raw_output)
    sql_text = extract_sql(raw_output)

    execution = None
    if request.execute_sql and action_type == "sql" and sql_text:
        execution = ExecutionPayload(**execute_sql(sql_text))

    return AskResponse(
        model_id=model.model_id,
        provider_mode=model.provider_mode,
        raw_output=raw_output,
        reasoning=reasoning,
        action_type=action_type,
        sql=sql_text,
        execution=execution,
        total_elapsed_ms=int((time.perf_counter() - total_started) * 1000),
        model_elapsed_ms=model_elapsed_ms,
    )
