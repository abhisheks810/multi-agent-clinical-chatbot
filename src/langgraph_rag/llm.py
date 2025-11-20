# src/langgraph_rag/llm.py

from __future__ import annotations

import json
import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI
from .config import IngestionConfig
from .prompts import (
    get_oncologist_system_prompt,
    get_oncologist_prompt_id,
    get_planner_system_prompt,
    get_planner_prompt_id,
    get_writer_system_prompt,
    get_writer_prompt_id,
    ANALYST_CODE_SYSTEM_PROMPT,
)

load_dotenv()

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
DEFAULT_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")


def _chat_completion(messages: List[Dict[str, str]], model: str | None = None) -> str:
    """Small helper to hit the OpenAI chat completion endpoint and return content."""
    if model is None:
        model = DEFAULT_CHAT_MODEL

    resp = _client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def _safe_json_parse(text: str) -> Dict[str, Any]:
    """Try to parse JSON; if it fails, wrap the raw text."""
    try:
        return json.loads(text)
    except Exception:
        return {"raw_text": text, "parse_error": True}


# -------- Oncologist: interpret query and produce oncologist_view JSON --------

def generate_oncologist_view(
    user_query: str,
    model: str | None = None,
) -> Dict[str, Any]:
    system_prompt = get_oncologist_system_prompt()
    prompt_id = get_oncologist_prompt_id()

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "User question:\n"
                f"{user_query}\n\n"
                "Remember: respond ONLY with a JSON object following the specified schema."
            ),
        },
    ]

    raw = _chat_completion(messages, model=model)
    view = _safe_json_parse(raw)
    view["_prompt_id"] = prompt_id  # useful for MLflow later
    return view


# -------- Planner: build execution plan JSON --------

from pathlib import Path

METADATA_PATH = (
    Path(__file__)
    .resolve()
    .parents[2]  # adjust if needed
    / "data"
    / "msk_chord.txt"
)

def generate_planner_plan(user_query: str, oncologist_view: Dict[str, Any], model: str | None = None) -> Dict[str, Any]:
    system_prompt = get_planner_system_prompt()
    prompt_id = get_planner_prompt_id()

    oncologist_view_json = json.dumps(oncologist_view, indent=2)

    try:
        metadata_text = METADATA_PATH.read_text()
    except FileNotFoundError:
        metadata_text = "Dataset metadata file not found."

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Original user question:\n"
                f"{user_query}\n\n"
                "Oncologist agent interpretation (JSON):\n"
                f"{oncologist_view_json}\n\n"
                "Dataset metadata (table and column information):\n"
                f"{metadata_text}\n\n"
                "Using ONLY the table and column names mentioned in this metadata, "
                "produce the plan JSON following the specified schema. "
                "Do not invent new table or column names."
            ),
        },
    ]

    raw = _chat_completion(messages, model=model)
    plan = _safe_json_parse(raw)
    plan["_prompt_id"] = prompt_id
    return plan

# --------- Analyst: execute plan steps (code generation) --------
def generate_analyst_code(
    user_query: str,
    oncologist_view: Dict[str, Any],
    plan: Dict[str, Any],
    config: IngestionConfig,
    model: str | None = None,
) -> str:
    """
    Use a (possibly code-specialized) LLM to generate a Python function
    `run_analysis()` that implements the Planner's execution plan.

    The generated code must follow the contract defined in
    ANALYST_CODE_SYSTEM_PROMPT.
    """

    oncologist_json = json.dumps(oncologist_view, indent=2)
    plan_json = json.dumps(plan, indent=2)

    # Build a small schema of available tables from the ingestion config
    table_infos = []
    for t in config.tables:
        # We assume t has .name (logical name) and .path (TSV path)
        table_infos.append(f"- logical_name: {t.name}, path: {t.path}")
    tables_text = "\n".join(table_infos)

    user_content = f"""
        User question:
        {user_query}

        Oncologist interpretation (JSON):
        {oncologist_json}

        Planner execution plan (JSON):
        {plan_json}

        Available tables (logical_name and TSV path):
        {tables_text}

        REQUIREMENTS RECAP:
        - Implement EXACTLY one function: `def run_analysis():`
        - Use pandas to load the TSV files above.
        - Follow the steps in plan["execution_plan"] in order.
        - For a step with tool == "cohort_sql":
            - Filter the DataFrame using the `filter` dict (keys are column names, values are filter values).
            - Record a `cohort_summary` dict with at least:
                - "table_used"
                - "row_count"
                - optionally "patient_count" if a patient ID column is available.
        - For a step with tool == "feature_descriptives":
            - Compute numeric statistics (mean, median, std, min, max, IQR) for the specified feature.
            - Put those in a `metrics` dict.

        - ALWAYS build a `result` dict:
            result = {{
                "steps": [ step_result_1, step_result_2, ... ],
                "overall_status": "success" or "partial_success" or "failed",
            }}
        and `return result` at the end of run_analysis().
        """

    messages = [
        {"role": "system", "content": ANALYST_CODE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    # Allow overriding with a code-optimized model via env; fallback to your default chat model
    code_model = model or os.getenv("OPENAI_CODE_MODEL") or os.getenv("OPENAI_MODEL")  # or DEFAULT_CHAT_MODEL
    raw = _chat_completion(messages, model=code_model)

    # The model must return pure Python code (no backticks)
    return raw.strip()

# -------- Writer: final user-facing answer (markdown) --------

def generate_writer_answer(
    user_query: str,
    oncologist_view: Dict[str, Any],
    plan: Dict[str, Any],
    execution_result: Dict[str, Any],
    model: str | None = None,
) -> str:
    system_prompt = get_writer_system_prompt()
    prompt_id = get_writer_prompt_id()

    oncologist_json = json.dumps(oncologist_view, indent=2)
    plan_json = json.dumps(plan, indent=2)
    exec_json = json.dumps(execution_result, indent=2)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "User question:\n"
                f"{user_query}\n\n"
                "Oncologist interpretation (JSON):\n"
                f"{oncologist_json}\n\n"
                "Planner execution plan (JSON):\n"
                f"{plan_json}\n\n"
                "Analyst execution_result (JSON):\n"
                f"{exec_json}\n\n"
                "Now write the final user-facing answer."
            ),
        },
    ]

    answer = _chat_completion(messages, model=model)
    # you might later include prompt_id in MLflow; for now we just return the answer
    return answer
