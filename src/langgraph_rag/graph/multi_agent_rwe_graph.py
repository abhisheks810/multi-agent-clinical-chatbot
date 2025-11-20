# src/langgraph_rag/graph/multi_agent_rwe_graph.py

from __future__ import annotations

from typing import TypedDict, Dict, Any, List

from langgraph.graph import StateGraph, END

from langgraph_rag.llm import (
    generate_oncologist_view,
    generate_planner_plan,
    generate_writer_answer,
    generate_analyst_code,
)
from langgraph_rag.config import IngestionConfig
from langgraph_rag.tools.cohort_query import (
    load_tables_from_config,
    execute_cohort_filter_step,
    execute_feature_descriptives_step,
)


class ChatState(TypedDict, total=False):
    user_query: str

    oncologist_view: Dict[str, Any]
    plan: Dict[str, Any]
    execution_result: Dict[str, Any]
    final_answer: str

    # Optional debug / observability fields
    analyst_generated_code: str
    analyst_error: str

def build_rwe_multi_agent_graph(
    config: IngestionConfig,
    *,
    use_smart_analyst: bool = True,
):
    """
    Multi-agent graph:
      oncologist -> planner -> analyst -> writer -> END

    If use_smart_analyst is True, the Analyst will:
      - Use an LLM to generate Python code (run_analysis).
      - Execute it to obtain execution_result.

    Otherwise it can fall back to a simpler/dumb analyst implementation.
    """

    # --- Node 1: Oncologist ---

    def oncologist_node(state: ChatState) -> ChatState:
        user_query = state["user_query"]
        view = generate_oncologist_view(user_query)
        return {"oncologist_view": view}

    # --- Node 2: Planner ---

    def planner_node(state: ChatState) -> ChatState:
        user_query = state["user_query"]
        oncologist_view = state["oncologist_view"]
        plan = generate_planner_plan(user_query, oncologist_view)
        return {"plan": plan}

    # --- Node 3a: Smart Analyst (code-generating) ---

    def analyst_node_smart(state: ChatState) -> ChatState:
        user_query = state["user_query"]
        oncologist_view = state.get("oncologist_view", {})
        plan = state.get("plan", {})

        # 1) Generate the analysis code
        try:
            code = generate_analyst_code(
                user_query=user_query,
                oncologist_view=oncologist_view,
                plan=plan,
                config=config,
            )
        except Exception as e:
            execution_result: Dict[str, Any] = {
                "steps": [],
                "overall_status": "failed",
                "notes": "Code generation for Analyst failed.",
                "error": str(e),
            }
            return {
                "execution_result": execution_result,
                "analyst_error": str(e),
            }

        # 2) Execute the generated code in an isolated namespace
        local_ns: Dict[str, Any] = {}
        try:
            exec(code, {}, local_ns)
            if "run_analysis" not in local_ns or not callable(local_ns["run_analysis"]):
                raise RuntimeError("Generated code did not define run_analysis()")

            result = local_ns["run_analysis"]()
        except Exception as e:
            execution_result = {
                "steps": [],
                "overall_status": "failed",
                "notes": "Execution of generated analysis code failed.",
                "error": str(e),
                "generated_code": code[:4000],  # truncate to avoid huge payloads
            }
            return {
                "execution_result": execution_result,
                "analyst_generated_code": code[:4000],
                "analyst_error": str(e),
            }

        # 3) Normalize the result
        if not isinstance(result, dict):
            execution_result = {
                "steps": [],
                "overall_status": "failed",
                "notes": "Generated analysis code did not return a dict.",
                "raw_return": str(result),
                "generated_code": code[:4000],
            }
        else:
            steps = result.get("steps", [])
            overall_status = result.get("overall_status", "success")
            execution_result = {
                "steps": steps,
                "overall_status": overall_status,
                "notes": "Analyst executed a dynamically generated analysis script.",
            }

        return {
            "execution_result": execution_result,
            "analyst_generated_code": code[:4000],
        }

    # --- Node 3b: (Optional) Dumb Analyst fallback ---

    def analyst_node_dumb(state: ChatState) -> ChatState:
        # You can keep your previous Pandas-based executor here if you like.
        # For now, just mark it unimplemented.
        execution_result = {
            "steps": [],
            "overall_status": "failed",
            "notes": "Dumb analyst not implemented; enable use_smart_analyst=True.",
        }
        return {"execution_result": execution_result}

    # --- Node 4: Writer ---

    def writer_node(state: ChatState) -> ChatState:
        user_query = state["user_query"]
        oncologist_view = state.get("oncologist_view", {})
        plan = state.get("plan", {})
        execution_result = state.get("execution_result", {})

        final_answer = generate_writer_answer(
            user_query=user_query,
            oncologist_view=oncologist_view,
            plan=plan,
            execution_result=execution_result,
        )

        return {"final_answer": final_answer}

    # --- Assemble the graph ---

    graph = StateGraph(ChatState)
    graph.add_node("oncologist", oncologist_node)
    graph.add_node("planner", planner_node)

    if use_smart_analyst:
        graph.add_node("analyst", analyst_node_smart)
    else:
        graph.add_node("analyst", analyst_node_dumb)

    graph.add_node("writer", writer_node)

    graph.set_entry_point("oncologist")
    graph.add_edge("oncologist", "planner")
    graph.add_edge("planner", "analyst")
    graph.add_edge("analyst", "writer")
    graph.add_edge("writer", END)

    return graph.compile()
