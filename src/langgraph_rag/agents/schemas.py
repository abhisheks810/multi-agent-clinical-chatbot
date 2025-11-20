# src/langgraph_rag/agents/schemas.py

from __future__ import annotations

from typing import List, Dict, TypedDict, Any

class OncologistView(TypedDict, total=False):
    """
    Structured output from the Oncologist agent.
    """
    disease_area: str                     # e.g. "Non-Small Cell Lung Cancer"
    diagnosis_terms: List[str]            # e.g. ["NSCLC", "Stage 4"]
    target_features: List[str]            # e.g. ["TMB (nonsynonymous)", "Overall Survival (Months)"]
    question_intent: List[str]            # e.g. ["patient_counts", "descriptive_stats"]
    can_be_answered: bool                 # true/false based on dataset assumptions
    limitations: List[str]                # human-readable limitations
    notes_for_planner: str                # guidance for planner (tables/columns to consider)


class PlanStep(TypedDict, total=False):
    """
    A single step in the Planner's execution plan.
    """
    step_id: int
    name: str                             # human-readable name
    tool: str                             # e.g. "cohort_sql", "feature_descriptives", "vector_search"
    filter: Dict[str, Any]                # filter spec for cohort queries
    feature: str | None                   # e.g. "TMB (nonsynonymous)" for descriptives
    columns_used: List[str]
    tables_used: List[str]
    comment: str


class PlannerPlan(TypedDict, total=False):
    """
    Structured output from the Planner agent.
    """
    analysis_type: List[str]              # e.g. ["filter_patients", "patient_descriptives"]
    execution_plan: List[PlanStep]
    what_we_do_summary: str
    what_cannot_be_done: List[str]


class ChatState(TypedDict, total=False):
    """
    Shared state for the LangGraph.
    """
    user_query: str

    oncologist_view: OncologistView
    plan: PlannerPlan

    # We'll add these later for Analyst and Writer:
    # execution_result: Dict[str, Any]
    # final_answer: str
