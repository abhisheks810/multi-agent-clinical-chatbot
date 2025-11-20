# app.py

from __future__ import annotations

import streamlit as st

from langgraph_rag.config import IngestionConfig, TableConfig
from langgraph_rag.graph.multi_agent_rwe_graph import build_rwe_multi_agent_graph
from langgraph_rag.observability.mlflow_utils import (
    init_mlflow,
    start_query_run,
    finish_query_run,
    log_feedback,
)

# ---------- 1. Initialize MLflow once ----------

# If you have a custom tracking URI, pass it here; otherwise it uses the default ./mlruns
init_mlflow(experiment_name="langgraph_rag_e2e")


# ---------- 2. Build / cache the graph & config ----------

@st.cache_resource
def get_graph_and_config():
    
    config = IngestionConfig(
        tables=[
            TableConfig(
                name="patients",
                path="data/patients.tsv",
                id_column="PATIENT_ID",  # adjust if your column differs
            )
        ],
        metadata_text_path="data/clinical_patient_meta.txt",
    )
    graph = build_rwe_multi_agent_graph(config, use_smart_analyst=True)
    return graph, config


graph, _ = get_graph_and_config()


# ---------- 3. Session state: store last run_id ----------

if "last_run_id" not in st.session_state:
    st.session_state["last_run_id"] = None


# ---------- 4. Streamlit UI: ask question and run graph ----------

st.title("RWE Multi-Agent RAG (MSK CHORD Cohort)")

query = st.text_input("Ask a question about the cohort")

if st.button("Run analysis") and query:
    # 4a. Start MLflow run
    run_id, start_time = start_query_run(query)
    st.session_state["last_run_id"] = run_id

    # 4b. Run the multi-agent graph
    with st.spinner("Thinking..."):
        result = graph.invoke({"user_query": query})

    answer = result.get("final_answer", "(No answer generated)")

    # 4c. Finish MLflow run (log latency + artifacts)
    finish_query_run(start_time, result)

    # 4d. Show answer
    st.markdown("### Answer")
    st.write(answer)


# ---------- 5. Feedback section (wired to MLflow) ----------

if st.session_state.get("last_run_id"):
    st.markdown("---")
    st.markdown("### Was this answer useful?")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("👍 Yes, useful"):
            log_feedback(
                run_id=st.session_state["last_run_id"],
                useful=True,
            )
            st.success("Thanks! Feedback recorded as useful ✅")

    with col2:
        if st.button("👎 No, not useful"):
            log_feedback(
                run_id=st.session_state["last_run_id"],
                useful=False,
            )
            st.info("Thanks! Feedback recorded as not useful 👎")

    # Optional: free-text feedback
    with st.expander("Optional: tell us why"):
        feedback_text = st.text_area("What made this answer useful or not?")
        if st.button("Submit feedback details"):
            if feedback_text.strip():
                log_feedback(
                    run_id=st.session_state["last_run_id"],
                    useful=True,  # or leave as last clicked; you can decide policy
                    comment=feedback_text.strip(),
                )
                st.success("Thanks for the detailed feedback 🙌")
            else:
                st.warning("Please enter some text before submitting.")
