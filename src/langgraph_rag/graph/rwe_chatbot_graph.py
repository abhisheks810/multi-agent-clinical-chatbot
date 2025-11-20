# src/langgraph_rag/graph/rwe_chatbot_graph.py

from __future__ import annotations

from typing import List, Dict, TypedDict

from langgraph.graph import StateGraph, END

from langgraph_rag.config import IngestionConfig
from langgraph_rag.tools.vector_search import vector_search_patients
from langgraph_rag.llm import generate_oncologist_answer


class ChatState(TypedDict, total=False):
    question: str
    answer: str
    context: List[Dict]


def build_rwe_oncologist_graph(config: IngestionConfig):
    def oncologist_node(state: ChatState) -> ChatState:
        question = state["question"]

        retrieval_results = vector_search_patients(config, question, k=5)
        answer = generate_oncologist_answer(question, retrieval_results)

        return {
            "question": question,
            "answer": answer,
            "context": retrieval_results,
        }

    graph = StateGraph(ChatState)
    graph.add_node("oncologist", oncologist_node)
    graph.set_entry_point("oncologist")
    graph.add_edge("oncologist", END)

    return graph.compile()
