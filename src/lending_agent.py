from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage

from src.model_inference import predict_risk_score
from src.rag_pipeline import build_policy_query, get_policy_context


def _format_policy_exception_guidance(policy_context: str) -> str:
    if not policy_context.strip():
        return "No relevant policy exceptions were retrieved."
    return policy_context


def _build_llm():
    provider = os.getenv("LENDING_AGENT_PROVIDER", "").strip().lower()

    if provider in {"groq", ""} and os.getenv("GROQ_API_KEY"):
        from langchain_groq import ChatGroq

        model_name = os.getenv("LENDING_AGENT_MODEL", "llama-3.3-70b-versatile")
        return ChatGroq(model=model_name, temperature=0)

    if provider in {"openai", ""} and os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI

        model_name = os.getenv("LENDING_AGENT_MODEL", "gpt-4o")
        return ChatOpenAI(model=model_name, temperature=0)

    if provider in {"anthropic", ""} and os.getenv("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic

        model_name = os.getenv("LENDING_AGENT_MODEL", "claude-3-5-sonnet-latest")
        return ChatAnthropic(model=model_name, temperature=0)

    raise RuntimeError(
        "No supported LLM provider is configured. Set GROQ_API_KEY, OPENAI_API_KEY, or "
        "ANTHROPIC_API_KEY, and optionally LENDING_AGENT_PROVIDER / LENDING_AGENT_MODEL."
    )


def _build_fallback_verdict(
    borrower_profile: Dict[str, Any],
    prediction: Dict[str, Any],
    policy_context: str,
) -> Dict[str, Any]:
    risk_score = prediction["risk_score"]
    risk_band = prediction["risk_band"]
    risk_factors = prediction["risk_factors"]

    if risk_band == "High":
        verdict = "Conditional Review"
        summary = (
            "The borrower is classified as high risk by the ML model, so the application "
            "should not move to straight-through approval. Policy guidance was retrieved to "
            "check whether compensating controls such as collateral, tighter approval thresholds, "
            "or exception-handling rules might justify a manual override."
        )
    else:
        verdict = "Pre-Approve"
        summary = (
            "The borrower is classified as lower risk by the ML model. No strong exception "
            "signals were required, so the application can proceed toward pre-approval subject "
            "to standard underwriting checks."
        )

    return {
        "final_verdict": verdict,
        "risk_band": risk_band,
        "risk_score": risk_score,
        "model_name": prediction["model_name"],
        "reasoning": summary,
        "risk_factors": risk_factors,
        "policy_context": policy_context,
        "borrower_profile": borrower_profile,
        "decision_source": "fallback",
    }


class SimpleConversationBufferMemory:
    """
    Small local replacement for ConversationBufferMemory to avoid version-specific
    memory dependencies on deployment targets.
    """

    def __init__(self) -> None:
        self.chat_history: List[Any] = []

    def load_memory_variables(self, _: Dict[str, Any]) -> Dict[str, Any]:
        return {"chat_history": self.chat_history}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        question = inputs.get("question")
        answer = outputs.get("answer")
        if question:
            self.chat_history.append(HumanMessage(content=str(question)))
        if answer:
            self.chat_history.append(AIMessage(content=str(answer)))


def _get_memory():
    return SimpleConversationBufferMemory()


class LendingDecisionState(TypedDict, total=False):
    borrower_profile: Dict[str, Any]
    model: Optional[Any]
    model_name: Optional[str]
    model_path: Optional[str]
    prediction: Dict[str, Any]
    policy_query: str
    policy_context: str
    reasoning: str
    decision_source: str
    llm_error: str


def _score_borrower_node(state: LendingDecisionState) -> LendingDecisionState:
    borrower_profile = state["borrower_profile"]
    prediction = predict_risk_score(
        borrower_profile=borrower_profile,
        model=state.get("model"),
        model_name=state.get("model_name"),
        model_path=state.get("model_path"),
    )
    policy_query = build_policy_query(
        borrower_profile=borrower_profile,
        risk_score=prediction["risk_score"],
    )
    return {
        "prediction": prediction,
        "policy_query": policy_query,
        "policy_context": "",
    }


def _should_lookup_policy(state: LendingDecisionState) -> str:
    prediction = state["prediction"]
    return "policy_lookup" if prediction["risk_band"] == "High" else "draft_recommendation"


def _policy_lookup_node(state: LendingDecisionState) -> LendingDecisionState:
    try:
        policy_context = get_policy_context(state["policy_query"])
    except Exception as exc:
        policy_context = f"Policy lookup unavailable: {exc}"
    return {"policy_context": policy_context}


def _draft_recommendation_node(state: LendingDecisionState) -> LendingDecisionState:
    prediction = state["prediction"]
    policy_context = _format_policy_exception_guidance(state.get("policy_context", ""))

    try:
        from langchain_core.prompts import ChatPromptTemplate

        llm = _build_llm()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an underwriting decision agent operating inside a LangGraph workflow. "
                    "Use the scored borrower profile, risk factors, and policy context already provided. "
                    "Return a concise but complete recommendation with these headings: "
                    "`Final Lending Verdict`, `Risk Score`, `Reasoning`, and `Policy Citation Summary`. "
                    "Do not claim to have used tools. Ground every claim in the provided inputs.",
                ),
                (
                    "human",
                    "Borrower profile:\n{borrower_profile}\n\n"
                    "Model prediction:\n{prediction}\n\n"
                    "Policy query:\n{policy_query}\n\n"
                    "Policy context:\n{policy_context}",
                ),
            ]
        )
        chain = prompt | llm
        response = chain.invoke(
            {
                "borrower_profile": json.dumps(state["borrower_profile"], default=str, indent=2),
                "prediction": json.dumps(prediction, default=str, indent=2),
                "policy_query": state["policy_query"],
                "policy_context": policy_context,
            }
        )
        reasoning = getattr(response, "content", str(response))
        return {
            "reasoning": reasoning,
            "policy_context": policy_context,
            "decision_source": "langgraph",
        }
    except Exception as exc:
        fallback = _build_fallback_verdict(
            borrower_profile=state["borrower_profile"],
            prediction=prediction,
            policy_context=policy_context,
        )
        return {
            "reasoning": fallback["reasoning"],
            "policy_context": fallback["policy_context"],
            "decision_source": fallback["decision_source"],
            "llm_error": str(exc),
        }


def _build_lending_graph():
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(LendingDecisionState)
    graph.add_node("score_borrower", _score_borrower_node)
    graph.add_node("policy_lookup", _policy_lookup_node)
    graph.add_node("draft_recommendation", _draft_recommendation_node)

    graph.add_edge(START, "score_borrower")
    graph.add_conditional_edges(
        "score_borrower",
        _should_lookup_policy,
        {
            "policy_lookup": "policy_lookup",
            "draft_recommendation": "draft_recommendation",
        },
    )
    graph.add_edge("policy_lookup", "draft_recommendation")
    graph.add_edge("draft_recommendation", END)
    return graph.compile()


def answer_follow_up_question(
    question: str,
    borrower_profile: Dict[str, Any],
    lending_decision: Dict[str, Any],
    memory: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Answer a follow-up question about the latest borrower decision while preserving conversation history.
    """
    active_memory = memory or _get_memory()
    memory_variables = active_memory.load_memory_variables({})
    chat_history = memory_variables.get("chat_history", [])

    policy_summary = lending_decision.get("policy_context", "No policy citations were retrieved.")
    risk_factors = (
        "\n".join(f"- {factor}" for factor in lending_decision.get("risk_factors", []))
        or "- No extra risk factors recorded."
    )

    try:
        from langchain_core.prompts import ChatPromptTemplate

        llm = _build_llm()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a lending risk copilot answering follow-up questions about one borrower. "
                    "Use the existing decision, borrower profile, risk factors, and policy summary. "
                    "Explain the answer in very simple, plain language. "
                    "Start with a direct answer in one sentence, then give short sections named "
                    "`Why`, `What mattered most`, and `What could change the decision`. "
                    "If the question is about a declined or risky borrower, clearly say what would need "
                    "to improve. Do not contradict the recorded verdict.",
                ),
                ("placeholder", "{chat_history}"),
                (
                    "human",
                    "Borrower profile:\n{borrower_profile}\n\n"
                    "Recorded decision:\n{decision}\n\n"
                    "Risk factors:\n{risk_factors}\n\n"
                    "Policy summary:\n{policy_summary}\n\n"
                    "Follow-up question: {question}",
                ),
            ]
        )
        chain = prompt | llm
        response = chain.invoke(
            {
                "chat_history": chat_history,
                "borrower_profile": json.dumps(borrower_profile, default=str, indent=2),
                "decision": json.dumps(lending_decision, default=str, indent=2),
                "risk_factors": risk_factors,
                "policy_summary": policy_summary,
                "question": question,
            }
        )
        answer = getattr(response, "content", str(response))
    except Exception:
        answer = (
            f"Direct answer: The current borrower decision is `{lending_decision.get('final_verdict', 'Unavailable')}` "
            f"with a risk score of {lending_decision.get('risk_score', 0):.2f}.\n\n"
            f"Why:\n{', '.join(lending_decision.get('risk_factors', [])) or 'No extra risk signals were recorded.'}\n\n"
            f"What mattered most:\nThe decision used the borrower profile, model score, and policy guidance.\n\n"
            f"What could change the decision:\n{policy_summary}"
        )

    active_memory.save_context({"question": question}, {"answer": answer})
    return {"answer": answer, "memory": active_memory}


def run_agentic_lending_decision(
    borrower_profile: Dict[str, Any],
    model: Optional[Any] = None,
    model_name: Optional[str] = None,
    model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Produce a final lending verdict using a LangGraph workflow over ML outputs and policy retrieval.
    """
    graph = _build_lending_graph()
    final_state = graph.invoke(
        {
            "borrower_profile": borrower_profile,
            "model": model,
            "model_name": model_name,
            "model_path": model_path,
        }
    )

    prediction = final_state["prediction"]
    policy_context = _format_policy_exception_guidance(final_state.get("policy_context", ""))
    decision_source = final_state.get("decision_source", "langgraph")

    if decision_source == "fallback":
        return {
            **_build_fallback_verdict(
                borrower_profile=borrower_profile,
                prediction=prediction,
                policy_context=policy_context,
            ),
            "policy_query": final_state["policy_query"],
        }

    return {
        "final_verdict": "LangGraph Recommendation",
        "risk_band": prediction["risk_band"],
        "risk_score": prediction["risk_score"],
        "model_name": prediction["model_name"],
        "reasoning": final_state["reasoning"],
        "risk_factors": prediction["risk_factors"],
        "policy_query": final_state["policy_query"],
        "policy_context": policy_context,
        "borrower_profile": borrower_profile,
        "decision_source": decision_source,
    }
