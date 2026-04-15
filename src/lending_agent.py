from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from src.model_inference import predict_risk_score
from src.rag_pipeline import build_policy_query, get_policy_context


def _borrower_profile_from_json(payload: str) -> Dict[str, Any]:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("Borrower payload must deserialize to a JSON object.")
    return data


def _format_policy_exception_guidance(policy_context: str) -> str:
    if not policy_context.strip():
        return "No relevant policy exceptions were retrieved."
    return policy_context


def _build_llm():
    provider = os.getenv("LENDING_AGENT_PROVIDER", "").strip().lower()

    if provider in {"openai", ""} and os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI

        model_name = os.getenv("LENDING_AGENT_MODEL", "gpt-4o")
        return ChatOpenAI(model=model_name, temperature=0)

    if provider in {"anthropic", ""} and os.getenv("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic

        model_name = os.getenv("LENDING_AGENT_MODEL", "claude-3-5-sonnet-latest")
        return ChatAnthropic(model=model_name, temperature=0)

    raise RuntimeError(
        "No supported LLM provider is configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY, "
        "and optionally LENDING_AGENT_PROVIDER / LENDING_AGENT_MODEL."
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


def _get_memory():
    from langchain.memory import ConversationBufferMemory

    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="question",
        output_key="answer",
    )


def build_lending_tools(
    model: Optional[Any] = None,
    model_name: Optional[str] = None,
    model_path: Optional[str] = None,
):
    from langchain_core.tools import tool

    @tool("predict_risk_score")
    def predict_risk_score_tool(borrower_payload_json: str) -> str:
        """
        Run the trained credit-risk model on a borrower payload encoded as JSON.
        """
        borrower_profile = _borrower_profile_from_json(borrower_payload_json)
        prediction = predict_risk_score(
            borrower_profile=borrower_profile,
            model=model,
            model_name=model_name,
            model_path=model_path,
        )
        prediction["policy_query"] = build_policy_query(
            borrower_profile=borrower_profile,
            risk_score=prediction["risk_score"],
        )
        return json.dumps(prediction, default=str)

    @tool
    def search_policy_docs(query: str) -> str:
        """
        Search lending-policy PDFs for rules, exceptions, mitigation guidance, and underwriting constraints.
        """
        return get_policy_context(query)

    return [predict_risk_score_tool, search_policy_docs]


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
    risk_factors = "\n".join(f"- {factor}" for factor in lending_decision.get("risk_factors", [])) or "- No extra risk factors recorded."

    try:
        from langchain_core.prompts import ChatPromptTemplate

        llm = _build_llm()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a lending risk copilot answering follow-up questions about one borrower. "
                    "Use the existing decision, borrower profile, risk factors, and policy summary. "
                    "Be precise and concise, and do not contradict the recorded verdict.",
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
            f"The current borrower verdict is `{lending_decision.get('final_verdict', 'Unavailable')}` "
            f"with risk score {lending_decision.get('risk_score', 0):.2f}. "
            f"Key factors were: {', '.join(lending_decision.get('risk_factors', [])) or 'no extra signals recorded'}. "
            f"Policy context considered: {policy_summary}"
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
    Produce a final lending verdict using tool-based reasoning over ML outputs and policy retrieval.
    """
    prediction = predict_risk_score(
        borrower_profile=borrower_profile,
        model=model,
        model_name=model_name,
        model_path=model_path,
    )
    policy_query = build_policy_query(
        borrower_profile=borrower_profile,
        risk_score=prediction["risk_score"],
    )

    policy_context = ""
    if prediction["risk_band"] == "High":
        try:
            policy_context = get_policy_context(policy_query)
        except Exception as exc:
            policy_context = f"Policy lookup unavailable: {exc}"

    try:
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        from langchain_core.prompts import ChatPromptTemplate

        llm = _build_llm()
        tools = build_lending_tools(
            model=model,
            model_name=model_name,
            model_path=model_path,
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an underwriting decision agent. Always call `predict_risk_score` first. "
                    "If the risk band is High, call `search_policy_docs` using the policy_query returned by the "
                    "prediction tool to see whether mitigants or documented exceptions apply. "
                    "Return a concise but complete lending recommendation that includes: "
                    "1) Final Lending Verdict, 2) Risk Score, 3) Reasoning paragraph, "
                    "4) Policy Citation Summary. Ground every claim in tool outputs.",
                ),
                (
                    "human",
                    "Assess this borrower and produce the final lending verdict.\nBorrower JSON:\n{borrower_payload}",
                ),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )
        agent = create_tool_calling_agent(llm, tools, prompt)
        executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
        result = executor.invoke({"borrower_payload": json.dumps(borrower_profile, default=str)})

        return {
            "final_verdict": "LLM Agent Recommendation",
            "risk_band": prediction["risk_band"],
            "risk_score": prediction["risk_score"],
            "model_name": prediction["model_name"],
            "reasoning": result["output"],
            "risk_factors": prediction["risk_factors"],
            "policy_query": policy_query,
            "policy_context": _format_policy_exception_guidance(policy_context),
            "borrower_profile": borrower_profile,
            "decision_source": "llm_agent",
        }
    except Exception:
        return {
            **_build_fallback_verdict(
                borrower_profile=borrower_profile,
                prediction=prediction,
                policy_context=_format_policy_exception_guidance(policy_context),
            ),
            "policy_query": policy_query,
        }
