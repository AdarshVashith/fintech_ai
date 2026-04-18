from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage

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

    if provider in {"groq", ""} and os.getenv("GROQ_API_KEY"):
        from langchain_groq import ChatGroq

        model_name = os.getenv("LENDING_AGENT_MODEL", "llama-3.3-70b-versatile")
        return ChatGroq(model=model_name, temperature=0.7)

    if provider in {"openai", ""} and os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI

        model_name = os.getenv("LENDING_AGENT_MODEL", "gpt-4o")
        return ChatOpenAI(model=model_name, temperature=0.7)

    if provider in {"anthropic", ""} and os.getenv("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic

        model_name = os.getenv("LENDING_AGENT_MODEL", "claude-3-5-sonnet-latest")
        return ChatAnthropic(model=model_name, temperature=0.7)

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
    LangChain memory import issues on deployment targets.
    """

    def __init__(self) -> None:
        self.chat_history: list[Any] = []

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
        verdict = lending_decision.get("final_verdict", "Unavailable")
        score = lending_decision.get("risk_score", 0)
        factors = lending_decision.get("risk_factors", [])
        
        answer = (
            f"**Direct Answer**: Based on the recorded analysis, the current verdict is **{verdict}** "
            f"with a quantitative risk score of **{score:.2f}**.\n\n"
            f"### Why\n"
            f"The primary influencers for this assessment include: " + 
            (", ".join(factors) if factors else "standard profile metrics without anomalous risk signals") + ".\n\n"
            f"### What mattered most\n"
            f"The model prioritized the relationship between your requested credit amount, "
            f"loan duration, and existing account stability. These factors combined to place "
            f"the application in the `{lending_decision.get('risk_band', 'N/A')}` risk category.\n\n"
            f"### What could change the decision\n"
            f"{policy_summary}\n\n"
            f"*(Note: This is a data-driven fallback response as the AI reasoning engine is currently limited.)*"
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
                    "You are a professional underwriting agent. Always call `predict_risk_score` first. "
                    "Then call `search_policy_docs` using the policy_query returned by the prediction tool "
                    "to ground your decision in actual lending standards. "
                    "Return ONLY a clean JSON object with no markdown wrappers. "
                    "\n\nFormat requirements for JSON fields:\n"
                    "- `final_verdict`: High-level status.\n"
                    "- `reasoning`: Technical breakdown in paragraphs.\n"
                    "- `recommendations`: Actionable steps as a Markdown bulleted list.\n"
                    "- `references`: A structured Markdown list of citations. For each source, use: **Source: [Filename] (Page X)** followed by bulleted clauses.\n"
                    "- `disclaimer`: Standard underwriting caveat.\n\n"
                    "Ensure your output is valid JSON and contains NO preamble or postscript.",
                ),
                (
                    "human",
                    "Assess this borrower and produce the final lending report.\nBorrower JSON:\n{borrower_payload}",
                ),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )
        agent = create_tool_calling_agent(llm, tools, prompt)
        executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
        result = executor.invoke({"borrower_payload": json.dumps(borrower_profile, default=str)})

        # Parse the JSON output from the agent if possible
        try:
            raw_output = result["output"]
            # Basic cleaning in case of markdown wrappers
            cleaned = raw_output.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(cleaned)
            
            return {
                "final_verdict": parsed.get("final_verdict", "LLM Agent Recommendation"),
                "risk_band": prediction["risk_band"],
                "risk_score": prediction["risk_score"],
                "model_name": prediction["model_name"],
                "reasoning": parsed.get("reasoning", result["output"]),
                "recommendations": parsed.get("recommendations", "Manual review required."),
                "references": parsed.get("references", _format_policy_exception_guidance(policy_context)),
                "disclaimer": parsed.get("disclaimer", "Standard underwriting terms apply."),
                "risk_factors": prediction["risk_factors"],
                "policy_query": policy_query,
                "borrower_profile": borrower_profile,
                "decision_source": "llm_agent",
            }
        except:
            # Fallback to unstructured if parsing fails
            return {
                "final_verdict": "LLM Agent Recommendation",
                "risk_band": prediction["risk_band"],
                "risk_score": prediction["risk_score"],
                "model_name": prediction["model_name"],
                "reasoning": result["output"],
                "recommendations": "Evaluate debt-to-income and collateral strength.",
                "references": _format_policy_exception_guidance(policy_context),
                "disclaimer": "This analysis is for advisory purposes only.",
                "risk_factors": prediction["risk_factors"],
                "policy_query": policy_query,
                "borrower_profile": borrower_profile,
                "decision_source": "llm_agent",
            }
    except Exception:
        fallback = _build_fallback_verdict(
            borrower_profile=borrower_profile,
            prediction=prediction,
            policy_context=_format_policy_exception_guidance(policy_context),
        )
        # Ensure new keys exist in fallback
        fallback.update({
            "recommendations": "Manual review of credit history and verification of stated income sources.",
            "references": _format_policy_exception_guidance(policy_context),
            "disclaimer": "Advisory report only. Final approval subject to bank compliance officers.",
            "policy_query": policy_query
        })
        return fallback
