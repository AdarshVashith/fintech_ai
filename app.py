from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from src.feature_engineering import create_features
from src.lending_agent import answer_follow_up_question, run_agentic_lending_decision
from src.preprocessing_pipeline import preprocess_uploaded_dataset
from src.report_export import generate_lending_report_pdf


st.set_page_config(
    page_title="Agentic Lending Command Center",
    layout="wide",
    initial_sidebar_state="expanded",
)


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "german_credit_data.csv"
POLICY_DIR = BASE_DIR / "data" / "policies"
MODEL_PATHS = {
    "Logistic Regression": BASE_DIR / "models" / "logistic_regression.pkl",
    "Decision Tree": BASE_DIR / "models" / "decision_tree.pkl",
}


def inject_theme() -> None:
    st.markdown(
        """
        <style>
            :root {
                --bg: #08111f;
                --bg-soft: #0d1830;
                --panel: rgba(13, 24, 48, 0.78);
                --panel-strong: rgba(8, 17, 31, 0.96);
                --line: rgba(91, 214, 255, 0.22);
                --text: #e8f7ff;
                --muted: #8baecc;
                --accent: #3ddcff;
                --accent-2: #6f8cff;
                --success: #24f2b3;
                --danger: #ff5f8f;
                --warning: #ffd166;
            }

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(61, 220, 255, 0.20), transparent 24%),
                    radial-gradient(circle at top right, rgba(111, 140, 255, 0.16), transparent 28%),
                    linear-gradient(180deg, #050b16 0%, #08111f 48%, #050b16 100%);
                color: var(--text);
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, rgba(8, 17, 31, 0.98), rgba(10, 20, 40, 0.92));
                border-right: 1px solid var(--line);
            }

            [data-testid="stSidebar"] * {
                color: var(--text);
            }

            .block-container {
                padding-top: 1.6rem;
                padding-bottom: 2rem;
            }

            h1, h2, h3, h4, p, label, span, div {
                color: var(--text);
            }

            .hero {
                padding: 1.6rem 1.8rem;
                border: 1px solid var(--line);
                border-radius: 26px;
                background:
                    linear-gradient(135deg, rgba(61, 220, 255, 0.10), rgba(111, 140, 255, 0.10)),
                    rgba(9, 18, 35, 0.88);
                box-shadow: 0 0 0 1px rgba(61, 220, 255, 0.05), 0 26px 80px rgba(0, 0, 0, 0.35);
                margin-bottom: 1.2rem;
            }

            .hero-eyebrow {
                font-size: 0.76rem;
                letter-spacing: 0.18em;
                text-transform: uppercase;
                color: var(--accent);
                margin-bottom: 0.65rem;
            }

            .hero-title {
                font-size: 2.5rem;
                line-height: 1.05;
                font-weight: 700;
                margin: 0;
            }

            .hero-copy {
                color: var(--muted);
                max-width: 60rem;
                margin-top: 0.7rem;
            }

            .glass-card {
                border: 1px solid var(--line);
                border-radius: 24px;
                background: var(--panel);
                padding: 1.2rem 1.2rem 1rem 1.2rem;
                box-shadow: inset 0 1px 0 rgba(255,255,255,0.02), 0 20px 60px rgba(0, 0, 0, 0.22);
            }

            .kpi-card {
                border: 1px solid rgba(61, 220, 255, 0.20);
                border-radius: 22px;
                padding: 1rem 1.1rem;
                background: linear-gradient(180deg, rgba(12, 26, 49, 0.95), rgba(8, 17, 31, 0.92));
                box-shadow: 0 0 0 1px rgba(61,220,255,0.04), 0 0 26px rgba(61,220,255,0.10);
                min-height: 128px;
            }

            .kpi-label {
                font-size: 0.78rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: var(--muted);
            }

            .kpi-value {
                font-size: 2rem;
                font-weight: 700;
                margin: 0.4rem 0 0.25rem 0;
            }

            .kpi-foot {
                color: var(--muted);
                font-size: 0.9rem;
            }

            .chat-shell {
                border: 1px solid var(--line);
                border-radius: 26px;
                padding: 1rem;
                background: linear-gradient(180deg, rgba(9, 18, 35, 0.92), rgba(7, 14, 28, 0.96));
            }

            .agent-note {
                color: var(--muted);
                font-size: 0.92rem;
                margin-bottom: 0.9rem;
            }

            .section-label {
                font-size: 0.78rem;
                text-transform: uppercase;
                letter-spacing: 0.14em;
                color: var(--accent);
                margin-bottom: 0.75rem;
            }

            .upload-zone {
                border: 1px dashed rgba(61, 220, 255, 0.4);
                border-radius: 18px;
                padding: 0.8rem;
                background: rgba(61, 220, 255, 0.04);
            }

            [data-testid="stTabs"] button {
                border-radius: 999px;
                border: 1px solid transparent;
                background: rgba(255, 255, 255, 0.02);
                color: var(--muted);
                padding: 0.55rem 1rem;
            }

            [data-testid="stTabs"] button[aria-selected="true"] {
                color: var(--text);
                border: 1px solid rgba(61, 220, 255, 0.34);
                background: linear-gradient(135deg, rgba(61, 220, 255, 0.16), rgba(111, 140, 255, 0.18));
                box-shadow: 0 0 24px rgba(61, 220, 255, 0.18);
            }

            .stDataFrame, [data-testid="stMetric"], [data-testid="stMarkdownContainer"] {
                color: var(--text);
            }

            .stAlert {
                border-radius: 18px;
                border: 1px solid rgba(61, 220, 255, 0.18);
                background: rgba(11, 21, 41, 0.92);
            }

            .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div,
            .stFileUploader section, .stTextArea textarea {
                background: rgba(7, 14, 28, 0.92) !important;
                color: var(--text) !important;
                border: 1px solid rgba(61, 220, 255, 0.18) !important;
                border-radius: 14px !important;
            }

            .stButton button, .stDownloadButton button, .stFormSubmitButton button {
                border-radius: 999px;
                border: 1px solid rgba(61, 220, 255, 0.35);
                background: linear-gradient(135deg, rgba(61, 220, 255, 0.18), rgba(111, 140, 255, 0.22));
                color: var(--text);
                box-shadow: 0 0 22px rgba(61, 220, 255, 0.16);
            }

            .stButton button:hover, .stFormSubmitButton button:hover {
                border-color: rgba(61, 220, 255, 0.7);
                box-shadow: 0 0 30px rgba(61, 220, 255, 0.26);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-eyebrow">Agentic Underwriting Studio</div>
            <h1 class="hero-title">Professional Lending Intelligence Dashboard</h1>
            <p class="hero-copy">
                Monitor portfolio health, interrogate lending decisions with the agent, and manage
                policy knowledge from a single dark-mode control surface designed for fintech teams.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model_registry() -> Dict[str, Any]:
    registry: Dict[str, Any] = {}
    for name, path in MODEL_PATHS.items():
        if path.exists():
            registry[name] = joblib.load(path)
    return registry


@st.cache_data
def load_local_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def load_active_dataset(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return load_local_dataset()


def actual_model(model: Any) -> Any:
    return model.best_estimator_ if hasattr(model, "best_estimator_") else model


def align_features(model: Any, features: pd.DataFrame) -> pd.DataFrame:
    resolved = actual_model(model)
    aligned = features.copy()
    if hasattr(resolved, "feature_names_in_"):
        expected = list(resolved.feature_names_in_)
        for col in expected:
            if col not in aligned.columns:
                aligned[col] = 0
        aligned = aligned[expected]
        aligned = aligned.fillna(aligned.median(numeric_only=True))
    return aligned


def score_dataset(dataset: pd.DataFrame, model: Any) -> pd.DataFrame:
    prepared = preprocess_uploaded_dataset(dataset)
    working = prepared["normalized"]
    features = create_features(working)
    aligned = align_features(model, features)
    probabilities = actual_model(model).predict_proba(aligned)[:, 1]

    scored = working.copy()
    if "Unnamed: 0" in scored.columns:
        scored = scored.drop(columns=["Unnamed: 0"])
    scored["Predicted Default Probability"] = probabilities
    scored["Estimated Credit Score"] = (850 - (probabilities * 550)).round(0).astype(int)
    scored["Predicted Decision"] = scored["Predicted Default Probability"].apply(
        lambda score: "High Risk" if score >= 0.5 else "Low Risk"
    )
    return scored


def build_model_summary_rows(model_scores: Dict[str, pd.DataFrame]) -> list[Dict[str, Any]]:
    summary_rows = []
    for name, scored in model_scores.items():
        summary_rows.append(
            {
                "Model": name,
                "Avg Risk": float(scored["Predicted Default Probability"].mean()),
                "High-Risk Share": float(scored["Predicted Default Probability"].ge(0.5).mean()),
                "Avg Credit Score": float(scored["Estimated Credit Score"].mean()),
            }
        )
    return summary_rows


def normalize_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)

    mapping = {
        "bad": 1,
        "default": 1,
        "high": 1,
        "1": 1,
        "good": 0,
        "non-default": 0,
        "low": 0,
        "0": 0,
    }
    normalized = series.astype(str).str.strip().str.lower().map(mapping)
    return normalized


def add_glow_line(fig: go.Figure, x, y, name: str, color: str) -> None:
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color=color, width=10),
            opacity=0.14,
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=name,
            line=dict(color=color, width=3),
        )
    )


def neon_layout(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,17,31,0.65)",
        font=dict(color="#e8f7ff"),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(
            bgcolor="rgba(8,17,31,0.45)",
            bordercolor="rgba(61,220,255,0.18)",
            borderwidth=1,
        ),
        xaxis=dict(
            gridcolor="rgba(61,220,255,0.10)",
            zerolinecolor="rgba(61,220,255,0.12)",
        ),
        yaxis=dict(
            gridcolor="rgba(61,220,255,0.10)",
            zerolinecolor="rgba(61,220,255,0.12)",
        ),
    )
    return fig


def build_gauge(risk_score: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_score * 100,
            number={"suffix": "%", "font": {"size": 44, "color": "#e8f7ff"}},
            title={"text": "Borrower Risk Signal", "font": {"size": 20, "color": "#8baecc"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8baecc"},
                "bar": {"color": "#ff5f8f" if risk_score >= 0.5 else "#24f2b3"},
                "bgcolor": "#08111f",
                "bordercolor": "rgba(61,220,255,0.20)",
                "steps": [
                    {"range": [0, 40], "color": "rgba(36,242,179,0.18)"},
                    {"range": [40, 60], "color": "rgba(255,209,102,0.18)"},
                    {"range": [60, 100], "color": "rgba(255,95,143,0.22)"},
                ],
                "threshold": {"line": {"color": "#3ddcff", "width": 5}, "value": 50},
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def save_uploaded_policies(files) -> list[str]:
    POLICY_DIR.mkdir(parents=True, exist_ok=True)
    saved_files: list[str] = []
    for file in files or []:
        file_path = POLICY_DIR / file.name
        file_path.write_bytes(file.getbuffer())
        saved_files.append(file.name)
    return saved_files


def render_kpi_card(label: str, value: str, footnote: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-foot">{footnote}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_user_summary(profile: Dict[str, Any]) -> str:
    return (
        f"Borrower request: {profile['purpose']} loan for {profile['credit_amount']} DM over "
        f"{profile['duration']} months. Profile: {profile['sex']}, age {profile['age']}, "
        f"housing {profile['housing']}, savings {profile['saving_accounts']}, "
        f"checking {profile['checking_account']}, job bucket {profile['job']}."
    )


def render_sidebar() -> None:
    st.sidebar.markdown("## Policy Management")
    st.sidebar.markdown(
        "<div class='upload-zone'>Upload new lending or regulatory PDFs to refresh the RAG knowledge base.</div>",
        unsafe_allow_html=True,
    )
    uploads = st.sidebar.file_uploader(
        "Guideline PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key="policy_uploads",
    )
    if st.sidebar.button("Save Policy Files", use_container_width=True):
        saved = save_uploaded_policies(uploads)
        if saved:
            st.sidebar.success(f"Saved {len(saved)} file(s): {', '.join(saved)}")
        else:
            st.sidebar.info("No new PDF files selected.")

    existing = sorted(p.name for p in POLICY_DIR.glob("*.pdf")) if POLICY_DIR.exists() else []
    st.sidebar.markdown("### Active Policy Library")
    if existing:
        for name in existing:
            st.sidebar.markdown(f"- {name}")
    else:
        st.sidebar.caption("No policy PDFs stored yet.")

    st.sidebar.markdown("---")
    st.sidebar.caption("Tip: upload a labeled validation CSV with a `Risk` column to unlock full ROC-AUC and confusion-matrix views in Model Lab.")


inject_theme()
render_sidebar()
render_hero()

if "agent_chat_history" not in st.session_state:
    st.session_state.agent_chat_history = []
if "latest_decision" not in st.session_state:
    st.session_state.latest_decision = None
if "latest_borrower_profile" not in st.session_state:
    st.session_state.latest_borrower_profile = None
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = None

model_registry = load_model_registry()
if not model_registry:
    st.error("Model pipeline not found in `models/`. Please train or restore the saved estimators.", icon="🚫")
    st.stop()

primary_model_name = "Logistic Regression" if "Logistic Regression" in model_registry else next(iter(model_registry))
primary_model = model_registry[primary_model_name]

dashboard_tab, agent_tab, model_lab_tab = st.tabs(["Dashboard", "Agentic Analysis", "Model Lab"])

with dashboard_tab:
    st.markdown("<div class='section-label'>Portfolio Intelligence</div>", unsafe_allow_html=True)
    dataset_upload = st.file_uploader(
        "Upload portfolio CSV for overview analytics",
        type=["csv"],
        key="portfolio_dataset",
    )
    dataset = load_active_dataset(dataset_upload)
    scored_dataset = score_dataset(dataset, primary_model)

    avg_credit_score = int(scored_dataset["Estimated Credit Score"].mean())
    default_rate = scored_dataset["Predicted Default Probability"].ge(0.5).mean() * 100
    avg_ticket = scored_dataset["Credit amount"].mean()
    avg_duration = scored_dataset["Duration"].mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card("Avg Credit Score", f"{avg_credit_score}", "Estimated from model risk probabilities")
    with c2:
        render_kpi_card("Default Rate", f"{default_rate:.1f}%", "Share of portfolio flagged as high risk")
    with c3:
        render_kpi_card("Avg Exposure", f"{avg_ticket:,.0f} DM", "Mean requested credit amount")
    with c4:
        render_kpi_card("Avg Duration", f"{avg_duration:.1f} mo", "Average loan tenor in portfolio")

    chart_col1, chart_col2 = st.columns([1.2, 1], gap="large")
    with chart_col1:
        risk_hist = px.histogram(
            scored_dataset,
            x="Predicted Default Probability",
            nbins=26,
            title="Portfolio Risk Distribution",
            color_discrete_sequence=["#3ddcff"],
        )
        risk_hist.update_traces(marker_line_color="#7bf0ff", marker_line_width=1.2, opacity=0.88)
        st.plotly_chart(neon_layout(risk_hist), use_container_width=True)

    with chart_col2:
        purpose_mix = (
            scored_dataset.groupby("Purpose", dropna=False)["Credit amount"]
            .mean()
            .reset_index(name="Average Credit Amount")
        )
        purpose_fig = px.pie(
            purpose_mix,
            names="Purpose",
            values="Average Credit Amount",
            hole=0.58,
            title="Purpose Mix by Average Exposure",
            color_discrete_sequence=px.colors.sequential.Bluered_r,
        )
        purpose_fig.update_traces(textinfo="percent+label")
        purpose_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e8f7ff"),
            legend=dict(bgcolor="rgba(8,17,31,0.45)"),
        )
        st.plotly_chart(purpose_fig, use_container_width=True)

    heatmap_col, table_col = st.columns([1, 1.2], gap="large")
    with heatmap_col:
        pivot = (
            scored_dataset.pivot_table(
                index="Housing",
                columns="Purpose",
                values="Predicted Default Probability",
                aggfunc="mean",
            )
            .fillna(0)
        )
        heat = px.imshow(
            pivot,
            text_auto=".2f",
            color_continuous_scale=["#08111f", "#14345c", "#3ddcff", "#ff5f8f"],
            title="Risk Heatmap by Housing and Purpose",
        )
        st.plotly_chart(neon_layout(heat), use_container_width=True)

    with table_col:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### Underwriting Feed")
        st.dataframe(
            scored_dataset[
                [
                    "Age",
                    "Sex",
                    "Housing",
                    "Credit amount",
                    "Duration",
                    "Purpose",
                    "Estimated Credit Score",
                    "Predicted Default Probability",
                    "Predicted Decision",
                ]
            ].sort_values("Predicted Default Probability", ascending=False),
            use_container_width=True,
            height=440,
        )
        st.markdown("</div>", unsafe_allow_html=True)

with agent_tab:
    st.markdown("<div class='section-label'>Decision Narrative</div>", unsafe_allow_html=True)
    agent_form_col, agent_chat_col = st.columns([0.95, 1.35], gap="large")

    with agent_form_col:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### Borrower Intake")
        with st.form("agentic_borrower_form", border=False):
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            sex = st.selectbox("Sex", ["male", "female"])
            job = st.selectbox(
                "Job Category",
                [0, 1, 2, 3],
                format_func=lambda x: {
                    0: "Unemployed / Unskilled (Non-Resident)",
                    1: "Unskilled (Resident)",
                    2: "Skilled Employee / Official",
                    3: "Management / Highly Skilled / Self-Employed",
                }[x],
            )
            housing = st.selectbox(
                "Housing Status",
                ["own", "free", "rent"],
                format_func=lambda x: {"own": "Owns Property", "free": "Lives For Free", "rent": "Renting"}[x],
            )
            saving_accounts = st.selectbox(
                "Savings Account",
                ["NA", "little", "moderate", "quite rich", "rich"],
            )
            checking_account = st.selectbox(
                "Checking Account",
                ["NA", "little", "moderate", "rich"],
            )
            credit_amount = st.number_input("Requested Amount (DM)", min_value=100, value=2500)
            duration = st.slider("Duration (Months)", min_value=4, max_value=72, value=24)
            purpose = st.selectbox(
                "Purpose",
                [
                    "radio/TV",
                    "education",
                    "furniture/equipment",
                    "car",
                    "business",
                    "domestic appliances",
                    "repairs",
                    "vacation/others",
                ],
            )
            submitted = st.form_submit_button("Run Agentic Analysis", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        borrower_profile = {
            "age": age,
            "sex": sex,
            "job": job,
            "housing": housing,
            "saving_accounts": saving_accounts,
            "checking_account": checking_account,
            "credit_amount": credit_amount,
            "duration": duration,
            "purpose": purpose,
        }
        user_message = build_user_summary(borrower_profile)
        decision = run_agentic_lending_decision(
            borrower_profile=borrower_profile,
            model=primary_model,
            model_name=primary_model_name,
        )
        agent_message = (
            f"Final Lending Verdict: {decision['final_verdict']}\n\n"
            f"Reasoning:\n{decision['reasoning']}\n\n"
            f"Policy Summary:\n{decision.get('policy_context', 'No policy context retrieved.')}"
        )
        st.session_state.conversation_memory = None
        st.session_state.agent_chat_history.append(("user", user_message))
        st.session_state.agent_chat_history.append(("assistant", agent_message))
        st.session_state.latest_decision = decision
        st.session_state.latest_borrower_profile = borrower_profile

    with agent_chat_col:
        st.markdown("<div class='chat-shell'>", unsafe_allow_html=True)
        st.markdown("<div class='agent-note'>The underwriting agent explains each decision using model output plus retrieved policy context.</div>", unsafe_allow_html=True)

        if not st.session_state.agent_chat_history:
            st.info("Submit a borrower profile to open the decision conversation.")
        else:
            for role, message in st.session_state.agent_chat_history:
                with st.chat_message(role):
                    st.markdown(message)

        if st.session_state.latest_decision:
            gauge_col, signal_col = st.columns([0.9, 1.1], gap="large")
            with gauge_col:
                st.plotly_chart(build_gauge(st.session_state.latest_decision["risk_score"]), use_container_width=True)
            with signal_col:
                st.markdown("### Decision Signals")
                for factor in st.session_state.latest_decision.get("risk_factors", []):
                    st.markdown(f"- {factor}")
                st.caption(f"Decision source: {st.session_state.latest_decision.get('decision_source', 'n/a')}")
                if st.session_state.latest_decision.get("policy_query"):
                    st.caption(f"Policy query: {st.session_state.latest_decision['policy_query']}")

            st.markdown("### Follow-Up Questions")
            follow_up = st.text_input(
                "Ask the agent to clarify this borrower decision",
                placeholder="Why was this borrower marked high risk, and what exception would matter most?",
                key="follow_up_question",
            )
            action_col1, action_col2 = st.columns([0.8, 1.2], gap="medium")
            with action_col1:
                if st.button("Ask Follow-Up", use_container_width=True):
                    if follow_up.strip():
                        follow_up_result = answer_follow_up_question(
                            question=follow_up.strip(),
                            borrower_profile=st.session_state.latest_borrower_profile,
                            lending_decision=st.session_state.latest_decision,
                            memory=st.session_state.conversation_memory,
                        )
                        st.session_state.conversation_memory = follow_up_result["memory"]
                        st.session_state.agent_chat_history.append(("user", follow_up.strip()))
                        st.session_state.agent_chat_history.append(("assistant", follow_up_result["answer"]))
                        st.rerun()
                    else:
                        st.warning("Enter a follow-up question to continue the conversation.")
            with action_col2:
                report_payload = generate_lending_report_pdf(
                    borrower_profile=st.session_state.latest_borrower_profile,
                    lending_decision=st.session_state.latest_decision,
                    model_metrics=build_model_summary_rows(
                        {name: score_dataset(load_local_dataset(), model) for name, model in model_registry.items()}
                    ),
                )
                st.download_button(
                    "Export Report",
                    data=report_payload,
                    file_name="agentic_lending_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)

with model_lab_tab:
    st.markdown("<div class='section-label'>Model Diagnostics</div>", unsafe_allow_html=True)
    validation_upload = st.file_uploader(
        "Optional labeled validation CSV for ROC-AUC and confusion matrix",
        type=["csv"],
        key="validation_dataset",
    )
    validation_df = load_active_dataset(validation_upload)

    model_scores: Dict[str, pd.DataFrame] = {}
    for name, model in model_registry.items():
        model_scores[name] = score_dataset(validation_df, model)
    summary_rows = build_model_summary_rows(model_scores)

    target_available = "Risk" in validation_df.columns and normalize_target(validation_df["Risk"]).notna().all()

    top_col1, top_col2 = st.columns([1.1, 1], gap="large")
    with top_col1:
        if target_available:
            truth = normalize_target(validation_df["Risk"]).astype(int)
            roc_fig = go.Figure()
            palette = {
                "Logistic Regression": "#3ddcff",
                "Decision Tree": "#ff5f8f",
            }
            metric_cards = []

            for name, scored in model_scores.items():
                probs = scored["Predicted Default Probability"]
                preds = (probs >= 0.5).astype(int)
                fpr, tpr, _ = roc_curve(truth, probs)
                auc = roc_auc_score(truth, probs)
                add_glow_line(roc_fig, fpr, tpr, f"{name} (AUC {auc:.3f})", palette.get(name, "#6f8cff"))
                metric_cards.append((name, auc, confusion_matrix(truth, preds)))

            roc_fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Random Baseline",
                    line=dict(color="rgba(232,247,255,0.45)", dash="dash"),
                )
            )
            roc_fig.update_layout(
                title="ROC-AUC Comparison",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
            )
            st.plotly_chart(neon_layout(roc_fig), use_container_width=True)

            choice = st.selectbox("Confusion Matrix Model", list(model_scores.keys()), key="cm_model")
            probs = model_scores[choice]["Predicted Default Probability"]
            preds = (probs >= 0.5).astype(int)
            cm = confusion_matrix(truth, preds)
            cm_fig = px.imshow(
                cm,
                text_auto=True,
                x=["Predicted Good", "Predicted Default"],
                y=["Actual Good", "Actual Default"],
                color_continuous_scale=["#08111f", "#14345c", "#3ddcff", "#ff5f8f"],
                title=f"{choice} Confusion Matrix",
            )
            st.plotly_chart(neon_layout(cm_fig), use_container_width=True)
        else:
            st.warning(
                "No `Risk` column was found in the active dataset, so true ROC-AUC and confusion-matrix evaluation cannot be computed yet."
            )
            comparison_fig = go.Figure()
            for name, scored in model_scores.items():
                ranked = scored["Predicted Default Probability"].sort_values().reset_index(drop=True)
                add_glow_line(
                    comparison_fig,
                    ranked.index,
                    ranked.values,
                    name,
                    "#3ddcff" if name == "Logistic Regression" else "#ff5f8f",
                )
            comparison_fig.update_layout(
                title="Model Score Curves on Active Portfolio",
                xaxis_title="Borrower Rank",
                yaxis_title="Predicted Default Probability",
            )
            st.plotly_chart(neon_layout(comparison_fig), use_container_width=True)

    with top_col2:
        summary_df = pd.DataFrame(summary_rows)
        summary_fig = px.bar(
            summary_df,
            x="Model",
            y="High-Risk Share",
            color="Model",
            color_discrete_map={
                "Logistic Regression": "#3ddcff",
                "Decision Tree": "#ff5f8f",
            },
            title="High-Risk Share by Model",
        )
        summary_fig.update_traces(marker_line_color="#e8f7ff", marker_line_width=1.0)
        st.plotly_chart(neon_layout(summary_fig), use_container_width=True)

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### Model Snapshot")
        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
