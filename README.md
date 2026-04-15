# INTELLIGENT CREDIT RISK SCORING & AGENTIC LENDING DECISION SUPPORT

##  Project Overview

This project involves the design and implementation of an system that evaluates borrower credit risk and evolves in decision support assistant. AI-driven credit analytics of an agentic AI lending.

---

##  Problem Statement

Financial institutions face significant challenges in assessing borrower creditworthiness accurately. Manual risk evaluation processes are often time consuming, inconsistent, and prone to human bias.

This project addresses this problem by implementing an automated credit risk scoring system that uses machine learning algorithms to analyze borrower data and classify applicants into risk categories.


##  Key Features

- Upload borrower dataset through an interactive UI  
- Automatic data preprocessing pipeline  
- Support for categorical encoding and feature scaling  
- Training and comparison of multiple ML models  
- Real-time credit risk prediction  
- Retrieval-Augmented Generation (RAG) over lending-policy PDFs  
- Agentic lending verdicts that combine ML risk scoring and policy lookup  
- Automatic preprocessing for raw borrower CSVs using OneHotEncoder and StandardScaler  
- Follow-up decision Q&A with conversation memory and PDF report export  
- Visualization of evaluation metrics  
- Clean and user-friendly Streamlit interface  


##  Machine Learning Models Used

The following supervised learning models were implemented:

**Logistic Regression**
- Used for probabilistic classification
- Estimates default likelihood

**Decision Tree Classifier**
- Rule-based classification model
- Identifies important risk driving features


##  Evaluation Metrics

Model performance is evaluated using:

- Accuracy Score
- ROC-AUC Score
- Confusion Matrix
- ROC Curve Visualization
- Feature Importance Analysis


## Installation and Setup Instructions

Follow these steps to run the project locally.

### Step 1: Clone the Repository

```bash
git clone https://github.com/CWAbhi/Gen-AI_Capstone.git
cd Gen-AI_Capstone
```
### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```
### Step 3: Add Policy PDFs for RAG

Copy lending-policy or regulatory-guideline PDF files into:

```bash
data/policies/
```

The application will automatically build a FAISS index the first time policy
retrieval is used.
### Step 4: Launch the Application

Start the Streamlit server:
``` bash
streamlit run app.py
```
The application will open automatically in your browser.

## Policy Retrieval Module

The RAG pipeline lives in `src/rag_pipeline.py` and provides:

- `ingest_policy_documents()` to parse PDFs and build the FAISS index
- `get_policy_context(query)` to retrieve policy passages for borrower-specific scenarios
- `build_policy_query(profile, risk_score)` to convert borrower risk factors into a retrieval query

Example:

```python
from src.rag_pipeline import get_policy_context

query = "High DTI borrower with low savings needs compensating-factor policy"
context = get_policy_context(query)
print(context)
```

## Agentic Lending Layer

The Phase 2 agent logic lives in `src/lending_agent.py` and wraps the existing model + RAG
pipeline as LangChain tools:

- `predict_risk_score` runs the trained classifier and returns a structured risk payload
- `search_policy_docs` retrieves policy passages from the FAISS-backed lending corpus
- `run_agentic_lending_decision(profile)` orchestrates the final verdict

To enable the LLM-backed tool-calling agent, configure one of these:

```bash
export OPENAI_API_KEY=...
export LENDING_AGENT_PROVIDER=openai
export LENDING_AGENT_MODEL=gpt-4o
```

or

```bash
export ANTHROPIC_API_KEY=...
export LENDING_AGENT_PROVIDER=anthropic
export LENDING_AGENT_MODEL=claude-3-5-sonnet-latest
```

If no API key is configured, the application falls back to a deterministic decision engine
that still uses the ML score and policy retrieval outputs.

## Phase 4 Enhancements

- Uploaded borrower datasets now flow through an automated sklearn preprocessing layer
  with imputation, `OneHotEncoder`, and `StandardScaler` for analytics and model-lab views.
- The Agentic Analysis tab uses conversation memory so users can ask follow-up questions
  about the latest borrower decision without losing context.
- An `Export Report` action generates a PDF summary containing the borrower profile,
  final verdict, policy context, and model metric snapshot.


## Team Contribution

| Member                     | Contribution                                                              |
| -------------------------- | ------------------------------------------------------------------------- |
| Anshika Seth (2401010080)  | Data Cleaning & EDA, Complete Model Development, Streamlit UI, Deployment |
| Abhijeet Dey (2401010014)  | Helped Model Development, Deployment                                      |
| Aditya Ranjan (2401010035) | Documentation & Testing                                                   |


## Conclusion

The Credit Risk Prediction System successfully demonstrates how Machine Learning can automate loan risk assessment. The trained model achieved strong performance and can assist financial institutions in making reliable lending decisions.
# Fintech_ai
