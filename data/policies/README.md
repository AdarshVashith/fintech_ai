Place lending-policy and regulatory-guideline PDFs in this folder.

Examples:
- internal_underwriting_policy.pdf
- central_bank_credit_guidelines.pdf
- high_risk_mitigation_rules.pdf

The RAG pipeline reads every `*.pdf` file in this directory, splits the text into chunks,
embeds the chunks with SentenceTransformers, and stores the FAISS index under
`models/policy_faiss_index/`.
