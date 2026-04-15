from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, Iterable

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _paragraphs_from_lines(lines: Iterable[str], style: ParagraphStyle) -> list[Any]:
    elements: list[Any] = []
    for line in lines:
        if not line:
            continue
        elements.append(Paragraph(line.replace("\n", "<br/>"), style))
        elements.append(Spacer(1, 0.12 * inch))
    return elements


def generate_lending_report_pdf(
    borrower_profile: Dict[str, Any],
    lending_decision: Dict[str, Any],
    model_metrics: Iterable[Dict[str, Any]],
) -> bytes:
    """
    Generate a PDF summary of the borrower decision, policy context, and model metrics.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=42,
        rightMargin=42,
        topMargin=42,
        bottomMargin=42,
    )
    styles = getSampleStyleSheet()
    title = styles["Title"]
    heading = styles["Heading2"]
    body = styles["BodyText"]
    body.leading = 15

    accent_style = ParagraphStyle(
        "Accent",
        parent=body,
        textColor=colors.HexColor("#0b3d91"),
        fontSize=10,
        leading=14,
    )

    story: list[Any] = [
        Paragraph("Agentic Lending Decision Report", title),
        Spacer(1, 0.15 * inch),
        Paragraph("Underwriting summary generated from model scoring, policy retrieval, and agent reasoning.", accent_style),
        Spacer(1, 0.22 * inch),
    ]

    profile_rows = [["Borrower Attribute", "Value"]]
    for key, value in borrower_profile.items():
        profile_rows.append([str(key).replace("_", " ").title(), str(value)])
    profile_table = Table(profile_rows, colWidths=[2.2 * inch, 3.6 * inch])
    profile_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b3d91")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#b7c9e6")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f5f8ff")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    story.extend([Paragraph("Borrower Profile", heading), Spacer(1, 0.1 * inch), profile_table, Spacer(1, 0.2 * inch)])

    decision_lines = [
        f"<b>Final Verdict:</b> {lending_decision.get('final_verdict', 'Unavailable')}",
        f"<b>Risk Band:</b> {lending_decision.get('risk_band', 'Unavailable')}",
        f"<b>Risk Score:</b> {float(lending_decision.get('risk_score', 0)):.3f}",
        f"<b>Model:</b> {lending_decision.get('model_name', 'Unavailable')}",
        f"<b>Decision Source:</b> {lending_decision.get('decision_source', 'Unavailable')}",
        f"<b>Reasoning:</b> {lending_decision.get('reasoning', 'No reasoning available.')}",
        f"<b>Policy Summary:</b> {lending_decision.get('policy_context', 'No policy context available.')}",
    ]
    story.extend([Paragraph("Agent Decision", heading), Spacer(1, 0.1 * inch), *_paragraphs_from_lines(decision_lines, body)])

    risk_factor_lines = [f"- {factor}" for factor in lending_decision.get("risk_factors", [])] or ["- No specific risk factors recorded."]
    story.extend([Paragraph("Risk Factors", heading), Spacer(1, 0.1 * inch), *_paragraphs_from_lines(risk_factor_lines, body)])

    metric_rows = [["Model", "Avg Risk", "High-Risk Share", "Avg Credit Score"]]
    for metric in model_metrics:
        metric_rows.append(
            [
                str(metric.get("Model", "")),
                str(metric.get("Avg Risk", "")),
                str(metric.get("High-Risk Share", "")),
                str(metric.get("Avg Credit Score", "")),
            ]
        )
    metric_table = Table(metric_rows, colWidths=[1.8 * inch, 1.4 * inch, 1.6 * inch, 1.5 * inch])
    metric_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123d6a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#eef4ff")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cad9f2")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend([Paragraph("Model Metrics Snapshot", heading), Spacer(1, 0.1 * inch), metric_table])

    doc.build(story)
    return buffer.getvalue()
