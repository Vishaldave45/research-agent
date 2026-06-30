"""
Phase 10 - Structured Output

Defines five structured output chains corresponding to the future Phase 12 agents.
Uses Pydantic schemas and with_structured_output().
"""
from langchain_core.prompts import ChatPromptTemplate
from models.llm import llm_precise
from schemas.response_schema import (
    ResearchResult,
    SummaryResult,
    FactCheckResult,
    QAResult,
    ReportResult,
)

# ── 1. Research Chain ────────────────────────────────────────────────────────
research_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert Research Agent. Conduct a detailed research search on the requested topic. Extract key findings, cite sources if any, and estimate your confidence."),
    ("human", "Research topic: {topic}")
])
research_chain = research_prompt | llm_precise.with_structured_output(ResearchResult)

def research(topic: str) -> ResearchResult:
    """Run research structured chain."""
    return research_chain.invoke({"topic": topic})

# ── 2. Summarization Chain ───────────────────────────────────────────────────
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Summarization Agent. Summarize the provided text, extract key bullet points, provide a descriptive title, and estimate the original word count."),
    ("human", "Text to summarize:\n{text}")
])
summary_chain = summary_prompt | llm_precise.with_structured_output(SummaryResult)

def summarize(text: str) -> SummaryResult:
    """Run summarization structured chain."""
    return summary_chain.invoke({"text": text})

# ── 3. Fact Checking Chain ───────────────────────────────────────────────────
fact_check_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Fact Checking Agent. Analyze the claim, determine if it is TRUE, FALSE, PARTIALLY TRUE, or UNVERIFIABLE, provide supporting evidence, and rate your confidence."),
    ("human", "Claim: {claim}")
])
fact_check_chain = fact_check_prompt | llm_precise.with_structured_output(FactCheckResult)

def fact_check(claim: str) -> FactCheckResult:
    """Run fact-checking structured chain."""
    return fact_check_chain.invoke({"claim": claim})

# ── 4. Question Answering Chain ──────────────────────────────────────────────
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Question Answering Agent. Provide a clear answer along with your step-by-step reasoning and a confidence score."),
    ("human", "Question: {question}")
])
qa_chain = qa_prompt | llm_precise.with_structured_output(QAResult)

def ask_structured(question: str) -> QAResult:
    """Run QA structured chain."""
    return qa_chain.invoke({"question": question})

# ── 5. Report Writing Chain ──────────────────────────────────────────────────
report_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Report Writing Agent. Draft a structured report about the given topic. Return a title, a list of sections (each with a 'heading' and 'content' key), a concluding summary, and a list of references."),
    ("human", "Report Topic: {topic}")
])
report_chain = report_prompt | llm_precise.with_structured_output(ReportResult)

def generate_report(topic: str) -> ReportResult:
    """Run report structured chain."""
    return report_chain.invoke({"topic": topic})


if __name__ == "__main__":
    # uv run python -m chains.structured_chain
    print("Testing Research Chain...")
    res = research("LangChain Expression Language (LCEL)")
    print(res)

    print("\nTesting Fact Check Chain...")
    fc = fact_check("Water boils at 100 degrees Celsius under standard atmospheric pressure.")
    print(fc)

    print("\nTesting Summarization Chain...")
    sm = summarize("LangChain is a framework for developing applications powered by large language models (LLMs). It simplifies building chains, agents, and RAG architectures using a unified expression language (LCEL).")
    print(sm)
