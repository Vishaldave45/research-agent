"""
Phase 12 - Multi-Agent Workflow

Orchestrates multiple specialized agents (Research, Summarize, Fact Check, QA, Report Writer)
using a central Coordinator Agent to route requests.
"""
from langchain_core.prompts import ChatPromptTemplate
from models.llm import llm_precise
from schemas.response_schema import (
    RoutingDecision,
    ResearchResult,
    SummaryResult,
    FactCheckResult,
    QAResult,
    ReportResult,
)
from chains.structured_chain import (
    research as run_research_chain,
    summarize as run_summarize_chain,
    fact_check as run_fact_check_chain,
    ask_structured as run_qa_chain,
    generate_report as run_report_chain,
)
from tools.tools import document_search

# ── Coordinator Agent Definition ─────────────────────────────────────────────
coordinator_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are the Coordinator Agent for a Multi-Agent AI system. "
               "Your task is to analyze the user's input and select the most appropriate specialized agent. "
               "Choose from:\n"
               "- 'research': If the user wants a deep search, topic exploration, or analysis of a complex topic.\n"
               "- 'summarization': If the user provides a body of text and explicitly asks for a summary, key points, or a quick read.\n"
               "- 'fact_checking': If the user presents a statement, claim, or fact and wants it verified, debunked, or checked.\n"
               "- 'qa': If the user has a direct question, general inquiry, or math/logical query.\n"
               "- 'report_writer': If the user wants a structured, formatted document, outline, or detailed report.\n\n"
               "Explain your reasoning and extract the core query to pass to the selected agent."),
    ("human", "{query}")
])

coordinator_chain = coordinator_prompt | llm_precise.with_structured_output(RoutingDecision)


# ── Specialized Agent Runners ────────────────────────────────────────────────
def run_research_agent(topic: str) -> ResearchResult:
    """Research Agent: Queries local knowledge base first, then research chain."""
    docs_context = document_search.invoke({"query": topic})
    enriched_topic = topic
    if docs_context and "No documents have been indexed" not in docs_context:
        enriched_topic = f"{topic}\n\n[Context from indexed document library]:\n{docs_context}"
    return run_research_chain(enriched_topic)


def run_summarization_agent(text: str) -> SummaryResult:
    """Summarization Agent: Summarizes text structured."""
    return run_summarize_chain(text)


def run_fact_checking_agent(claim: str) -> FactCheckResult:
    """Fact Checking Agent: Verifies claims against vector store context."""
    docs_context = document_search.invoke({"query": claim})
    enriched_claim = claim
    if docs_context and "No documents have been indexed" not in docs_context:
        enriched_claim = f"{claim}\n\n[Context from indexed document library]:\n{docs_context}"
    return run_fact_check_chain(enriched_claim)


def run_qa_agent(question: str) -> QAResult:
    """Question Answering Agent: Answers questions with reasoning."""
    return run_qa_chain(question)


def run_report_agent(topic: str) -> ReportResult:
    """Report Writing Agent: Drafts formatted multi-section reports."""
    return run_report_chain(topic)


# ── Coordinator Orchestration ────────────────────────────────────────────────
def coordinate(query: str) -> dict:
    """
    Classify the query and run the corresponding specialized agent.

    Returns:
        dict: {
            "agent": str,
            "reason": str,
            "result": PydanticModel (the output from the selected agent)
        }
    """
    decision: RoutingDecision = coordinator_chain.invoke({"query": query})
    agent = decision.selected_agent
    processed_query = decision.processed_query

    if agent == "research":
        result = run_research_agent(processed_query)
    elif agent == "summarization":
        result = run_summarization_agent(processed_query)
    elif agent == "fact_checking":
        result = run_fact_checking_agent(processed_query)
    elif agent == "report_writer":
        result = run_report_agent(processed_query)
    else:  # qa
        result = run_qa_agent(processed_query)

    return {
        "agent": agent,
        "reason": decision.routing_reason,
        "result": result
    }


if __name__ == "__main__":
    # uv run python -m chains.multi_agent_workflow
    print("Testing Coordinator Routing & Execution:")
    
    test_queries = [
        "Research the key features of LangChain agents.",
        "Check this claim: water freezes at 32 degrees Fahrenheit.",
        "Write a detailed report on solar power integration in smart grids."
    ]
    
    for q in test_queries:
        print(f"\nQuery: '{q}'")
        output = coordinate(q)
        print(f"Routed to: {output['agent']}")
        print(f"Reasoning: {output['reason']}")
        print(f"Result: {output['result']}")
