"""
Phase 2 - LCEL Pipelines
Three distinct execution patterns using LangChain Expression Language.
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from models.llm import llm

parser = StrOutputParser()

# ── Pattern 1: Sequential (RunnableSequence via |) ──────────────────────────
# Question -> Prompt -> LLM -> Parser
summarize_prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarize the following text in 2 sentences."),
    ("human", "{text}"),
])
sequential_chain = summarize_prompt | llm | parser


# ── Pattern 2: Parallel (RunnableParallel) ───────────────────────────────────
# Run two independent chains on the same input, combine results.
keywords_prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract 5 keywords from this text. Return comma-separated."),
    ("human", "{text}"),
])
keywords_chain = keywords_prompt | llm | parser

parallel_chain = RunnableParallel(
    summary=sequential_chain,
    keywords=keywords_chain,
)
# parallel_chain.invoke({"text": "..."}) -> {"summary": "...", "keywords": "..."}


# ── Pattern 3: Passthrough (RunnablePassthrough) ─────────────────────────────
# Forward the original input alongside a transformed value - useful when
# you need both the raw question AND a processed result downstream (RAG uses this).
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question. Mention you also see the original input."),
    ("human", "Original question: {original}\nProcessed: {processed}"),
])

passthrough_chain = (
    RunnableParallel(
        original=RunnablePassthrough(),
        processed=lambda x: x.upper(),
    )
    | answer_prompt
    | llm
    | parser
)


if __name__ == "__main__":
    print("--- Sequential ---")
    print(sequential_chain.invoke({"text": "LangChain is a framework for building LLM apps."}))

    print("\n--- Parallel ---")
    print(parallel_chain.invoke({"text": "LangChain is a framework for building LLM apps."}))

    print("\n--- Passthrough ---")
    print(passthrough_chain.invoke("what is rag"))
