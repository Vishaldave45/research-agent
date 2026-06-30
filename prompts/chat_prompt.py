"""
Reusable ChatPromptTemplate objects.
Keeping prompts separate from chains makes them easy to test and swap.
"""
from langchain_core.prompts import ChatPromptTemplate

# Phase 1 - basic chatbot
basic_chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI research assistant. Be concise and precise."),
    ("human", "{question}"),
])

# Phase 6 - RAG: answer using retrieved context only
rag_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a research assistant. Answer the user's question using ONLY the "
     "context below. If the answer isn't in the context, say you don't know.\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])

# Phase 7 - conversational (memory-aware) prompt
conversational_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI research assistant. Use the conversation "
               "history to understand follow-up questions."),
    ("placeholder", "{history}"),
    ("human", "{question}"),
])

# Phase 9 - agent system prompt
agent_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an AI research assistant with access to tools and a document "
     "search system. Decide whether to answer directly, search documents, or "
     "use a tool. Always explain your reasoning briefly before acting."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
