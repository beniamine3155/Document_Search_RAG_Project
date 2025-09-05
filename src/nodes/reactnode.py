"""Langgraph nodes for rag workflow and react agent."""
from typing import List, Optional
from src.state.rag_state import RAGState

from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# Wikipedia tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun


class RAGNodes:

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self._agent = None

    
    def retrieve_docs(self, state: RAGState) -> RAGState:
        """ Retrieves documents based on the query in the state."""
        docs = self.retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=docs
        )
    
    def _build_tools(self) -> List[Tool]:
        """ Build the tools for the agent."""
        
        def retriever_tool_fn(query: str) -> str:
            """ Tool function to retrieve documents."""
            docs = List[Document] = self.retriever.invoke(query)
            if not docs:
                return "No relevant documents found."
            merged = []
            for i, d in enumerate(docs[:8], start=1):
                meta = d.metadata if hasattr(d, "metadata") else {}
                title = meta.get("title") or meta.get("source") or f"doc_{i}"
                merged.append(f"[{i}] {title}\n{d.page_content}")
            return "\n\n".join(merged)
        
        retrieve_tool = Tool(
            name="Document Retriever",
            description="Fetch passeges from index corpus for the given query.",
            func=retriever_tool_fn
        )

        wiki = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=3, lang="en")
        )

        wiki_tool = Tool(
            name="Wikipedia",
            description="Search the wikipedia for a topic and return the summary.",
            func=wiki.run
        )

        return [retrieve_tool, wiki_tool]
    

    
    def _build_agent(self):
        """ Build the react agent."""
        tools = self._build_tools()
        system_prompt = (
            "You are a helpful RAG agent that helps find the information. "
            "Prefer 'retriever' for user-provided docs; use 'wikipedia' for general knowledge. "
            "Return only the final useful answer."
            
        )

        self._agent = create_react_agent(self.llm, tools, system_prompt=system_prompt)
                
        
    def generate_answer(self, state: RAGState) -> RAGState:
        """ Generates an answer using react agent based on the question and retrieved documents in the state."""
        if self._agent is None:
            self._build_agent()
        
        result = self._agent.invoke({"messages": [HumanMessage(content=state.question)]})
        messages = result.get("messages", [])
        answer: Optional[str] = None
        if messages:
            answer_msg = messages[-1]
            answer = getattr(answer_msg, "content", None)
        
        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer or "Could not find an answer."
        )
        