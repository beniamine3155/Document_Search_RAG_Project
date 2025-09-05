"""State management for the RAG application."""
from typing import List
from pydantic import BaseModel
from langchain.schema import Document

class RAGState(BaseModel):
    """Class to manage the state of the RAG application."""
    question: str
    retrieved_docs: List[Document] = []
    answer: str = ""