"""Vector store module for document embeddings and retrieval."""

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from typing import List

class VectorStore:
    """Class to handle vector store operations."""

    def __init__(self):
        """Initialize the vector store with OpenAI embeddings."""
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.retriever = None

    
    def create_vectorstore(self, documents: List[Document]):
        """"
        Create a vector store from a list of documents.
        Args:
            documents (List[Document]): List of documents to be added to the vector store.
        """
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        self.retriever = self.vector_store.as_retriever()

    def get_retriever(self):
        """Get the retriever for the vector store."""
        if self.retriever is None:
            raise ValueError("Vector store not created. Call create_vector_store first.")
        return self.retriever
    
    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieve documents similar to the query.
        Args:
            query (str): The query string.
            k (int): Number of similar documents to retrieve.
        Returns:
            List[Document]: List of retrieved documents.
        """
        if self.retriever is None:
            raise ValueError("Vector store not created. Call create_vector_store first.")
        return self.retriever.invoke(query)[:k]
