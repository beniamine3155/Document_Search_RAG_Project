
from src.nodes.nodes import RAGNodes
from src.state.rag_state import RAGState
from langgraph.graph import StateGraph, END

class GraphBuilder:
    """Builds and manages the RAG operation graph."""

    def __init__(self, retriever, llm):
        """
        Initializes the GraphBuilder with a retriever and a language model (LLM).
        Args:
            retriever: The document retriever to fetch relevant documents.
            llm: The language model to generate responses.
        """
        self.nodes = RAGNodes(retriever, llm)
        self.state = None

    
    def build(self):
        """
        Builds the RAG operation graph.
        returns:
            Compiled graph instance.
        """

        builder = StateGraph(RAGState)

        # Create nodes
        builder.add_node("retriever", self.nodes.retrieve_docs)
        builder.add_node("generator", self.nodes.generate_answer)

        # set the entry point 
        builder.set_entry_point("retriever")

        # Add edges
        builder.add_edge("retriever", "generator")
        builder.add_edge("generator", END)

        # Compile the graph
        self.graph = builder.compile()
        return self.graph
    
    def run(self, question: str) -> dict:
        """
        Runs the RAG operation graph with the given question.
        Args:
            question (str): The input question to process.
        Returns:
            dict: The final state containing the answer and other details.
        """
        if self.graph is None:
            self.build()

        initial_state = RAGState(question=question)



