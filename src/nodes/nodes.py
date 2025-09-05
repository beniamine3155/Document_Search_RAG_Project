
from src.state.rag_state import RAGState

class RAGNodes:
    """Contains all nodes for RAG operations."""

    def __init__(self, retriever, llm):
        """
        Initializes the RAGNodes with a retriever and a language model (LLM).
        Args:
            retriever: The document retriever to fetch relevant documents.
            llm: The language model to generate responses.
        """
        self.retriever = retriever
        self.llm = llm

    def retrieve_docs(self, state:RAGState)-> RAGState:
        """
        Retrieves documents based on the query in the state.
        Args:
            state (RAGState): The current state containing the query.
        Returns:
            RAGState: Updated state with retrieved documents.
        """
        docs = self.retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=docs
        )
    

    def generate_answer(self, state:RAGState) -> RAGState:
        """
        Generates an answer based on the question and retrieved documents in the state.
        Args:
            state (RAGState): The current state containing the question and retrieved documents.
        Returns:
            RAGState: Updated state with the generated answer.
        """
        # Combine the question and retrieved documents into a single prompt
        context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])

        # create the final prompt
        prompt = f"""Answer the question based on the context below.
            Context: {context}
            Question: {state.question}"""
        
        response = self.llm.invoke(prompt)
        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=response
        )

    

