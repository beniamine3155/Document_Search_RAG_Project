""""Document Processor Module"""

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

from typing import List, Union
from pathlib import Path
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
    PyPDFDirectoryLoader
)


class DocumentProcessor:
    """Class for processing documents from various sources and splitting them into manageable chunks."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """"
        Initialize the DocumentProcessor with specified chunk size and overlap.
        Args:
            chunk_size (int): The size of each text chunk.
            chunk_overlap (int): The overlap between consecutive chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_url(self, url: str) -> List[Document]:
        """
        Load a document from a URL.
        Args:
            url (str): The URL of the document to load.
        Returns:
            List[Document]: A list of processed Document objects.
        """
        loader = WebBaseLoader(url)
        return loader.load()
    
    
    def load_pdf_dir(self, directory: Union[str, Path]) -> List[Document]:
        """
        Load PDF documents from a directory.
        Args:
            directory (Union[str, Path]): The directory containing PDF files.
        Returns:
            List[Document]: A list of processed Document objects.
        """
        loader = PyPDFDirectoryLoader(directory)
        return loader.load()
    

    def load_text(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Load a text document from a file.
        Args:
            file_path (Union[str, Path]): The path to the text file.
        Returns:
            List[Document]: A list of processed Document objects.
        """
        loader = TextLoader(str(file_path), encoding='utf-8')
        return loader.load()
    
    
    def load_pdf(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Load a PDF document from a file.
        Args:
            file_path (Union[str, Path]): The path to the PDF file.
        Returns:
            List[Document]: A list of processed Document objects.
        """
        loader = PyPDFLoader(str(file_path))
        return loader.load()
    

    def load_documents(self, sources: List[str]) -> List[Document]:
        """
        Load documents from URLs, PDF directories, or TXT files
        Args:
            sources: List of URLs, PDF folder paths, or TXT file paths
        Returns:
            List[Document]: A list of processed Document objects.
        """
        docs: List[Document] = []
        for source in sources:
            if source.startswith("http://") or source.startswith("https://"):
                docs.extend(self.load_url(source))

            path = Path("data")
            if path.is_dir():
                docs.extend(self.load_pdf_dir(path)) 
                
            elif path.suffix.lower() == ".txt":
                docs.extend(self.load_text(path))
            else:
                raise ValueError(
                    f"Unsupported source type: {source}. Must be a URL, PDF directory, or TXT file."
                )
        return docs
    

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.
        Args:
            documents (List[Document]): The list of Document objects to split.
        Returns:
            List[Document]: A list of split Document objects.
        """
        return self.splitter.split_documents(documents)
    

    def process_urls(self, urls: List[str]) -> List[Document]:
        """
        Complete processing pipeline for URLs: load and split documents.
        Args:
            urls (List[str]): List of URLs to process.
        Returns:
            List[Document]: A list of processed document chunks.
        """
        docs = self.load_documents(urls)
        return self.split_documents(docs)
        


   