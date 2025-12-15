from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from typing import List
from langchain_core.documents import Document
import time
from schema import *
from unstructured.partition.md import partition_md
from unstructured.cleaners.core import clean
from langchain_text_splitters import TokenTextSplitter



chunk_size_type_map = {
    "user_guide": 512,
    "qna": 128,
    "blog": 256
}



async def  chunk_content_semantic(payload: AddContentRequest, embedding_model: OpenAIEmbeddings) -> List[Document]:
    """
    Chunk content by paragraphs and generate semantic embeddings for each chunk.
    
    Args:
        content (str): The raw content to chunk.
        embedding_model (OpenAIEmbeddings): OpenAI embeddings model instance.
    
    Returns:
        List[Document]: List of Documents with embeddings stored in metadata.
    """
    # Split content by paragraphs

    # Generate a unique ID (could use UUID, here just int timestamp for simplicity)

    raw_content = payload.content
    category = payload.category

    content = preprocess_md_file(raw_content)

    doc_id_base = str(int(time.time() * 1000))
    documents = []
    chunker = SemanticChunker(
        embeddings=embedding_model,
        min_chunk_size=256
    )
    chunks = chunker.split_text(content)
    for i, chunk in enumerate(chunks):
        doc_id = f"{doc_id_base}"
        embedding_vector = embedding_model.embed_query(chunk)  # generate embedding

        doc = Document(
            page_content=chunk,
            metadata={
                "id": doc_id,
                "embedding": embedding_vector,
                "category": category,
                "chunk_index": i
            }
        )
        documents.append(doc)

    return documents, doc_id_base

async def  chunk_content_token(payload: AddContentRequest) -> List[str]:
    """
    Convert markdown content into LangChain Documents with metadata.
    """

    # Clean and get content
    raw_content = payload.content
    type = payload.type

    content = preprocess_md_file(raw_content)
    # Initialize TokenTextSplitter

    chunk_size = chunk_size_type_map.get(type)
    chunk_overlap = int(chunk_size * 0.1)  # 10% overlap
    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_text(content)
    return chunks


def preprocess_md_file(raw_content: str) -> str:
    """
    Read and clean a markdown file.
    
    Args:
        file_path (str): Path to the markdown file. 
    """
        # Clean and get content

    # Use partition_md directly with content string
    elements = partition_md(text=raw_content)

    # Convert to LangChain Documents
    cleaned_content = ""
    for e in elements:
        if e.category.lower() in ["title", "listitem", "uncategorizedtext", "narrativetext"]:
            cleaned_content += e.text

    cleaned_content = clean(cleaned_content)
    return cleaned_content
