from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from typing import List
from langchain.docstore.document import Document
import time
from schema import *



def chunk_content_semantic(payload: AddContentRequest, embedding_model: OpenAIEmbeddings) -> List[Document]:
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

    content = payload.content
    category = payload.category
    type = payload.type
    language = payload.language

    doc_id_base = str(int(time.time() * 1000))
    documents = []
    chunker = SemanticChunker(
        embeddings=embedding_model,
        min_chunk_size=256
    )
    chunks = chunker.split_text(content)
    for i, chunk in enumerate(chunks):
        doc_id = f"{doc_id_base}_{i}"
        embedding_vector = embedding_model.embed_query(chunk)  # generate embedding

        doc = Document(
            page_content=chunk,
            metadata={
                "id": doc_id,
                "embedding": embedding_vector,
                "category": category
            }
        )
        documents.append(doc)

    return documents, doc_id_base


def chunk_content_metadata(content: str, type:str, language: str):
    """
    Chunk content based on type.
    For 'text', split by paragraphs.
    For 'code', split by lines.
    """
    if type == "text":
        # Split by double newlines for paragraphs
        chunks = [para.strip() for para in content.split("\n\n") if para.strip()]
    elif type == "code":
        # Split by single newlines for code lines
        chunks = [line.strip() for line in content.split("\n") if line.strip()]
    else:
        # Default to whole content if unknown type
        chunks = [content.strip()] if content.strip() else []
    
    return chunks