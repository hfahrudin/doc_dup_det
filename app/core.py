from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os
from typing import List
from langchain.docstore.document import Document
from schema import *
from chunking import chunk_content_semantic, chunk_content_metadata

KB_FOLDER_DEFAULT = "media/faiss_index"


class KnowledgeBaseManager:
    def __init__(self, kb_folder: str = KB_FOLDER_DEFAULT):
        self.kb_folder = kb_folder
        self.embeddings = OpenAIEmbeddings()

        # Load or create FAISS index
        if os.path.exists(kb_folder) and os.listdir(kb_folder):
            print("Loading existing FAISS index...")
            self.vectorstore = FAISS.load_local(kb_folder, self.embeddings,     allow_dangerous_deserialization=True)
        else:
            # Create a dummy document to initialize FAISS
            dummy_doc = Document(page_content="init", metadata={"id": "dummy"})
            self.vectorstore = FAISS.from_documents([dummy_doc], self.embeddings)

            os.makedirs(kb_folder, exist_ok=True)
            self.vectorstore.save_local(kb_folder)
            print("Empty FAISS index created and saved.")
            
    async def add_documents(self, payload: AddContentRequest):
        """
        Add documents to FAISS index and save.
        Returns the new document ID and status.
        """
 
        docs, doc_id_base = chunk_content_semantic(payload, self.embeddings)


        self.vectorstore.add_documents(docs)
        self.vectorstore.save_local(self.kb_folder)

        return {"status": "success", "id": doc_id_base}
    

    async def delete_documents(self, payload : DeleteContentRequest):
        """
        Delete documents from FAISS index by metadata ID.
        FAISS itself doesn't support deletion, so we rebuild the index.
        Assumes each Document has a unique 'id' in metadata.
        """
        return ""

    async def duplicate_search(self, payload: InvokeRequest):
        """
        Simple similarity search.
        """
        return ""
    

    async def get_all_documents(self):
        """
        Return all documents in the FAISS index with their content and metadata.
        """
        # Access all documents from FAISS docstore
        all_docs = list(self.vectorstore.docstore._dict.values())

        # Convert to simple dicts for easier use
        result = [
            {
                "id": doc.metadata.get("id"),
                "content": doc.page_content,
                "category": doc.metadata.get("category"),
                "tags": doc.metadata.get("type")
            }
            for doc in all_docs
            if doc.metadata.get("id") != "dummy"  # ignore dummy doc
        ]

        return {"status": "success", "documents": result}