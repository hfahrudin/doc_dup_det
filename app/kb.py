from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import os
from typing import List
from langchain.docstore.document import Document

KB_FOLDER_DEFAULT = "/media/faiss_index"


class KnowledgeBaseManager:
    def __init__(self, kb_folder: str = KB_FOLDER_DEFAULT):
        self.kb_folder = kb_folder
        self.embeddings = OpenAIEmbeddings()

        # Load or create FAISS index
        if os.path.exists(kb_folder) and os.listdir(kb_folder):
            print("Loading existing FAISS index...")
            self.vectorstore = FAISS.load_local(kb_folder, self.embeddings)
        else:
            print("FAISS index does not exist. Creating empty index...")
            self.vectorstore = FAISS.from_documents([], self.embeddings)
            os.makedirs(kb_folder, exist_ok=True)
            self.vectorstore.save_local(kb_folder)
            print("Empty FAISS index created and saved.")

    def add_documents(self, docs: List[Document]):
        """
        Add documents to FAISS index and save.
        """
        if not docs:
            return
        self.vectorstore.add_documents(docs)
        self.vectorstore.save_local(self.kb_folder)
        print(f"Added {len(docs)} documents to FAISS index.")

    def delete_documents(self, doc_ids: List[str]):
        """
        Delete documents from FAISS index by metadata ID.
        FAISS itself doesn't support deletion, so we rebuild the index.
        Assumes each Document has a unique 'id' in metadata.
        """
        all_docs = self.vectorstore.get_all_documents()
        remaining_docs = [d for d in all_docs if d.metadata.get("id") not in doc_ids]

        # Rebuild FAISS index
        self.vectorstore = FAISS.from_documents(remaining_docs, self.embeddings)
        self.vectorstore.save_local(self.kb_folder)
        print(f"Deleted {len(doc_ids)} documents from FAISS index.")

    def search(self, query: str, k: int = 5):
        """
        Simple similarity search.
        """
        return self.vectorstore.similarity_search(query, k=k)