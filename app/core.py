from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os
from typing import List
from langchain.docstore.document import Document
from schema import *
from chunking import chunk_content_semantic, chunk_content_token
from grader import score_md_chunk, symmetric_overlap_func

import numpy as np

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
 
        docs, doc_id_base = await chunk_content_semantic(payload, self.embeddings)


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
        category = payload.category
        top_docs = payload.top_docs      # ✅ Number of final docs to keep
        top_nchunk = payload.top_nchunk  # ✅ Number of chunks to consider
        top_k = payload.top_k    

        all_docs = {}
        #=================== FAST ===================

        # Score each chunk
        chunks = await chunk_content_token(payload)
        total_chunks = len(chunks)

        chunk_scores = [
            (chunk, score_md_chunk(chunk, i, total_chunks))
            for i, chunk in enumerate(chunks)
        ]

        # Sort by score, descending
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        top_chunks = chunk_scores[:top_nchunk]

        for chunk, _ in top_chunks:

        #=================== SEMANTIC ===================
        # docs, doc_id_base = chunk_content_semantic(payload, self.embeddings)
        # top_chunks = [d.page_content for d in docs]
        # for chunk in top_chunks:


        # for chunk, _ in top_chunks:
            results = await self.vectorstore.asimilarity_search_with_relevance_scores(
                chunk,  # searching using the chunk content
                k=top_k,    # top 5 similar items
                filter={"category": category}
            )

            # Attach chunk score to results
            for result_content, similarity_score in results:
                docs_id = result_content.metadata.get("id")

                if docs_id not in all_docs:
                    all_docs[docs_id] = {
                        "accumulated_score": float(similarity_score),
                        "chunk": [result_content.metadata.get("chunk_index")]
                    }
                else:
                    all_docs[docs_id]["accumulated_score"]+=float(similarity_score)
                    all_docs[docs_id]["chunk"].append(result_content.metadata.get("chunk_index"))

        sorted_docs = sorted(
            all_docs.items(),
            key=lambda x: x[1]["accumulated_score"],
            reverse=True
        )

        similar_docs_candidate = dict(sorted_docs[:top_docs])

        target_vector = [self.embeddings.embed_query(c[0]) for c in top_chunks] 

        candidates = {}
        for d in similar_docs_candidate.keys():
            doc_sim_score = await self.evaluate_candidate(target_vector, d)
            candidates[d] = doc_sim_score
        return candidates
    
    async def evaluate_candidate(self, target_vector, candidate_id):
        #TODO:Better find a way to optimize chunked stuff
        
        candidate_vector = []

        # FAISS index stores vectors in the same order as docstore keys
        for faiss_id, docstore_id in self.vectorstore.index_to_docstore_id.items():
            doc = self.vectorstore.docstore.search(docstore_id)
            if doc.metadata.get("id") == candidate_id:
                # Step 2: reconstruct vector using FAISS ID
                vec = self.vectorstore.index.reconstruct(faiss_id)
                candidate_vector.append(vec)

        if not candidate_vector:
            print("No candidate vectors found")
            return 0.0  # or handle as needed


        return symmetric_overlap_func(target_vector, candidate_vector)


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
                "chunk_index": doc.metadata.get("chunk_index")
            }
            for doc in all_docs
            if doc.metadata.get("id") != "dummy"  # ignore dummy doc
        ]

        return {"status": "success", "documents": result}