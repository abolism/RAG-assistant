# retriever.py
import os
os.environ["USE_TF"] = "0"
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

class Retriever:
    def __init__(self):
        # self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        # self.embedding_model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
        # self.embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        self.embedding_model = SentenceTransformer("msmarco-distilbert-base-v4")
        # self.chroma_client = chromadb.Client(Settings(
        #     chroma_db_impl="duckdb+parquet",
        #     persist_directory=None
        # ))
        self.chroma_client = chromadb.Client()

        try:
            self.chroma_client.delete_collection("documents")
        except:
            pass  # ignore if doesn't exist

        
        self.collection = self.chroma_client.get_or_create_collection(name="documents", metadata={"hnsw:space":"cosine"})

    def list_documents(self):
        return self.collection.count()
        
    def chunk_text(self, text, chunk_size=300, overlap=50):
        """
        Split text into overlapping chunks.
        chunk_size: number of words per chunk
        overlap: number of words to overlap between chunks
        """
        words = text.split()
        chunks = []
        start = 0

        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += chunk_size - overlap  # slide with overlap

        return chunks

    def add_documents(self, docs, ids=None, chunk_size=200, overlap=50):
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(docs))]

        all_chunks, all_ids = [], []
        for base_id, doc in zip(ids, docs):
            chunks = self.chunk_text(doc, chunk_size, overlap)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_ids.append(f"{base_id}_chunk{i}")

        embeddings = self.embedding_model.encode(all_chunks).tolist()

        self.collection.add(
            documents=all_chunks,
            embeddings=embeddings,
            ids=all_ids
        )

    # def retrieve(self, query, top_k=3):
    #     query_embedding = self.embedding_model.encode([query]).tolist()[0]
    #     results = self.collection.query(
    #         query_embeddings=[query_embedding],
    #         n_results=top_k,
    #         include=["documents", "distances"]
    #     )
    #     return results["documents"][0]
    def retrieve(self, query, k=3):
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        docs = results["documents"][0]
        scores = results["distances"][0]  # similarity scores
        return list(zip(docs, scores))  # [(doc, score), ...]

