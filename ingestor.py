import os

class DocumentIngestor:
    def __init__(self, retriever):
        self.retriever = retriever

    def ingest_file(self, file_path, chunk_size=200, overlap=50):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")
        
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        doc_id = os.path.basename(file_path)
        self.retriever.add_documents([text], ids=[doc_id], chunk_size=chunk_size, overlap=overlap)
        print(f"Ingested file {file_path}.")

    def ingest_text(self, text, doc_id="dynamic_doc", chunk_size=200, overlap=50):
        self.retriever.add_documents([text], ids=[doc_id], chunk_size=chunk_size, overlap=overlap)
        print(f"Ingested text {doc_id}.")


    