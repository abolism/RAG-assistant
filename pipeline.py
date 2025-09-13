# pipeline.py
class RAGPipeline:
    def __init__(self, retriever=None, llm=None, tts=None):
        self.retriever = retriever
        self.llm = llm
        self.tts = tts

    def run(self, query, return_speech=False):
        docs = self.retriever.retrieve(query) if self.retriever else []
        answer_text = self.llm.generate(query, docs) if self.llm else "No LLM available"
        if return_speech and self.tts:
            answer_speech = self.tts.speak(answer_text)
            return answer_text, answer_speech
        return answer_text

