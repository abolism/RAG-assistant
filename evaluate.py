from retriever import Retriever
from generator import Generator


class Evaluation:
    def __init__(self, retriever:Retriever, generator:Generator, test_data):
        self.retriever = retriever
        self.generator = generator
        self.data = test_data

    def evaluate(self):
        correct_retrievals = 0
        for sample in self.data:
            q = sample["query"]
            gold = sample["gold"]

            retrieved = self.retriever.retrieve(q, 2)
            docs = [doc for doc , _ in retrieved]

            if any(word.lower() in " ".join(docs).lower() for word in gold.split()):
                correct_retrievals += 1

            answer = self.generator.generate(q, docs)

            print("\n📌 Query:", q)
            print("📑 Retrieved:", docs)
            print("🤖 Answer:", answer)
            print("✅ Gold:", gold)

        print(f"\nRetriever Recall@2: {correct_retrievals}/{len(self.data)}")
            