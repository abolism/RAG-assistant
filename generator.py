from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Generator:
    def __init__(self, model="google/flan-t5-small", device="cpu"):
        # self.llm = model
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model)
        self.device = device
        self.model.to(device)

    def generate(self, query, docs, max_new_tokens=100):
        context = " ".join(docs)
        prompt = f"Answer the question using the context.\n Context: {context}, \nQuestion: {query}\n Answer:"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    