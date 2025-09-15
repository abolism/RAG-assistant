from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline
# from openai import OpenAI
# client = OpenAI()

class Generator:
    def __init__(self, model="google/flan-t5-small", device="cpu", isopenai=False):
        # self.llm = model
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.isopenai = isopenai
        if not self.isopenai:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model,
                device_map="auto",
                load_in_8bit=True  # needs bitsandbytes
            )

            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if device == "cuda" else -1
            )
        self.device = device
        self.model.to(device)

    def generate(self, query, docs, max_new_tokens=200):
        context = "\n\n".join([doc for doc, _ in docs])
        prompt = f"""
        You are a knowledgeable assistant. 
        Use only the following context to answer the question.
        If the answer cannot be found, say "I don’t know".

        Context:
        {context}

        Question:
        {query}

        Answer:
        """
        if not self.isopenai:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        output = self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return output[0]["generated_text"]


    