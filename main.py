from retriever import Retriever, HybridRetriever
from generator import Generator
from ingestor import DocumentIngestor as Ingestor
from evaluate import Evaluation
from speech import SpeechSynthesis


if __name__ == "__main__":
    rag = Retriever()
    rag.add_documents([
        "Machine learning is a field of artificial intelligence.",
        "Deep learning uses neural networks to model data.",
        "Natural Language Processing helps computers understand human language.",
        "FAISS is a library for efficient similarity search of embeddings.",
        "Nowadays the margin between different fields of DL is very narrow, leading to the creation of inter-fields such as Multimodal Natural Language Processing.",
        "Deep learning algorithms can be applied in different contexts such as vision, text, and speech."
    ])
    rag.add_documents([
        "Our company provides AI-driven solutions. We specialize in language models and retrieval systems.",
        "Customer satisfaction is our priority. Text-to-speech and OCR are also supported."
    ], ids=["doc1", "doc2"], chunk_size=200, overlap=50)

    dl_notes = [
        """Gradient Descent is an optimization algorithm used to minimize loss functions. 
        It updates parameters θ by moving in the opposite direction of the gradient:
        θ = θ - η * ∇L(θ), where η is the learning rate.""",

        """Convolutional Neural Networks (CNNs) are specialized for image processing. 
        They use convolutional layers with filters/kernels to capture spatial features, 
        followed by pooling layers for dimensionality reduction.""",

        """Recurrent Neural Networks (RNNs) are designed for sequential data. 
        The hidden state h_t is updated as h_t = f(Wxh * x_t + Whh * h_{t-1}). 
        However, RNNs suffer from vanishing gradients, which LSTMs and GRUs solve.""",

        """Dropout is a regularization technique where units are randomly set to zero 
        during training. This prevents overfitting by ensuring the network does not 
        rely on specific neurons.""",

        """Batch Normalization normalizes activations across the batch, improving training stability. 
        It computes: y = (x - μ) / √(σ² + ε) * γ + β, where μ and σ² are batch mean and variance.""",

        """Transformers use self-attention mechanisms. The scaled dot-product attention is:
        Attention(Q,K,V) = softmax(QK^T / √d_k) V. 
        Multi-head attention allows the model to attend to different representations simultaneously.""",

        """Cross-entropy loss is widely used for classification tasks. 
        For true label y and prediction p, the loss is: L = -Σ y_i log(p_i).""",

        """Overfitting occurs when the model performs well on training data but poorly on test data. 
        Common solutions include regularization, dropout, early stopping, and more data.""",

        """Autoencoders are neural networks trained to reconstruct input data. 
        They consist of an encoder that compresses input into a latent space, 
        and a decoder that reconstructs the input from the latent representation.""",

        """GANs (Generative Adversarial Networks) consist of a generator G and a discriminator D. 
        The generator tries to fool the discriminator, while the discriminator distinguishes 
        real from fake samples. They are trained in a minimax game: 
        min_G max_D V(D,G) = E[log D(x)] + E[log(1 - D(G(z)))]"""
    ]

    dl_ret = Retriever()
    hybrid_retriever = HybridRetriever(dl_notes)
    dl_ret.add_documents(dl_notes, chunk_size=100, overlap=15)
    # dl_ret.peek()
    gen = Generator()
    speaker = SpeechSynthesis()
    # # Test dataset
    # test_data = [
    #     {
    #     "query": "Explain gradient descent and its update rule.",
    #     "gold": "Gradient Descent updates parameters by moving opposite the gradient: θ = θ - η ∇L(θ)."
    #     },
    #     {
    #     "query": "What is dropout and why is it useful?",
    #     "gold": "Dropout randomly sets units to zero during training to prevent overfitting."
    #     },
    #     {
    #     "query": "Write the formula for batch normalization.",
    #     "gold": "y = (x - μ) / √(σ² + ε) * γ + β"
    #     }
    # ]
    # evaluator = Evaluation(dl_ret, gen, test_data)
    # evaluator.evaluate()

    # exit()

    ing = Ingestor(dl_ret) 
    ing.ingest_text("A growing direction in generative modeling is multimodality, where models learn joint distributions over different data types, such as text, images, and audio. By aligning modalities in a shared latent space, these models can perform tasks like text-to-image synthesis, audio-driven animation, and cross-modal retrieval. Recent advances, including CLIP-guided diffusion and large-scale multimodal transformers, have shown remarkable capability in generating coherent and semantically aligned outputs across diverse data forms.",chunk_size=100, overlap=15)
    file_path = "sample_text.txt"
    ing.ingest_file(file_path, chunk_size=100, overlap=15)

    # query = "What does the company specialize in?"

    # Example exam-like queries
    queries = [
        "Explain gradient descent and its update rule.",
        "How do CNNs reduce dimensionality?",
        "Why do RNNs suffer from vanishing gradients?",
        "What is dropout and why is it useful?",
        "Write the formula for batch normalization.",
        "Explain the attention mechanism in transformers.",
        "When is cross-entropy loss used?",
        "How can we prevent overfitting?",
        "What are autoencoders used for?",
        "What is the objective function of GANs?"
    ]

    # for q in queries:
    #     retrieved = dl_ret.retrieve(q, 2)
    #     retrieved = [t[0] for t in retrieved]
    #     # retrieved = ["", ""]
    #     # print(q, retrieved)
    #     answer = gen.generate(q, retrieved)
    #     print("\n📌 Query:", q)
    #     print("📑 Retrieved:", retrieved)
    #     print("🤖 Answer:", answer)

    q = input("\n Ask a question: ")
    retrieved = dl_ret.retrieve(q, k=2)
    h_retrieved = hybrid_retriever.retrieve(q, top_k=3, alpha=0.6)
    for doc, score in h_retrieved:
        print(f"Score: {score:.4f} | Doc: {doc[:80]}...")
    docs = [doc for doc,_ in retrieved]
    print(docs)
    exit()
    answer = gen.generate(q, docs)

    print("\n Answer:", answer)
    speaker.speak(answer)


    
    # results = rag.retrieve(query, 2)
    # print("Query:", query)
    # print("Retrieved:", results)

    # generator = Generator()
    # answer = generator.generate(query, results)
    # print(answer)