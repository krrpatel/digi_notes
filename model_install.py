from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from haystack.nodes import EmbeddingRetriever, FARMReader
import os

# -------------------------------
# Paths
# -------------------------------
CHATBOT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHATBOT_READER_MODEL = "deepset/roberta-base-squad2"

GRAMMAR_MODEL = "vennify/t5-base-grammar-correction"

SAVE_CHATBOT_EMBEDDING = "chatbot_models/embedding"
SAVE_CHATBOT_READER = "chatbot_models/reader"
SAVE_GRAMMAR = "local_models/t5-grammar"

# -------------------------------
# Download Chatbot Models
# -------------------------------
def download_chatbot_models():
    print("🔄 Downloading Embedding Retriever model...")
    retriever = EmbeddingRetriever(embedding_model=CHATBOT_EMBEDDING_MODEL)
    retriever.embed_queries(texts=["hello"])  # trigger model loading
    retriever.model.save(SAVE_CHATBOT_EMBEDDING)
    print("✅ Embedding Retriever saved to", SAVE_CHATBOT_EMBEDDING)

    print("🔄 Downloading Reader model...")
    reader = FARMReader(model_name_or_path=CHATBOT_READER_MODEL)
    reader.model.save_pretrained(SAVE_CHATBOT_READER)
    reader.tokenizer.save_pretrained(SAVE_CHATBOT_READER)
    print("✅ Reader model saved to", SAVE_CHATBOT_READER)

# -------------------------------
# Download Grammar Correction Model
# -------------------------------
def download_grammar_model():
    print("🔄 Downloading Grammar Correction model...")
    tokenizer = AutoTokenizer.from_pretrained(GRAMMAR_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(GRAMMAR_MODEL)
    tokenizer.save_pretrained(SAVE_GRAMMAR)
    model.save_pretrained(SAVE_GRAMMAR)
    print("✅ Grammar model saved to", SAVE_GRAMMAR)

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    os.makedirs(SAVE_CHATBOT_EMBEDDING, exist_ok=True)
    os.makedirs(SAVE_CHATBOT_READER, exist_ok=True)
    os.makedirs(SAVE_GRAMMAR, exist_ok=True)

    download_chatbot_models()
    download_grammar_model()

    print("✅ All models installed successfully.")
