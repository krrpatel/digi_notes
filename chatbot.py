import fitz
import pdfplumber
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import EmbeddingRetriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import google.generativeai as genai
import os
import warnings
warnings.filterwarnings("ignore")

# Configure Gemini API key
genai.configure(api_key="AIzaSyDezNjrU81jycTLtDmSP-2XkdVV45rdlqA")
gemini = genai.GenerativeModel("gemini-2.0-flash")

# Load grammar correction model
print("🔁 Loading OCR model...")
tokenizer = AutoTokenizer.from_pretrained("./local_models/t5-grammar")
model = AutoModelForSeq2SeqLM.from_pretrained("./local_models/t5-grammar")
print("Models Loaded.")

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def create_qa_pipeline_from_pdf(pdf_path):
    # Step 1: Extract PDF text
    full_text = extract_text_from_pdf(pdf_path)
    if not full_text.strip():
        raise ValueError("No readable text found in the PDF.")

    # Step 2: Save the text for Gemini
    with open("parsed_pdf.txt", "w", encoding="utf-8") as f:
        f.write(full_text)

    # Step 3: Init document store
    document_store = InMemoryDocumentStore(embedding_dim=384)

    # Step 4: Embedder (RAG Retriever)
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        use_gpu=True
    )

    # Step 5: Store document
    document_store.write_documents([{"content": full_text}])
    document_store.update_embeddings(retriever)

    # Step 6: Reader (QA Model)
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

    # Step 7: Create QA pipeline
    pipe = ExtractiveQAPipeline(reader=reader, retriever=retriever)
    return pipe

def run_chatbot(pdf_path=None):
    if not pdf_path:
        pdf_path = input("Enter the name of the PDF file (with extension): ").strip()

    print("⏳ Loading and processing PDF...")
    try:
        qa_pipeline = create_qa_pipeline_from_pdf(pdf_path)
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    use_ai = input("🧠 Do you want to use AI (Gemini) for detailed answers? (yes/no): ").strip().lower() in ['yes', 'y']

    # Load full context for Gemini from saved file
    with open("parsed_pdf.txt", "r", encoding="utf-8") as f:
        full_context = f.read()

    print("\n✅ Ready! Ask your questions below (type 'quit' to exit):")
    while True:
        query = input("🧠 You: ")
        if query.lower() in ['quit', 'exit']:
            print("👋 Exiting. Goodbye!")
            break

        prediction = qa_pipeline.run(
            query=query,
            params={"Retriever": {"top_k": 5}, "Reader": {"top_k": 1}}
        )
        answers = prediction['answers']
        context_passages = [doc.content for doc in prediction.get('documents', [])]

        if use_ai:
            prompt = f"Only answer to the point without any extra explanation.\n\nQuestion: {query}\n\nBased on the following document:\n{full_context[:30000]}"
            try:
                gemini_response = gemini.generate_content(prompt)
                ai_answer = gemini_response.text.strip()
                print("🤖AI Answer:", ai_answer)
            except Exception as e:
                print("⚠️ GAI failed. Fallback to extractive QA.")
                if answers:
                    print("🤖 Extractive Answer:", answers[0].answer)
                else:
                    print("🤖 Sorry, I couldn't find an answer.")
        else:
            if answers:
                print("🤖 Answer:", answers[0].answer)
            else:
                print("🤖 Sorry, I couldn't find an answer.")

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def askbot(uid):
    pdf_path = f"outputs/{uid}.pdf"
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Extract full context for Gemini
    full_context = extract_text_from_pdf(pdf_path)

    # Setup Haystack QA Pipeline
    document_store = InMemoryDocumentStore(embedding_dim=384)
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="local_models/chatbot_models/embedding",
        use_gpu=False
    )

    docs = [{"content": full_context}]
    document_store.write_documents(docs)
    document_store.update_embeddings(retriever)

    reader = FARMReader(model_name_or_path="local_models/chatbot_models/reader", use_gpu=False)
    pipeline = ExtractiveQAPipeline(reader, retriever)

    # Define inner function for querying
    def query_fn(question, use_ai=False):
        prediction = pipeline.run(
            query=question,
            params={"Retriever": {"top_k": 3}, "Reader": {"top_k": 1}}
        )
        answers = prediction["answers"]

        if use_ai:
            prompt = f"Answer to the point from the document below.\n\nQuestion: {question}\n\nDocument:\n{full_context[:30000]}"
            try:
                gemini_response = gemini.generate_content(prompt)
                return gemini_response.text.strip()
            except Exception as e:
                print("⚠️ Gemini error, falling back:", e)

        if answers:
            return answers[0].answer
        return "❌ Sorry, I couldn't find an answer."

    return query_fn

if __name__ == "__main__":
    run_chatbot()
