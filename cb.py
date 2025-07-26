import pdfplumber
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import EmbeddingRetriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline
import warnings
warnings.filterwarnings("ignore")

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

    # Step 2: Init document store
    document_store = InMemoryDocumentStore(embedding_dim=384)

    # Step 3: Embedder (RAG Retriever)
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        use_gpu=True
    )

    # Step 4: Store document
    document_store.write_documents([{"content": full_text}])
    document_store.update_embeddings(retriever)

    # Step 5: Reader (QA Model)
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

    # Step 6: Create QA pipeline
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
        if answers:
            print("🤖 Answer:", answers[0].answer)
        else:
            print("🤖 Sorry, I couldn't find an answer.")

if __name__ == "__main__":
    run_chatbot()
