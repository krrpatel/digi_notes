import os
import io
import re
import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from google.cloud import vision
from concurrent.futures import ThreadPoolExecutor, as_completed
import google.generativeai as genai
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from pathlib import Path
from chatbot import run_chatbot
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# Ignore DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Ignore UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

def save_as_pdf(output_path, text, title=""):
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import inch

    pdf = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4
    x_margin, y_margin = inch * 0.75, inch * 0.75
    max_width = width - 2 * x_margin
    normal_font = "Helvetica"
    bold_font = "Helvetica-Bold"
    font_size = 11

    lines = text.split("\n")
    y = height - y_margin

    for line in lines:
        line = line.strip()
        if not line:
            y -= 15
            continue

        # Detect markdown bold (**text**) and render in bold
        if line.startswith("**") and line.endswith("**") and len(line) > 4:
            content = line.strip("**").strip()
            font = bold_font
        else:
            content = line
            font = normal_font

        # Wrap long lines manually
        words = content.split()
        current_line = ""
        pdf.setFont(font, font_size)
        for word in words:
            test_line = current_line + word + " "
            if pdf.stringWidth(test_line, font, font_size) < max_width:
                current_line = test_line
            else:
                pdf.drawString(x_margin, y, current_line.strip())
                y -= 15
                current_line = word + " "

        if current_line:
            pdf.drawString(x_margin, y, current_line.strip())
            y -= 15

        if y < y_margin:
            pdf.showPage()
            y = height - y_margin

    pdf.save()

# Set Google API keys
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/home/ubuntu/HACK/vision1.json"
genai.configure(api_key="AIzaSyDezNjrU81jycTLtDmSP-2XkdVV45rdlqA")  # Gemini API key

# Load grammar correction model
print("🔁 Loading OCR model...")
tokenizer = AutoTokenizer.from_pretrained("./local_models/t5-grammar")
model = AutoModelForSeq2SeqLM.from_pretrained("./local_models/t5-grammar")
print("Models Loaded.")

# Initialize Gemini
gemini = genai.GenerativeModel("gemini-2.0-flash")

def detect_text_from_image(image_data):
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_data)
    try:
        response = client.text_detection(image=image)
        if response.error.message:
            print(f"⚠️ API Error: {response.error.message}")
            return ""
        texts = response.text_annotations
        return texts[0].description if texts else ""
    except Exception as e:
        print(f"❌ Vision API failed: {e}")
        return ""

def preprocess_image(image):
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    return Image.fromarray(gray)

def process_page(page, page_number):
    buffer = io.BytesIO()
    page = preprocess_image(page)
    page.save(buffer, format="PNG")
    image_data = buffer.getvalue()
    extracted_text = detect_text_from_image(image_data)
    return (page_number, extracted_text)

def process_pdf(pdf_path):
    pages = convert_from_path(pdf_path, dpi=300)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_page, page, i) for i, page in enumerate(pages, start=1)]
        results = sorted([f.result() for f in as_completed(futures)], key=lambda x: x[0])
    full_text = ""
    for page_number, text in results:
        full_text += f"\n--- Page {page_number} ---\n{text}\n"
    return full_text

def process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = preprocess_image(image)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return detect_text_from_image(buffer.getvalue())

def correct_text_with_model(text):
    paragraphs = text.split('\n')
    cleaned = []
    for para in paragraphs:
        if not para.strip():
            cleaned.append("")
            continue
        input_text = "gec: " + para
        input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(input_ids, max_length=512)
        corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
        cleaned.append(corrected)
    return "\n".join(cleaned)

def format_headings(text):
    lines = text.splitlines()
    formatted = []
    for line in lines:
        line_strip = line.strip()
        if line_strip.isupper() or (len(line_strip.split()) <= 6 and line_strip.istitle()):
            formatted.append(f"**{line_strip}**")
        else:
            formatted.append(line_strip)
    return "\n\n".join(formatted)

def clean_ocr_text(text):
    # Basic cleaning
    text = re.sub(r'[→>*■•●▪◦‣⁃]+', '-', text)  # unify bullets
    text = re.sub(r'[^\x00-\x7F]+', '', text)    # non-ASCII
    text = re.sub(r'\n+', '\n', text)            # extra breaks
    text = re.sub(r'\s{2,}', ' ', text)          # extra spaces
    text = re.sub(r'(?<=\w)\s+(?=\w)', ' ', text)  # broken words

    # Grammar correction
    corrected = correct_text_with_model(text)

    # Heading formatting
    structured = format_headings(corrected)

    return structured

def beautify_with_gemini(raw_text):
    prompt = (
    "You are an OCR post-processor.\n"
    "Please rewrite and clean the following text **without summarizing, explaining, or changing the meaning**.\n"
    "Follow these rules strictly:\n"
    "- Do NOT summarize or interpret the text.\n"
    "- Retain the **original structure** and phrasing.\n"
    "- Fix broken words and clear grammar/spelling issues.\n"
    "- **Remove repeated words** like 'structure structure structure'.\n"
    "- Keep bullet points and headings structured cleanly.\n"
    "- If a term and its full form appear separately (e.g., DBMS, Database Management System), merge them.\n"
    "- Do not add anything beyond the cleaned version.\n\n"
    f"Text:\n{raw_text}")
    try:
        response = gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"⚠️ Gemini formatting failed: {e}")
        return raw_text

def save_text(output_path, text):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

if __name__ == "__main__":
    file_path = input("Enter the full path of the PDF/image file: ").strip().strip('"').strip("'")

    if not os.path.exists(file_path):
        print("❌ File does not exist.")
        exit()

    output_path = os.path.splitext(file_path)[0] + "_gemini_cleaned.txt"

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        print("📄 PDF detected. Processing with OCR...")
        text = process_pdf(file_path)
    elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
        print("🖼️ Image detected. Processing with OCR...")
        text = process_image(file_path)
    else:
        print("❌ Unsupported file type.")
        exit()

    print("🧹 Cleaning OCR output...")
    intermediate_text = clean_ocr_text(text)

    # Ask if user wants AI beautification
    choice = input("✨ Do you want to use Gemini AI to beautify the output? (yes/no): ").strip().lower()
    if choice in ["yes", "y"]:
        print("🔧 Beautifying the text with Gemini...")
        final_output = beautify_with_gemini(intermediate_text)
    else:
        print("💾 Skipping AI beautification. Using cleaned text.")
        final_output = intermediate_text

    pdf_name = "digital_" + Path(file_path).stem + ".pdf"
    pdf_path = os.path.join(os.path.dirname(file_path), pdf_name)
    save_as_pdf(pdf_path, final_output)
    print(f"✅ Cleaned PDF saved to: {pdf_path}")

    # Ask user if they want to use chatbot
    chatbot_choice = input("🤖 Do you want to ask questions using chatbot on this PDF? (yes/no): ").strip().lower()
    if chatbot_choice in ["yes", "y"]:
        print("🔄 Launching chatbot...\n")
        from pathlib import Path
        run_chatbot(pdf_path=pdf_path)  # Uses the earlier defined function
    else:
        print("👋 Done. Exiting.")

