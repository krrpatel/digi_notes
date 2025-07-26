from flask import Flask, render_template, request, send_from_directory
import os
import uuid
from pathlib import Path
from digi_note_maker import process_pdf, process_image, clean_ocr_text, beautify_with_gemini, save_as_pdf

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        uploaded_file = request.files["file"]
        if not uploaded_file:
            return render_template("index.html", error="No file uploaded.")

        ext = Path(uploaded_file.filename).suffix.lower()
        file_id = str(uuid.uuid4())
        input_path = os.path.join(UPLOAD_FOLDER, file_id + ext)
        uploaded_file.save(input_path)

        # OCR + Cleaning
        if ext == ".pdf":
            raw_text = process_pdf(input_path)
        elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
            raw_text = process_image(input_path)
        else:
            return render_template("index.html", error="Unsupported file type.")

        cleaned_text = clean_ocr_text(raw_text)
        beautified = beautify_with_gemini(cleaned_text)

        output_pdf_path = os.path.join(OUTPUT_FOLDER, f"{file_id}_cleaned.pdf")
        save_as_pdf(output_pdf_path, beautified)

        return render_template("index.html", download_path=f"/download/{file_id}_cleaned.pdf")

    return render_template("index.html")

@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=1435)
