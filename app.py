from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import uuid
import json
from pathlib import Path
from werkzeug.security import generate_password_hash, check_password_hash
from digi_note_maker import process_pdf, process_image, clean_ocr_text, beautify_with_gemini, save_as_pdf

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
USER_FOLDER = "users"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(USER_FOLDER, exist_ok=True)


# ========= AUTH ROUTES =========
@app.route("/api/auth/signup", methods=["POST"])
def signup():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")
    name = data.get("name", "")

    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400

    user_file = os.path.join(USER_FOLDER, f"{email}.json")
    if os.path.exists(user_file):
        return jsonify({"error": "Email already registered"}), 409

    hashed_pw = generate_password_hash(password)
    user_data = {"id": str(uuid.uuid4()), "email": email, "name": name, "password": hashed_pw}

    with open(user_file, "w") as f:
        json.dump(user_data, f)

    return jsonify({"message": "Signup successful", "user": {k: v for k, v in user_data.items() if k != "password"}}), 200


@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    user_file = os.path.join(USER_FOLDER, f"{email}.json")
    if not os.path.exists(user_file):
        return jsonify({"error": "User not found"}), 404

    with open(user_file, "r") as f:
        user_data = json.load(f)

    if not check_password_hash(user_data["password"], password):
        return jsonify({"error": "Invalid credentials"}), 401

    return jsonify({"message": "Login successful", "user": {k: v for k, v in user_data.items() if k != "password"}}), 200


# ========= FILE + NOTE ROUTE =========
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
    app.run(host="0.0.0.0", port=1435, ssl_context=("certs/cert.pem", "certs/key.pem"))
