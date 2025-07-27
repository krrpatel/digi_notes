from flask import Flask, render_template, request, jsonify, send_from_directory, redirect
import os
from main import run_pipeline  # your OCR & beautify logic
from chatbot import askbot  # your chatbot logic

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

pipelines = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_file():
    file = request.files['file']
    use_ai = 'use_ai' in request.form

    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(input_path)

    output_pdf_path = run_pipeline(input_path, use_ai)

    if not output_pdf_path:
        return jsonify({'error': 'Processing failed'}), 500

    uid = os.path.splitext(os.path.basename(output_pdf_path))[0]

    return jsonify({
        'download_url': f'/download/{uid}.pdf',
        'chat_url': f'/chat/{uid}'
    })

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)

@app.route('/chat/<uid>')
def chat(uid):
    return render_template('chat.html', uid=uid)

@app.route("/ask/<uid>", methods=["POST"])
def ask_question(uid):
    if uid not in pipelines:
        pipelines[uid] = askbot(uid)

    pipeline_fn = pipelines[uid]
    data = request.get_json()
    question = data.get("question")
    use_ai = data.get("use_ai", False)

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        answer = pipeline_fn(question, use_ai=use_ai)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": f"Exception occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1435)  # or 443 if using SSL
