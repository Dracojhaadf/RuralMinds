from flask import Flask, request, jsonify, render_template
import os
import re
import tempfile
import uuid
import numpy as np
import fitz  # PyMuPDF
import faiss
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

app = Flask(__name__)

# State
current_chunks = None
current_embeddings = None
current_index = None
current_model = None
qa_pipeline = None

# Ensure NLTK punkt
try:
	_nltk_tokenizer = nltk.data.find('tokenizers/punkt')
except LookupError:
	nltk.download('punkt')

# Models
print("Loading sentence transformer model...")
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading QA model...")
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
qa_pipeline = pipeline(
	"text2text-generation",
	model=model,
	tokenizer=tokenizer,
	device_map="auto"
)


def extract_pdf(pdf_path):
	# Extract text from PDF pages
	doc = fitz.open(pdf_path)
	pages = []
	for i, page in enumerate(doc):
		text = page.get_text("text")
		pages.append({"page": i + 1, "text": text})
	return pages


def clean_text(text: str) -> str:
	# Normalize whitespace and remove non-ASCII
	text = re.sub(r"\s+", " ", text)
	text = re.sub(r"[^\x00-\x7F]+", " ", text)
	return text.strip()


def sentence_based_chunking(text: str, max_sentences: int = 5, overlap: int = 2):
	sentences = sent_tokenize(text)
	chunks = []
	step = max(1, max_sentences - overlap)
	for i in range(0, len(sentences), step):
		chunk = " ".join(sentences[i:i + max_sentences])
		if chunk:
			chunks.append(chunk)
	return chunks


def word_overlap_chunking(text: str, max_words: int = 200, overlap: int = 40):
	# Fallback chunker using word windows with overlap
	if max_words <= overlap:
		overlap = max(0, max_words // 4)
	words = text.split()
	chunks = []
	step = max(1, max_words - overlap)
	for start in range(0, len(words), step):
		window = words[start:start + max_words]
		if not window:
			break
		chunks.append(" ".join(window))
	return chunks


def build_prompt(chunks, question, style_instruction=""):
	context = "\n".join(chunks)
	style_line = f"Answer style: {style_instruction}\n" if style_instruction else ""
	prompt = f"""
You are a helpful teaching assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't know based on the context."
{style_line}
Context:
{context}

Question: {question}

Answer:
"""
	return prompt


def process_document(file_path):
	global current_chunks, current_embeddings, current_index, current_model
	try:
		print("Extracting text from PDF...")
		pdf_pages = extract_pdf(file_path)
		raw_text = " ".join([p["text"] for p in pdf_pages if p["text"]])

		print("Cleaning text...")
		cleaned_text = clean_text(raw_text)

		print("Creating chunks...")
		# Prefer sentence chunks; fallback to word chunks if too few
		chunks = sentence_based_chunking(cleaned_text, max_sentences=5, overlap=2)
		if len(chunks) < 2:
			chunks = word_overlap_chunking(cleaned_text, max_words=200, overlap=40)

		print("Generating embeddings...")
		embeddings = sentence_model.encode(chunks, show_progress_bar=True)
		embeddings = np.array(embeddings).astype("float32")

		faiss.normalize_L2(embeddings)
		d = embeddings.shape[1]
		index_flat = faiss.IndexFlatIP(d)
		index = faiss.IndexIDMap(index_flat)
		ids = np.arange(len(embeddings)).astype('int64')
		index.add_with_ids(embeddings, ids)

		current_chunks = chunks
		current_embeddings = embeddings
		current_index = index
		current_model = sentence_model

		return {"success": True, "message": f"Document processed successfully! Created {len(chunks)} chunks.", "chunks_count": len(chunks)}
	except Exception as e:
		return {"success": False, "message": f"Error processing document: {str(e)}"}


@app.route('/')
def index():
	return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_document():
	if 'file' not in request.files:
		return jsonify({"success": False, "message": "No file uploaded"})
	file = request.files['file']
	if file.filename == '':
		return jsonify({"success": False, "message": "No file selected"})
	if not file.filename.lower().endswith('.pdf'):
		return jsonify({"success": False, "message": "Please upload a PDF file"})

	filename = str(uuid.uuid4()) + '.pdf'
	file_path = os.path.join(tempfile.gettempdir(), filename)
	file.save(file_path)
	try:
		result = process_document(file_path)
		os.remove(file_path)
		return jsonify(result)
	except Exception as e:
		if os.path.exists(file_path):
			os.remove(file_path)
		return jsonify({"success": False, "message": f"Error processing file: {str(e)}"})


@app.route('/ask', methods=['POST'])
def ask_question():
	global current_chunks, current_index, current_model, qa_pipeline
	if current_chunks is None or current_index is None:
		return jsonify({"success": False, "message": "Please upload a document first"})

	data = request.get_json()
	question = data.get('question', '').strip()
	if not question:
		return jsonify({"success": False, "message": "Please provide a question"})

	# Determine requested style
	q_lower = question.lower()
	is_brief = any(k in q_lower for k in ["brief", "briefly", "short", "concise"])
	is_detailed = any(k in q_lower for k in ["detail", "detailed", "elaborate", "explain fully", "in depth", "in-depth"])
	if is_brief and not is_detailed:
		style_instruction = "Be concise. Provide a brief 2-3 sentence answer."
		max_tokens = 100
	elif is_detailed and not is_brief:
		style_instruction = "Be thorough. Provide a detailed, structured explanation."
		max_tokens = 300
	else:
		style_instruction = ""
		max_tokens = 180

	try:
		# Embed query and retrieve top-1
		query_embedding = current_model.encode([question]).astype('float32')
		faiss.normalize_L2(query_embedding)
		distances, indices = current_index.search(query_embedding, 1)

		top_score = float(distances[0][0]) if len(distances) and len(distances[0]) else -1.0
		similarity_threshold = 0.35
		if top_score < similarity_threshold:
			return jsonify({
				"success": True,
				"answer": "Please give the question from the uploaded document.",
				"similarity_scores": [top_score],
				"retrieved_chunks": []
			})

		retrieve_chunks = [current_chunks[i] for i in indices[0][:1]]

		prompt = build_prompt(retrieve_chunks, question, style_instruction)
		result = qa_pipeline(prompt, max_new_tokens=max_tokens)[0]["generated_text"]

		return jsonify({
			"success": True,
			"answer": result,
			"similarity_scores": [top_score],
			"retrieved_chunks": retrieve_chunks
		})
	except Exception as e:
		return jsonify({"success": False, "message": f"Error processing question: {str(e)}"})


@app.route('/status')
def get_status():
	global current_chunks
	return jsonify({
		"document_loaded": current_chunks is not None,
		"chunks_count": len(current_chunks) if current_chunks else 0
	})


if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0', port=5000)