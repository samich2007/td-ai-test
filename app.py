from flask import Flask, request, jsonify, send_from_directory
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__, static_url_path="", static_folder=".")

# Laad model, index en documenten
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("index.faiss")
docs = np.load("docs.npy", allow_pickle=True)

@app.route("/")
def serve_homepage():
    return send_from_directory(".", "ai_homepage_travel_diaries.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    vraag = data.get("vraag", "")

    if not vraag:
        return jsonify({"antwoord": "Geen vraag ontvangen."})

    # Embed en zoek
    vraag_embedding = model.encode([vraag], convert_to_numpy=True)
    _, I = index.search(vraag_embedding, k=1)
    relevant_doc = docs[I[0][0]]

    # Prompt bouwen
    prompt = f"Beantwoord de volgende vraag op basis van deze tekst:\n\n{relevant_doc}\n\nVraag: {vraag}"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Je bent een reisexpert die vragen beantwoordt op basis van reiservaringen."},
                {"role": "user", "content": prompt}
            ]
        )
        antwoord = response["choices"][0]["message"]["content"]
        return jsonify({"antwoord": antwoord})
    except Exception as e:
        return jsonify({"antwoord": f"Fout bij AI-aanvraag: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)