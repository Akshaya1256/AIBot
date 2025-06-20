

from flask import Flask, request, jsonify, send_from_directory
import json
import random
import torch
import requests
from sentence_transformers import SentenceTransformer, util
from difflib import get_close_matches

app = Flask(__name__, static_folder='static')

# Load intents
with open('intents.json', 'r') as file:
    data = json.load(file)

# Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Precompute embeddings
pattern_data = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        embedding = model.encode(pattern, convert_to_tensor=True)
        pattern_data.append({
            'embedding': embedding,
            'response': random.choice(intent['responses'])
        })

# Vocabulary for fuzzy matching
vocab = set()
for intent in data['intents']:
    for pattern in intent['patterns']:
        for word in pattern.lower().split():
            vocab.add(word)

def correct_with_fuzzy(text):
    words = text.lower().split()
    corrected_words = []
    for w in words:
        matches = get_close_matches(w, vocab, n=1, cutoff=0.8)
        corrected_words.append(matches[0] if matches else w)
    return ' '.join(corrected_words)

TOGETHER_API_KEY = "Your_API_Key" 

def generate_fallback_response(user_input):
    try:
        headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant for answering college-related queries."},
                {"role": "user", "content": user_input}
            ],
            "max_tokens": 200,
            "temperature": 0.7,
            "top_p": 0.9
        }

        response = requests.post("https://api.together.xyz/v1/chat/completions", headers=headers, json=data)

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            print("Together API error:", response.text)
            return "Sorry, I couldn't find an answer right now. Please try again later."
    except Exception as e:
        print("Together API exception:", str(e))
        return "An error occurred while generating a response."

def chatbot_response(user_input):
    corrected_input = correct_with_fuzzy(user_input)
    user_embedding = model.encode(corrected_input, convert_to_tensor=True)

    best_score = 0.0
    best_response = ""

    for entry in pattern_data:
        similarity = util.pytorch_cos_sim(user_embedding, entry['embedding']).item()
        if similarity > best_score:
            best_score = similarity
            best_response = entry['response']

    if best_score < 0.65:
        return generate_fallback_response(user_input)

    return best_response

@app.route('/')
def home():
    return send_from_directory(app.static_folder, 'index1.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    if not user_input or user_input.strip() == '':
        return jsonify({'response': "Please say or type something."})
    response = chatbot_response(user_input)
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(debug=True)
