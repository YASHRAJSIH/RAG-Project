from flask import Flask, render_template, request, jsonify
import openai
import faiss
import pandas as pd
import numpy as np
import os
import pickle
from sentence_transformers import SentenceTransformer
import atexit

# Global variables
app = Flask(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"  # Limit threads to prevent semaphore issues

# Cleanup function to ensure resources are released
@atexit.register
def cleanup_resources():
    print("Cleaning up resources...")
    # Add any specific cleanup logic here if needed

def readcsv():
    data_path = './Cleaned_NYC_Jobs_Dataset.csv'
    df = pd.read_csv(data_path)
    return df

def compute_or_load_embeddings():
    embeddings_file = './embeddings.pkl'
    embedding_model = SentenceTransformer('all-mpnet-base-v2')
    df = readcsv()
    if os.path.exists(embeddings_file):
        with open(embeddings_file, 'rb') as f:
            embeddings = pickle.load(f)
            print("Embeddings loaded from file.")
    else:
        embeddings = embedding_model.encode(df['Job Description'].tolist(), show_progress_bar=True, batch_size=32)
        with open(embeddings_file, 'wb') as f:
            pickle.dump(embeddings, f)
            print("Embeddings computed and saved.")
    return embeddings

# Function to load or initialize FAISS index
def initialize_index():
    faiss_index_file = './faiss_index.index'
    if os.path.exists(faiss_index_file):
        # Load existing FAISS index
        index = faiss.read_index(faiss_index_file)
        print(f"FAISS index loaded with size: {index.ntotal}")
    else:
        # Create and populate FAISS index
        embeddings = compute_or_load_embeddings()
        embeddings = np.array(embeddings, dtype='float32')
        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(embeddings)
        print(f"FAISS index initialized with size: {index.ntotal}")
        faiss.write_index(index, faiss_index_file)
    return index

def get_relevant_jobs(user_query):
    # Initialize FAISS index
    embedding_model = SentenceTransformer('all-mpnet-base-v2')
    df = readcsv()
    index = initialize_index()
    try:
        query_embedding = embedding_model.encode([user_query]).astype('float32')
        k = 5
        distances, indices = index.search(query_embedding, k)

        if indices[0].size == 0:
            print("No matches found in the dataset.")
            return pd.DataFrame()

        relevant_jobs = df.iloc[indices[0]].reset_index(drop=True)
        
        return relevant_jobs
    except Exception as e:
        print(f"Error in get_relevant_jobs: {e}")
        return pd.DataFrame()

def generate_response(user_query, relevant_jobs, context):
    df = readcsv()
    openai.api_key = "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"
    try:
        if relevant_jobs.empty:
            return "I'm sorry, I couldn't find any matching job opportunities based on your query. Could you provide more details or try a different query?"

        prompt = "You are a helpful job advisor for New York City. Use the following job details and context to answer the user's query:\n\n"
        for _, row in relevant_jobs.iterrows():
            prompt += "---\n"
            for col in df.columns:
                prompt += f"{col}: {row[col]}\n"
        prompt += f"\nContext: {context}\nUser query: {user_query}\nResponse:"

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful job advisor."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error in generate_response: {e}")
        return "Sorry, there was an issue generating a response."

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/chat', methods=['POST'])
def chat():
    conversation_history = []
    current_context = {}

    if not request.is_json:
        return jsonify({"error": "Request must be JSON."}), 400

    user_query = request.json.get('query')
    if not user_query:
        return jsonify({"error": "Query field is required."}), 400

    conversation_history.append(f"You: {user_query}")
    relevant_jobs = get_relevant_jobs(user_query)

    if relevant_jobs.empty:
        response = "I'm sorry, I couldn't find any matching job opportunities. Please try another query."
    else:
        context = "\n".join(conversation_history) + "\n" + str(current_context)
        response = generate_response(user_query, relevant_jobs, context)

    conversation_history.append(f"AI: {response}")
    return jsonify({"history": "\n".join(conversation_history)})

if __name__ == "__main__":
    app.run(debug=True, port=5001)
