
# app.py
from flask import Flask, render_template, request, jsonify, session
import os
from langchain_huggingface import HuggingFaceEmbeddings


from langchain_pinecone import PineconeVectorStore
from groq import Groq
from dotenv import load_dotenv
import uuid

# Load environment variables from .env file
load_dotenv()

# Set API keys (use environment variables in production)
os.environ["PINECONE_API_KEY"] = "pcsk_5CsWGm_DTETbjaHK7ZP6P2eQaMNL2JdUTKitPSuGC3Ntx3nwJNjcWLGsjwopHmUrV58r5D"
os.environ["GROQ_API_KEY"] = "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"

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load existing Pinecone index
doc_store = PineconeVectorStore.from_existing_index(
    index_name="bible-help",
    embedding=embedding_model,
)

def get_scriptures(query, chat_history=[], num_scriptures=5):
    # Use similarity search to find relevant scriptures
    results = doc_store.similarity_search_with_score(query, k=num_scriptures)
    scriptures = [doc.page_content for doc, _ in results]
    
    history_text = "\n".join([f"User: {msg['content']}\nAssistant: {msg['response']}" 
                             for msg in chat_history[-6:]])
    
    prompt = f"""Previous conversation:
{history_text}
Bible Verses on the topic: {query}
Retrieved Scriptures:
{chr(10).join([f"{i+1}. {scripture}" for i, scripture in enumerate(scriptures)])}

User asked for verses about: {query}
Provide exactly {num_scriptures} Bible verses related to this topic. 
Format each scripture with its reference (book, chapter, verse) and the text.
Assistant:"""
    
    client = Groq(api_key="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")
    completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful Bible assistant. Provide relevant Bible verses on the requested topics. Always include the scripture reference (book, chapter, verse) with each verse."},
            {"role": "user", "content": prompt}
        ],
        model="gemma2-9b-it",
        max_tokens=1000
    )
    
    return completion.choices[0].message.content, scriptures

def explain_topic(query, chat_history=[]):
    # Retrieve more context for topic explanation
    results = doc_store.similarity_search_with_score(query, k=8)
    scriptures = [doc.page_content for doc, _ in results]
    
    history_text = "\n".join([f"User: {msg['content']}\nAssistant: {msg['response']}" 
                             for msg in chat_history[-6:]])
    
    prompt = f"""Previous conversation:
{history_text}
Biblical Context on the topic: {query}
Retrieved Scripture Context:
{chr(10).join([f"{i+1}. {scripture}" for i, scripture in enumerate(scriptures)])}

User asked to learn more about the biblical concept of: {query}
Based ONLY on the retrieved scripture context above, provide a detailed explanation of this biblical concept.
Include key aspects, how it's understood in biblical context, and how different passages relate to it.
Cite specific verses from the retrieved context to support your explanation.
Assistant:"""
    
    client = Groq(api_key=os.getenv("GROQ_API_KEY", "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"))
    completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a knowledgeable Bible teacher. Provide in-depth explanations of biblical concepts using only the scripture context provided. Cite verses to support your explanations."},
            {"role": "user", "content": prompt}
        ],
        model="gemma2-9b-it",
        max_tokens=1500
    )
    
    return completion.choices[0].message.content

# Parse the number of verses requested
def parse_scripture_count(query):
    # Default to 5 scriptures
    num_scriptures = 5
    
    # Check if the query explicitly asks for a specific number
    if "give me" in query.lower() and "scriptures" in query.lower():
        parts = query.lower().split("give me")
        if len(parts) > 1:
            for word in parts[1].split():
                if word.isdigit():
                    num_scriptures = int(word)
                    break
    
    return num_scriptures

# Extract topic from query
def extract_topic(query):
    topic = query
    if "scriptures on" in query.lower():
        topic = query.lower().split("scriptures on")[1].strip()
    elif "verses on" in query.lower():
        topic = query.lower().split("verses on")[1].strip()
    elif "verses about" in query.lower():
        topic = query.lower().split("verses about")[1].strip()
    elif "about" in query.lower():
        topic = query.lower().split("about")[1].strip()
    return topic

# Routes
@app.route('/')
def index():
    # Initialize chat history in session if it doesn't exist
    if 'chat_history' not in session:
        session['chat_history'] = []
    if 'current_topic' not in session:
        session['current_topic'] = ""
    
    return render_template('index.html', chat_history=session['chat_history'])

@app.route('/query', methods=['POST'])
def process_query():
    user_input = request.form['query']
    
    # Generate unique ID for each message
    message_id = str(uuid.uuid4())
    
    # Determine how many scriptures to return
    num_scriptures = parse_scripture_count(user_input)
    
    # Extract the topic from the query
    topic = extract_topic(user_input)
    
    # Store current topic for buttons
    session['current_topic'] = topic
    
    # Get AI response
    response, _ = get_scriptures(topic, session.get('chat_history', []), num_scriptures)
    
    # Add to chat history
    if 'chat_history' not in session:
        session['chat_history'] = []
    
    session['chat_history'].append({
        'id': message_id,
        'content': user_input,
        'response': response,
        'type': 'scriptures'
    })
    
    # Save session
    session.modified = True
    
    return jsonify({
        'id': message_id,
        'response': response,
        'topic': topic
    })

@app.route('/more-verses', methods=['POST'])
def more_verses():
    topic = session.get('current_topic', "")
    
    # Generate unique ID for this response
    message_id = str(uuid.uuid4())
    
    # Get more verses
    response, _ = get_scriptures(topic, session.get('chat_history', []), 5)
    
    # Add to chat history
    session['chat_history'].append({
        'id': message_id,
        'content': f"Show more verses about {topic}",
        'response': response,
        'type': 'more_scriptures'
    })
    
    # Save session
    session.modified = True
    
    return jsonify({
        'id': message_id,
        'response': response
    })

@app.route('/explain-topic', methods=['POST'])
def explain():
    topic = session.get('current_topic', "")
    
    # Generate unique ID for this response
    message_id = str(uuid.uuid4())
    
    # Get explanation
    response = explain_topic(topic, session.get('chat_history', []))
    
    # Add to chat history
    session['chat_history'].append({
        'id': message_id,
        'content': f"Tell me more about {topic}",
        'response': response,
        'type': 'explanation'
    })
    
    # Save session
    session.modified = True
    
    return jsonify({
        'id': message_id, 
        'response': response
    })

@app.route('/clear-history', methods=['POST'])
def clear_history():
    session['chat_history'] = []
    session['current_topic'] = ""
    session.modified = True
    
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True)