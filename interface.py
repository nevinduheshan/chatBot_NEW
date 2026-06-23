import streamlit as st
import os
from dotenv import load_dotenv
from pinecone import Pinecone
import google.generativeai as genai
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME")

# Initialize Pinecone and Gemini clients
pc = Pinecone(api_key=PINECONE_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Financial AI Assistant",
    page_icon="📊",
    layout="centered"
)

# Custom Styling for a premium look
st.markdown("""
    <style>
    .main-title {
        font-size: 40px;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-title {
        font-size: 18px;
        color: #4B5563;
        text-align: center;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📊 Financial AI Analyst</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Ask any question about your uploaded PDF reports instantly</div>', unsafe_allow_html=True)

# --- Sidebar Configuration ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.title("System Status")
    st.success("⚡ Pinecone: Connected")
    st.success("⚡ Gemini: Connected")
    st.info(f"🤖 Assistant: {ASSISTANT_NAME}")
    st.caption("Powered by SmartyAI")

# --- Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your AI Financial Analyst. How can I help you analyze your PDF documents today?"}
    ]

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Core RAG Function with History Memory ---
def process_ai_response(chat_history):
    # Step 1: Send full chat history to Pinecone so it understands the context of follow-up questions
    pinecone_response = pc.assistant.chat(
        assistant_name=ASSISTANT_NAME,
        messages=chat_history
    )
    raw_pdf_data = pinecone_response.message.content
    
    # Format the past conversation history into a clear text format for Gemini
    conversation_context = ""
    for msg in chat_history[:-1]:  # Exclude the very last query which is processed separately below
        author = "User" if msg["role"] == "user" else "Analyst"
        conversation_context += f"{author}: {msg['content']}\n"
        
    user_query = chat_history[-1]["content"]
    
    # Step 2: Format and analyze with Gemini using past history + new raw data
    master_prompt = f"""
    You are an expert Financial Analyst. Address the user's latest question by maintaining the continuity of the conversation history and utilizing the newly retrieved raw PDF context.
    
    [Conversation History]
    {conversation_context}
    
    [Retrieved Context from PDF]
    {raw_pdf_data}
    
    [Latest User Question]
    "{user_query}"
    
    Task: Provide a fluid, natural, and context-aware response to the latest question. Format professionally using clear markdown headings, bold text, and bullet points.
    """
    model = genai.GenerativeModel("gemini-2.5-flash")
    analysis_response = model.generate_content(master_prompt)
    
    return analysis_response.text

# --- Handle User Input ---
if user_input := st.chat_input("Type your question here..."):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Log the incoming question to server console
    print(f"\n[INFO] {current_time} - 📥 USER QUESTION: {user_input}")
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Add user message to chat history state FIRST before sending to RAG function
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Generate AI response with a loading spinner
    with st.chat_message("assistant"):
        with st.spinner("Analyzing data with context... Please wait..."):
            try:
                # Pass the entire conversation state history to the processor
                ai_response = process_ai_response(st.session_state.messages)
                st.markdown(ai_response)
                
                print(f"[INFO] {datetime.now().strftime('%H:%M:%S')} - 📤 AI RESPONSE: Generated Contextually.")
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
            except Exception as e:
                error_msg = f"❌ An error occurred: {e}"
                print(f"[ERROR] {datetime.now().strftime('%H:%M:%S')} - ❌ CRASH: {e}")
                st.error(error_msg)