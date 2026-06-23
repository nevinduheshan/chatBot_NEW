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

st.markdown('<div class="main-title">📊 Predictive Financial AI Analyst</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Analyze past data and forecast future financial trends instantly</div>', unsafe_allow_html=True)

# --- Sidebar Configuration ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.title("System Status")
    st.success("⚡ Pinecone: Connected")
    st.success("⚡ Gemini: Connected")
    st.info(f"🤖 Assistant: {ASSISTANT_NAME}")
    st.caption("Powered by SmartyAI Predictive Engine")

# --- Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your Advanced Financial Analyst. I can now analyze your historical PDF data, calculate trends, and provide financial forecasts for upcoming years (e.g., 2026, 2027). How can I assist you today?"}
    ]

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Core RAG Function with Forecasting Prompt ---
def process_ai_response(chat_history):
    # Step 1: Fetch relevant historical & forward-looking statements from Pinecone
    pinecone_response = pc.assistant.chat(
        assistant_name=ASSISTANT_NAME,
        messages=chat_history
    )
    raw_pdf_data = pinecone_response.message.content
    
    # Format the past conversation history
    conversation_context = ""
    for msg in chat_history[:-1]:
        author = "User" if msg["role"] == "user" else "Analyst"
        conversation_context += f"{author}: {msg['content']}\n"
        
    user_query = chat_history[-1]["content"]
    
    # Step 2: Advanced Predictive Prompting for Gemini
    master_prompt = f"""
    You are an Elite Financial Analyst and Predictive Forecasting Expert. 
    Your job is to address the user's latest question using the conversation history and the raw PDF context.

    CRITICAL CAPABILITY (FINANCIAL FORECASTING):
    If the user asks for predictions, forecasts, or future years (such as 2026, 2027, etc.):
    1. Look at the historical growth rates (Revenue, Net Profit, Asset growth) available in the [Retrieved Context from PDF].
    2. Check for any "Management Outlook", "Future Strategy", or "Macroeconomic Risks" sections inside the raw context.
    3. Calculate or estimate reasonable quantitative/qualitative projections based on those trends. 
    4. Provide a structured "Future Projections & Forecast" section with realistic metrics and explain the logic behind your calculations (e.g., "Assuming a stable CAGR of 12% based on 2023-2024 performance...").
    5. Always include a brief bullet point on "Key Assumptions & Risk Factors" that could impact this prediction (e.g., inflation, regulatory changes).

    [Conversation History]
    {conversation_context}
    
    [Retrieved Context from PDF]
    {raw_pdf_data}
    
    [Latest User Question]
    "{user_query}"
    
    Task: Generate a brilliant, professional, and context-aware answer. If future prediction is required, deliver it with a proper financial disclaimer at the end stating that it's an AI-generated projection based on historical data. Use clear markdown formatting, tables, bold text, and bullet points.
    """
    model = genai.GenerativeModel("gemini-2.5-flash")
    analysis_response = model.generate_content(master_prompt)
    
    return analysis_response.text

# --- Handle User Input ---
if user_input := st.chat_input("Type your question or forecasting request here..."):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[INFO] {current_time} - 📥 USER QUESTION: {user_input}")
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("assistant"):
        with st.spinner("Processing deep analysis and predictive modeling... Please wait..."):
            try:
                ai_response = process_ai_response(st.session_state.messages)
                st.markdown(ai_response)
                
                print(f"[INFO] {datetime.now().strftime('%H:%M:%S')} - 📤 AI RESPONSE: Generated Predictively.")
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
            except Exception as e:
                error_msg = f"❌ An error occurred: {e}"
                print(f"[ERROR] {datetime.now().strftime('%H:%M:%S')} - ❌ CRASH: {e}")
                st.error(error_msg)