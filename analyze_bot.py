import os
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
import google.generativeai as genai

load_dotenv()

# ==================== CONFIGURATION ====================dddd
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME")
# =======================================================

# 1. Initialize Pinecone and Gemini clients
pc = Pinecone(api_key=PINECONE_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

def get_advanced_analysis(user_query):
    print("Step 1: Searching PDF documents via Pinecone Assistant...")
    
    # Send user query to Pinecone Assistant
    pinecone_response = pc.assistant.chat(
        assistant_name=ASSISTANT_NAME,
        messages=[{"role": "user", "content": user_query}]
    )
    
    # Retrieve the extracted raw data from the PDF
    raw_pdf_data = pinecone_response.message.content
    
    print("Step 2: Analyzing and formatting data via Google Gemini...")
    
    master_prompt = f"""
    You are an expert Financial Analyst.
    User Question: "{user_query}"
    
    Raw Data from PDF:
    {raw_pdf_data}
    
    Task: Analyze and format this data professionally.
    """
    
    model = genai.GenerativeModel("gemini-2.5-flash")
    analysis_response = model.generate_content(master_prompt)
    
    return analysis_response.text

if __name__ == "__main__":
    test_question = "What is the best profit year?"
    final_output = get_advanced_analysis(test_question)
    print("\n================ FINAL GEMINI ANALYSIS ================")
    print(final_output)