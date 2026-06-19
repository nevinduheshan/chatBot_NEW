import os
from dotenv import load_dotenv
from pinecone import Pinecone
import google.generativeai as genai

# Load environment variables from the .env file
load_dotenv()

# Accessing variables from .env
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME")

# Initialize Pinecone and Gemini clients
pc = Pinecone(api_key=PINECONE_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

def get_advanced_analysis(user_query):
    print("\n🔍 Step 1: Searching PDF documents via Pinecone Assistant...")
    
    # Send user query to Pinecone Assistant
    pinecone_response = pc.assistant.chat(
        assistant_name=ASSISTANT_NAME,
        messages=[{"role": "user", "content": user_query}]
    )
    
    # Retrieve the extracted raw data from the PDF
    raw_pdf_data = pinecone_response.message.content
    
    print("🧠 Step 2: Analyzing and formatting data via Google Gemini...")
    
    master_prompt = f"""
    You are an expert Financial Analyst.
    User Question: "{user_query}"
    
    Raw Data from PDF:
    {raw_pdf_data}
    
    Task: Analyze and format this data professionally using markdown.
    """
    
    model = genai.GenerativeModel("gemini-2.5-flash")
    analysis_response = model.generate_content(master_prompt)
    
    return analysis_response.text

# --- Interactive Chat Loop ---
if __name__ == "__main__":
    print("\n🤖 Financial AI Bot is ready!")
    print("👉 Type your question and press Enter.")
    print("👉 Type 'exit' or 'quit' to close the program.")
    print("==========================================================")
    
    while True:
        # Get real-time question from the user
        user_question = input("\n💬 Ask a question about your PDF: ")
        
        # Check if the user wants to exit
        if user_question.lower() in ['exit', 'quit']:
            print("👋 Exiting chat. Goodbye!")
            break
            
        # Skip if the user just pressed enter without typing anything
        if not user_question.strip():
            continue
            
        try:
            # Process the question and print the answer
            final_output = get_advanced_analysis(user_question)
            print("\n================ 📊 FINAL GEMINI ANALYSIS ================")
            print(final_output)
            print("==========================================================")
        except Exception as e:
            print(f"❌ An error occurred: {e}")