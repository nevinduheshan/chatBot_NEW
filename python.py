import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
# API key එක දාන්න
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ඔයාට පාවිච්චි කරන්න පුළුවන් මොඩල් ලැයිස්තුව බලන්න
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)