import google.generativeai as genai

# API key එක දාන්න
genai.configure(api_key="AQ.Ab8RN6IiQc3pv9JVOUL09F6vpNepxfOM0xAE0zeB-fTkC-lk0w")

# ඔයාට පාවිච්චි කරන්න පුළුවන් මොඩල් ලැයිස්තුව බලන්න
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)