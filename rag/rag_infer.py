# rag/rag_infer.py
import google.generativeai as genai

# Setup Gemini
genai.configure(api_key="")

model = genai.GenerativeModel("gemini-1.5-flash")

def ask_gemini(prompt: str) -> str:
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# Example usage
if __name__ == "__main__":
    question = input("Enter your question: ")
    answer = ask_gemini(question)
    print("\nGemini Response:\n", answer)
