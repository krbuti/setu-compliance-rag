"""
Gradio chat interface for the SETU Compliance RAG chatbot.
Run via the Colab notebook or locally:
    python app_gradio.py
"""
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from main_agent import handle_query

DISCLAIMER = (
    "⚠️ **Academic prototype — not an official SETU service.**  \n"
    "Answers are generated from 48 SETU policy documents using AI and may be "
    "incomplete or inaccurate. Always verify with the official policy document "
    "or contact your HR / relevant SETU office before making decisions."
)


def chat(message: str, history: list) -> str:
    if not message.strip():
        return "Please enter a question about SETU policies."
    answer = handle_query(message)
    return answer + "\n\n---\n*Disclaimer: This is an academic AI prototype. Verify all information with official SETU policy documents or HR before acting on it.*"


demo = gr.ChatInterface(
    fn=chat,
    title="SETU Compliance Policy Assistant",
    description=(
        "Ask questions about SETU's 48 institutional policies — "
        "leave entitlements, recruitment, EDI, research conduct, data protection, and more.\n\n"
        + DISCLAIMER
    ),
    examples=[
        "How many weeks of maternity leave is a female staff member entitled to?",
        "What is SETU's data protection policy?",
        "Is Garda vetting required for all staff roles?",
        "What constitutes research misconduct at SETU?",
        "What support does SETU provide to staff with caring responsibilities?",
        "What does the Gen AI policy say about staff using AI tools?",
    ],
)

if __name__ == "__main__":
    from information_agent import load_vectorstore
    load_vectorstore()  # warm BM25 + cross-encoder before first user query
    demo.launch(server_name="0.0.0.0", server_port=7860)
