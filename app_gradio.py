"""
Gradio chat interface for the SETU Compliance RAG chatbot.
Run via the Colab notebook or locally:
    python app_gradio.py
"""
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from main_agent import handle_query


def chat(message: str, history: list) -> str:
    if not message.strip():
        return "Please enter a question about SETU policies."
    return handle_query(message)


demo = gr.ChatInterface(
    fn=chat,
    title="SETU Compliance Policy Assistant",
    description=(
        "Ask questions about SETU's 48 institutional policies — "
        "leave entitlements, recruitment, EDI, research conduct, data protection, and more."
    ),
    examples=[
        "How many weeks of maternity leave is a female staff member entitled to?",
        "What is SETU's data protection policy?",
        "Is Garda vetting required for all staff roles?",
        "What constitutes research misconduct at SETU?",
        "What support does SETU provide to staff with caring responsibilities?",
        "What does the Gen AI policy say about staff using AI tools?",
    ],
    theme=gr.themes.Soft(),
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch(share=True)
