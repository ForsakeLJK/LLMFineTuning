import os

import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

MAX_TOKENS = 256  # generation length

# Load the model from Hugging Face Hub
llm = Llama.from_pretrained(
    repo_id="JukeKth/ID2223-Lab2-Model",
    filename="model-1B-fine-tuned.gguf",
)

def build_prompt(message, history):
    lines = []

    for turn in (history or []):
        role = turn.get("role", None)
        content = turn.get("content", None)

        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            else:
                text_parts.append(str(part))
        content = "\n".join(text_parts)

        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")

    lines.append(f"User: {message}")
    lines.append("Assistant:")

    return "\n".join(lines)


def chat_fn(message, history):
    prompt = build_prompt(message, history)

    out = llm(
        prompt,
        max_tokens=MAX_TOKENS,
        temperature=0.7,
        top_p=0.9,
        stop=["User:", "Assistant:"],
    )
    text = out["choices"][0]["text"]
    return text.strip()


with gr.Blocks() as demo:
    gr.Markdown("# ID2223 Lab2 ChatBot")
    gr.ChatInterface(fn=chat_fn)

demo.launch()
