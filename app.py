import gradio as gr
from llama_cpp import Llama

MAX_TOKENS = 512
REPO_ID = "JukeKth/ID2223-Lab2-Model"
FILENAME = "model-1B-fine-tuned.gguf"
# FILENAME = "model-1B-fine-tuned-12051614.gguf"

llm = Llama.from_pretrained(
    repo_id=REPO_ID,
    filename=FILENAME,
    n_ctx=2048,
    verbose=False
)

def get_system_message(personality):
    prompts = {
        "Helpful": (
            "You are a friendly and helpful AI assistant. "
            "Your answers are clear, concise, and polite."
        ),
        "Doomer": (
            "You are a pessimistic AI. You always point out the risks, downsides, and futility of any situation. "
            "Use a gloomy tone, sigh often (*sigh*), but strictly avoid encouraging harm."
        ),
        "Sassy": (
            "You are a sassy AI with an attitude. You answer the question, but you make a sarcastic comment or a joke about it first. "
            "Use emojis like 🙄, 💅, or 😒."
        ),
        "Pirate": (
            "You are a pirate captain. Speak in heavy pirate slang (Ahoy, Matey, Yarr). "
            "Always reference the sea, treasure, or your ship."
        )
    }
    return prompts.get(personality, prompts["Helpful"])


def _extract_text(content):
    if content is None:
        return ""
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            else:
                text_parts.append(str(part))
        return "\n".join(text_parts)
    return str(content)


def build_prompt(message, history, system_message):
    lines = [f"System: {system_message}", ""]

    if history:
        first = history[0]

        for turn in history:
            role = turn.get("role")
            content = _extract_text(turn.get("content"))
            if role == "user":
                lines.append(f"User: {content}")
            elif role == "assistant":
                lines.append(f"Assistant: {content}")
    
    lines.append(f"User: {message}")
    lines.append("Assistant:")
    return "\n".join(lines)


def chat_fn(message, history, personality, temperature, top_p):
    system_msg = get_system_message(personality)
    prompt = build_prompt(message, history, system_msg)

    out = llm(
        prompt,
        max_tokens=MAX_TOKENS,
        temperature=temperature,
        top_p=top_p,
        stop=["User:", "Assistant:", "System:"],
    )

    text = out["choices"][0]["text"]
    return text.strip()


custom_css = """
.gradio-container {
    background: radial-gradient(at 0% 0%, hsla(253,16%,7%,1) 0, transparent 50%), 
                radial-gradient(at 50% 0%, hsla(225,39%,30%,1) 0, transparent 50%), 
                radial-gradient(at 100% 0%, hsla(339,49%,30%,1) 0, transparent 50%);
    background-color: #111827;
}

#sidebar_panel {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    padding: 20px;
}

#chatbot_box {
    height: 600px !important;
    border-radius: 20px;
    background: rgba(17, 24, 39, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.1);
}
.message.user {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border-radius: 15px 15px 0 15px !important;
    color: white !important;
}

.message.bot {
    background: #374151 !important;
    border-radius: 15px 15px 15px 0 !important;
    border: 1px solid #4b5563;
}

h1, h2, h3, span, p {
    color: #e5e7eb !important;
}
"""

with gr.Blocks() as demo:

    gr.Markdown(
        """
        # ✨ ID2223 Lab 2 ChatBot
        ### Chat with our fine-tuned 1B model in style.
        """
    )

    with gr.Row():
        with gr.Column(scale=1, elem_id="sidebar_panel"):
            gr.Markdown("### ⚙️ Settings")

            personality = gr.Dropdown(
                choices=["Helpful", "Doomer", "Sassy", "Pirate"],
                value="Helpful",
                label="Choose Persona",
                interactive=True
            )

            with gr.Accordion("🧠 Model Parameters", open=False):
                temp_slider = gr.Slider(
                    minimum=0.1, maximum=1.5, value=0.7, step=0.1,
                    label="Temperature (Creativity)"
                )
                top_p_slider = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.9, step=0.05,
                    label="Top-P (Focus)"
                )

            gr.Markdown(
                "*Tip: Higher temperature makes the model more creative but potentially less coherent.*"
            )

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                elem_id="chatbot_box",
                show_label=False,
                avatar_images=(
                    None,
                    "https://cdn-icons-png.flaticon.com/512/4712/4712027.png"
                ),
            )

            gr.ChatInterface(
                fn=chat_fn,
                chatbot=chatbot,
                additional_inputs=[personality, temp_slider, top_p_slider],
                examples=[
                    ["Hello! Who are you?", "Helpful", 0.7, 0.9],
                    ["What is the meaning of life?", "Doomer", 0.8, 0.9],
                    ["Tell me a joke.", "Sassy", 0.9, 0.9],
                    ["Write a poem about the ocean.", "Pirate", 0.7, 0.9]
                ],
                submit_btn="Send"
            )

demo.launch(theme=gr.themes.Soft(), css=custom_css)
