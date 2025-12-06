# ID2223 Lab 2

This repository contains the work for the ID2223 Lab 2 assignment by group **rice_cake**.  
Our goal for this lab is to achieve **grade A**.

## Summary of Work

We followed and extended the Unsloth example notebook, running all experiments on Google Colab.

### Model Training (SFT)
- Fine-tuned **`unsloth/Llama-3.2-1B-Instruct`**.  
  (We attempted the 3B model, but both training and inference were too slow on free Colab GPU/CPU.)
- Implemented:
  - Checkpoint saving to Google Drive  
  - Conversion to **GGUF** for use with `llama.cpp`

### Preference Alignment (ORPO)
- Applied **preference alignment** using the **ORPO Trainer** from `trl`.
- Used the **orpo-dpo-mix-40k** dataset (same author as FineTome-100k).

### Chatbot & Personas
- Built a Gradio chatbot with several personas:
  - *Helpful*
  - *Doomer*
  - *Sassy*
  - *Pirate*
- While resource constraints prevented detailed evaluation, qualitative testing in the chatbot shows improved behavior after training.

## Links

**Hugging Face Space (Live Demo):**  
https://huggingface.co/spaces/JukeKth/ID2223Lab2

## Repository Structure
```plain text
app.py                # Gradio application entry point
requirements.txt      # Dependencies for the Gradio app
ID2223Lab2.ipynb      # Training and alignment notebook (Colab)
```
