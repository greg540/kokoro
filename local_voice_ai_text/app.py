import os
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import requests
import soundfile as sf
from kokoro import KPipeline

# Defaults can be overridden with environment variables before launch.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1").rstrip("/")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:14b")
DEFAULT_VOICE = os.getenv("KOKORO_VOICE", "af_heart")
KOKORO_LANG = os.getenv("KOKORO_LANG", "a")  # a = American English
SAMPLE_RATE = 24000

_kokoro_pipeline = None


def get_kokoro_pipeline():
    """Initialize Kokoro on the first request, not while the UI is starting."""
    global _kokoro_pipeline
    if _kokoro_pipeline is None:
        _kokoro_pipeline = KPipeline(lang_code=KOKORO_LANG)
    return _kokoro_pipeline


def call_ollama(user_text, history, system_prompt, model_name):
    model_name = (model_name or DEFAULT_MODEL).strip()
    if not model_name:
        raise gr.Error("Enter a model name shown by `ollama list`.")

    messages = [{
        "role": "system",
        "content": (system_prompt or "You are a helpful voice assistant. Keep answers concise and natural when spoken aloud.").strip(),
    }]

    # Keep up to the last 10 conversation messages to preserve context without
    # sending the entire conversation indefinitely.
    for item in (history or [])[-10:]:
        if isinstance(item, dict) and item.get("role") in ("user", "assistant"):
            content = item.get("content", "")
            if isinstance(content, str) and content.strip():
                messages.append({"role": item["role"], "content": content})

    messages.append({"role": "user", "content": user_text})
    url = f"{OLLAMA_BASE_URL}/chat/completions"
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.7,
        "stream": False,
    }
    try:
        response = requests.post(url, json=payload, timeout=(10, 300))
        response.raise_for_status()
        data = response.json()
        reply = data["choices"][0]["message"]["content"].strip()
        if not reply:
            raise RuntimeError("Ollama returned an empty response.")
        return reply, model_name
    except requests.exceptions.ConnectionError as exc:
        raise gr.Error(
            f"Can't connect to Ollama at {OLLAMA_BASE_URL}. Start Ollama and try again."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise gr.Error("Ollama took too long to respond. Check that the model is running.") from exc
    except requests.exceptions.HTTPError as exc:
        body = response.text[:1000] if "response" in locals() else str(exc)
        raise gr.Error(f"Ollama API returned an error ({exc}).\n{body}") from exc
    except (KeyError, ValueError, TypeError) as exc:
        raise gr.Error(f"Couldn't read Ollama's response: {exc}") from exc


def make_speech(text, voice):
    """Generate a WAV file using the Kokoro KPipeline API."""
    voice = (voice or DEFAULT_VOICE).strip()
    pipeline = get_kokoro_pipeline()
    chunks = []
    try:
        for _graphemes, _phonemes, audio in pipeline(text, voice=voice, speed=1.0):
            if audio is None:
                continue
            # Kokoro normally yields torch tensors; explicitly move to CPU
            # before converting so this also works if its pipeline uses CUDA.
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            chunks.append(np.asarray(audio, dtype=np.float32).reshape(-1))
    except Exception as exc:
        raise gr.Error(
            f"Kokoro speech generation failed: {exc}\n"
            "Check that espeak-ng is installed and that the selected voice matches the language."
        ) from exc

    if not chunks:
        raise gr.Error("Kokoro did not return any audio. Try a shorter response or another voice.")

    waveform = np.concatenate(chunks)
    temp_dir = Path(tempfile.gettempdir()) / "local_voice_ai_text"
    temp_dir.mkdir(parents=True, exist_ok=True)
    fd, output_path = tempfile.mkstemp(prefix="kokoro_reply_", suffix=".wav", dir=temp_dir)
    os.close(fd)
    sf.write(output_path, waveform, SAMPLE_RATE)
    return output_path


def send_message(user_text, history, system_prompt, model_name, voice):
    user_text = (user_text or "").strip()
    history = history or []
    if not user_text:
        return "", history, None, "Enter a message first."

    reply, actual_model = call_ollama(user_text, history, system_prompt, model_name)
    updated_history = history + [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": reply},
    ]
    audio_path = make_speech(reply, voice)
    status = f"Reply generated locally with `{actual_model}` and spoken using Kokoro voice `{voice}`."
    return "", updated_history, audio_path, status


def clear_conversation():
    return "", [], None, "Conversation cleared."


def test_ollama(model_name):
    model_name = (model_name or DEFAULT_MODEL).strip()
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/models", timeout=5)
        response.raise_for_status()
        data = response.json()
        models = [item.get("id", "") for item in data.get("data", [])]
        if models and model_name not in models:
            return f"Ollama is reachable, but `{model_name}` wasn't listed. Available models: " + ", ".join(models)
        return f"Ollama is reachable at `{OLLAMA_BASE_URL}`. Selected model: `{model_name}`."
    except Exception as exc:
        return f"Couldn't reach Ollama at `{OLLAMA_BASE_URL}`: {exc}"


with gr.Blocks(title="Local Text + Voice AI Test", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Local AI Text + Voice Test\n"
        "Type a message, get a response from your local Ollama model, then hear the response spoken by Kokoro.\n\n"
        "**No microphone or Vonage connection is used in this version.**"
    )

    with gr.Row():
        model_box = gr.Textbox(label="Ollama model", value=DEFAULT_MODEL, scale=2)
        voice_box = gr.Dropdown(
            label="Kokoro voice",
            choices=["af_heart", "af_bella", "af_nicole", "af_sarah", "am_adam", "am_michael"],
            value=DEFAULT_VOICE,
            allow_custom_value=True,
            scale=1,
        )

    with gr.Row():
        check_button = gr.Button("Test Ollama connection")
        connection_status = gr.Markdown("Ollama endpoint: `" + OLLAMA_BASE_URL + "`")

    system_box = gr.Textbox(
        label="System instructions",
        value="You are a helpful voice assistant. Keep answers concise, conversational, and easy to understand when spoken aloud.",
        lines=2,
    )
    chat = gr.Chatbot(label="Conversation", type="messages", height=380)
    with gr.Row():
        message_box = gr.Textbox(
            label="Your message",
            placeholder="Type a message and press Enter…",
            lines=2,
            scale=5,
        )
        send_button = gr.Button("Send", variant="primary", scale=1)

    audio_output = gr.Audio(label="Kokoro spoken reply", type="filepath")
    status = gr.Markdown("Ready. Enter a message to start testing.")
    clear_button = gr.Button("Clear conversation")

    inputs = [message_box, chat, system_box, model_box, voice_box]
    outputs = [message_box, chat, audio_output, status]
    send_button.click(send_message, inputs=inputs, outputs=outputs)
    message_box.submit(send_message, inputs=inputs, outputs=outputs)
    clear_button.click(clear_conversation, inputs=[], outputs=outputs)
    check_button.click(test_ollama, inputs=[model_box], outputs=[connection_status])

if __name__ == "__main__":
    # Bind only to the local machine; this is not exposed to your network.
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, share=False)
