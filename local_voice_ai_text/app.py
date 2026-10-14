import json
import os
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import requests
import soundfile as sf
from kokoro import KPipeline

# Defaults can be overridden with environment variables before launch.
# Ollama's native API is used for streaming responses and think=False.
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL", "http://localhost:11434/v1"
).rstrip("/")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:0.8b")
DEFAULT_VOICE = os.getenv("KOKORO_VOICE", "af_heart")
KOKORO_LANG = os.getenv("KOKORO_LANG", "a")  # a = American English
SAMPLE_RATE = 24000

_kokoro_pipeline = None


def ollama_native_base_url():
    """Convert http://localhost:11434/v1 to http://localhost:11434."""
    base_url = OLLAMA_BASE_URL.rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    return base_url


def get_kokoro_pipeline():
    """Initialize Kokoro on the first speech request."""
    global _kokoro_pipeline
    if _kokoro_pipeline is None:
        _kokoro_pipeline = KPipeline(lang_code=KOKORO_LANG)
    return _kokoro_pipeline


def build_messages(user_text, history, system_prompt):
    """Build a short request using only a few recent conversation messages."""
    prompt = (system_prompt or "").strip()
    if not prompt:
        prompt = (
            "You are a friendly, natural voice assistant. "
            "Reply in one or two short sentences, usually under 25 words. "
            "Do not show reasoning, markdown, or bullet points."
        )

    messages = [{"role": "system", "content": prompt}]

    # Chat history is stored as Gradio messages. Keep the last 6 messages
    # (usually 3 user/assistant turns) to reduce prompt-processing overhead.
    for item in (history or [])[-6:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content", "")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_text})
    return messages


def stream_ollama_reply(user_text, history, system_prompt, model_name):
    """Yield the complete response so far as Ollama sends streamed chunks."""
    model_name = (model_name or DEFAULT_MODEL).strip()
    if not model_name:
        raise RuntimeError("Enter a model name shown by `ollama list`.")

    url = f"{ollama_native_base_url()}/api/chat"
    payload = {
        "model": model_name,
        "messages": build_messages(user_text, history, system_prompt),
        "think": False,
        "stream": True,
        "keep_alive": "30m",
        "options": {
            "temperature": 0.5,
            "num_predict": 60,
            "num_ctx": 2048,
        },
    }

    accumulated = ""
    try:
        with requests.post(
            url,
            json=payload,
            stream=True,
            timeout=(10, 120),
        ) as response:
            if not response.ok:
                error_body = response.text[:1000]
                raise RuntimeError(
                    f"Ollama API returned HTTP {response.status_code}: {error_body}"
                )

            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue

                if chunk.get("error"):
                    raise RuntimeError(chunk["error"])

                message = chunk.get("message", {})
                content = message.get("content", "") if isinstance(message, dict) else ""
                if content:
                    accumulated += content
                    yield accumulated

                if chunk.get("done"):
                    break

    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Can't connect to Ollama at {ollama_native_base_url()}. "
            "Make sure Ollama is running."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            "Ollama timed out. Check that the model is loaded and responding."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc

    if not accumulated.strip():
        raise RuntimeError(
            "Ollama returned no visible text. Check that the selected model is installed "
            "and supports the requested API options."
        )


def make_speech(text, voice):
    """Generate a WAV file using the Kokoro KPipeline API."""
    voice = (voice or DEFAULT_VOICE).strip()
    pipeline = get_kokoro_pipeline()
    chunks = []

    try:
        for _graphemes, _phonemes, audio in pipeline(text, voice=voice, speed=1.0):
            if audio is None:
                continue
            # Kokoro normally yields torch tensors; move to CPU before conversion.
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            chunks.append(np.asarray(audio, dtype=np.float32).reshape(-1))
    except Exception as exc:
        raise RuntimeError(
            f"Kokoro speech generation failed: {exc}. Check eSpeak NG and the selected voice."
        ) from exc

    if not chunks:
        raise RuntimeError(
            "Kokoro did not return any audio. Try a shorter response or another voice."
        )

    waveform = np.concatenate(chunks)
    temp_dir = Path(tempfile.gettempdir()) / "local_voice_ai_text"
    temp_dir.mkdir(parents=True, exist_ok=True)
    fd, output_path = tempfile.mkstemp(
        prefix="kokoro_reply_", suffix=".wav", dir=temp_dir
    )
    os.close(fd)
    sf.write(output_path, waveform, SAMPLE_RATE)
    return output_path


def send_message(user_text, history, system_prompt, model_name, voice):
    """Stream the text into the chat, then generate and attach Kokoro audio."""
    user_text = (user_text or "").strip()
    history = list(history or [])
    if not user_text:
        yield "", history, None, "Enter a message first."
        return

    actual_model = (model_name or DEFAULT_MODEL).strip()
    updated_history = history + [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": ""},
    ]

    # Show the user's message immediately while Ollama starts responding.
    yield "", updated_history, None, f"Waiting for `{actual_model}` to respond..."

    reply = ""
    try:
        for partial_reply in stream_ollama_reply(
            user_text, history, system_prompt, actual_model
        ):
            reply = partial_reply
            updated_history[-1] = {"role": "assistant", "content": reply}
            yield "", updated_history, None, f"Receiving response from `{actual_model}`..."
    except Exception as exc:
        error_text = str(exc)
        updated_history[-1] = {
            "role": "assistant",
            "content": f"Ollama error: {error_text}",
        }
        yield "", updated_history, None, f"**Ollama error:** {error_text}"
        return

    # Important: display the complete text before generating the audio.
    yield (
        "",
        updated_history,
        None,
        f"Text ready from `{actual_model}`. Generating speech with Kokoro...",
    )

    try:
        audio_path = make_speech(reply, voice)
        yield (
            "",
            updated_history,
            audio_path,
            f"Reply generated with `{actual_model}` and spoken using Kokoro voice `{voice}`.",
        )
    except Exception as exc:
        yield (
            "",
            updated_history,
            None,
            f"Text reply is ready, but Kokoro could not generate audio: {exc}",
        )


def clear_conversation():
    return "", [], None, "Conversation cleared."


def test_ollama(model_name):
    model_name = (model_name or DEFAULT_MODEL).strip()
    url = f"{ollama_native_base_url()}/api/tags"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        models = [item.get("name", "") for item in data.get("models", [])]
        if models and model_name not in models:
            return (
                f"Ollama is reachable, but `{model_name}` wasn't listed. Available models: `"
                + "`, `".join(models)
                + "`"
            )
        if not models:
            return f"Ollama is reachable at `{url}`, but it returned no installed models."
        return f"Ollama is reachable. Selected model: `{model_name}`."
    except Exception as exc:
        return f"Couldn't reach Ollama at `{url}`: {exc}"


with gr.Blocks(title="Local Text + Voice AI Test", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Local AI Text + Voice Test\n"
        "Type a message, receive a streamed response from your local Ollama model, "
        "then hear the completed response spoken by Kokoro.\n\n"
        "**Text input only. No microphone or Vonage connection is used.**"
    )

    with gr.Row():
        model_box = gr.Textbox(label="Ollama model", value=DEFAULT_MODEL, scale=2)
        voice_box = gr.Dropdown(
            label="Kokoro voice",
            choices=[
                "af_heart", "af_bella", "af_nicole", "af_sarah",
                "am_adam", "am_michael",
            ],
            value=DEFAULT_VOICE,
            allow_custom_value=True,
            scale=1,
        )

    with gr.Row():
        check_button = gr.Button("Test Ollama connection")
        connection_status = gr.Markdown(
            "Ollama endpoint: `" + f"{ollama_native_base_url()}/api/chat" + "`"
        )

    system_box = gr.Textbox(
        label="System instructions",
        value=(
            "You are a friendly, natural conversational assistant. "
            "Reply in one or two short sentences, usually under 25 words. "
            "Do not show reasoning, markdown, or bullet points. Ask one question at a time."
        ),
        lines=3,
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
    # Bind only to the local machine; do not expose this UI to your network.
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        share=False,
    )
