
import json
import os
import re
import tempfile
import unicodedata
from pathlib import Path

import gradio as gr
import numpy as np
import requests
import soundfile as sf
from kokoro import KPipeline


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL", "http://localhost:11434/v1"
).rstrip("/")

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:0.8b")
DEFAULT_VOICE = os.getenv("KOKORO_VOICE", "af_heart")
KOKORO_LANG = os.getenv("KOKORO_LANG", "a")

SAMPLE_RATE = 24000
TEMP_DIR = Path(tempfile.gettempdir()) / "local_voice_ai_text"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

_kokoro_pipeline = None


# ============================================================
# SYSTEM PROMPT
# ============================================================

DEFAULT_SYSTEM_PROMPT = (
    "You are a natural, friendly conversational partner in a real-time "
    "voice conversation. Sound relaxed, warm, and human. Use contractions "
    "and natural conversational rhythm. "
    "ALWAYS begin every answer with exactly one short, natural conversational "
    "opener, chosen from phrases such as 'Hmm,', 'Okay,', 'I see,', "
    "'Well,', or 'Right,'. Vary the opener naturally. "
    "Immediately continue with the actual answer. Never repeat the opener. "
    "Do not say 'let me think about that', 'give me a moment', or other "
    "thinking announcements. Do not add a separate thinking statement. "
    "For example: 'Hmm, the easiest solution is to restart the application.' "
    "For example: 'Okay, change the port back to 7860.' "
    "For example: 'I see, that error means the port is already occupied.' "
    "Never reveal hidden reasoning. "
    "NEVER use emojis, emoticons, pictographs, or decorative symbols. "
    "Use plain text only, without markdown, headings, or bullet points. "
    "Usually answer in one to three short sentences, preferably under "
    "35 words. Avoid robotic or repetitive phrasing. "
    "Ask only one question at a time."
)


# ============================================================
# OLLAMA URL
# ============================================================

def ollama_native_base_url():
    base_url = OLLAMA_BASE_URL.rstrip("/")

    if base_url.endswith("/v1"):
        base_url = base_url[:-3]

    return base_url


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_response_text(text: str) -> str:
    """Remove emojis and pictographs while preserving ordinary text."""

    if not text:
        return ""

    cleaned = []

    for char in text:
        codepoint = ord(char)
        category = unicodedata.category(char)

        # Remove formatting controls and surrogate characters.
        if category in ("Cf", "Cs"):
            continue

        # Remove variation selectors.
        if (
            0xFE00 <= codepoint <= 0xFE0F
            or 0xE0100 <= codepoint <= 0xE01EF
        ):
            continue

        # Emoji, pictographs, flags, and common symbol blocks.
        if (
            0x1F000 <= codepoint <= 0x1FAFF
            or 0x1FC00 <= codepoint <= 0x1FFFF
            or 0x2600 <= codepoint <= 0x27BF
            or 0x2300 <= codepoint <= 0x23FF
            or 0x2B00 <= codepoint <= 0x2BFF
            or 0x1F1E6 <= codepoint <= 0x1F1FF
            or category == "So"
        ):
            continue

        cleaned.append(char)

    text = "".join(cleaned)

    # Remove common text emoticons.
    text = re.sub(
        r"(?<!\w)(?:[:;=8xX][-^']?[)(/\\DPp]|<3)(?!\w)",
        "",
        text,
    )

    # Remove markdown heading and bullet markers at line starts.
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", text)
    text = re.sub(r"(?m)^\s*[-*+]\s+", "", text)

    # Clean spacing left behind by removed characters.
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" *([,.!?])", r"\1", text)
    text = re.sub(r"([,.!?])([^\s])", r"\1 \2", text)

    return text.strip()


# ============================================================
# NATURAL OPENING
# ============================================================

OPENING_PATTERN = re.compile(
    r"^\s*(Hmm|Okay|I see|Well|Right)\s*[,!.]?\s*",
    re.IGNORECASE,
)


def ensure_natural_opener(text: str) -> str:
    """
    Ensure the answer begins with one conversational opener.
    Remove consecutive opening fillers to avoid duplication.
    """

    text = clean_response_text(text)

    if not text:
        return ""

    first_opener = None

    match = OPENING_PATTERN.match(text)

    if match:
        first_opener = match.group(1)
        text = text[match.end():].lstrip()

        # Remove any immediately repeated opener.
        while True:
            repeated = OPENING_PATTERN.match(text)

            if not repeated:
                break

            text = text[repeated.end():].lstrip()

        # Standardize punctuation for the first opener.
        opener = first_opener[0].upper() + first_opener[1:]

    else:
        # Reliable fallback if the model ignores the prompt.
        opener = "Okay"

    if text:
        # Avoid doubled punctuation after the opener.
        text = re.sub(r"^[,;:\s]+", "", text)
        return f"{opener}, {text}".strip()

    return f"{opener}."


# ============================================================
# KOKORO TEXT-TO-SPEECH
# ============================================================

def get_kokoro_pipeline():
    global _kokoro_pipeline

    if _kokoro_pipeline is None:
        _kokoro_pipeline = KPipeline(lang_code=KOKORO_LANG)

    return _kokoro_pipeline


def make_speech(text, voice):
    """Generate the final spoken answer."""

    # The exact cleaned, opened text is what gets spoken.
    text = ensure_natural_opener(text)

    if not text:
        raise RuntimeError("There is no text available to speak.")

    voice = (voice or DEFAULT_VOICE).strip()
    pipeline = get_kokoro_pipeline()
    audio_chunks = []

    try:
        for _, _, audio in pipeline(
            text,
            voice=voice,
            speed=1.0,
        ):
            if audio is None:
                continue

            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()

            audio_chunks.append(
                np.asarray(audio, dtype=np.float32).reshape(-1)
            )

    except Exception as exc:
        raise RuntimeError(
            f"Kokoro speech generation failed: {exc}"
        ) from exc

    if not audio_chunks:
        raise RuntimeError("Kokoro did not generate any audio.")

    waveform = np.concatenate(audio_chunks)

    fd, output_path = tempfile.mkstemp(
        prefix="kokoro_reply_",
        suffix=".wav",
        dir=TEMP_DIR,
    )
    os.close(fd)

    sf.write(output_path, waveform, SAMPLE_RATE)

    return output_path


# ============================================================
# CONVERSATION HISTORY
# ============================================================

def build_messages(user_text, history, system_prompt):
    prompt = (system_prompt or "").strip()

    if not prompt:
        prompt = DEFAULT_SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": prompt}
    ]

    for item in (history or [])[-12:]:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content", "")

        if (
            role in ("user", "assistant")
            and isinstance(content, str)
            and content.strip()
        ):
            messages.append({
                "role": role,
                "content": clean_response_text(content),
            })

    messages.append({
        "role": "user",
        "content": clean_response_text(user_text),
    })

    return messages


# ============================================================
# STREAM RESPONSE FROM OLLAMA
# ============================================================

def stream_ollama_reply(
    user_text,
    history,
    system_prompt,
    model_name,
):
    model_name = (model_name or DEFAULT_MODEL).strip()

    if not model_name:
        raise RuntimeError("Enter a model name from `ollama list`.")

    url = f"{ollama_native_base_url()}/api/chat"

    payload = {
        "model": model_name,
        "messages": build_messages(
            user_text,
            history,
            system_prompt,
        ),
        "think": False,
        "stream": True,
        "keep_alive": "30m",
        "options": {
            "temperature": 0.5,
            "num_predict": 100,
            "num_ctx": 2048,
        },
    }

    accumulated = ""

    try:
        with requests.post(
            url,
            json=payload,
            stream=True,
            timeout=(10, 180),
        ) as response:

            if not response.ok:
                raise RuntimeError(
                    f"Ollama returned HTTP {response.status_code}: "
                    f"{response.text[:1000]}"
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
                content = (
                    message.get("content", "")
                    if isinstance(message, dict)
                    else ""
                )

                if content:
                    accumulated += content

                    # Keep the displayed response clean and make sure
                    # it has an opener even if the model omits one.
                    yield ensure_natural_opener(accumulated)

                if chunk.get("done"):
                    break

    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Cannot connect to Ollama at {ollama_native_base_url()}. "
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
            "Ollama returned no visible text. Check that the selected "
            "model is installed and responding."
        )


# ============================================================
# SEND MESSAGE
# ============================================================

def send_message(
    user_text,
    history,
    system_prompt,
    model_name,
    voice,
):
    user_text = (user_text or "").strip()
    history = list(history or [])

    if not user_text:
        yield "", history, None, "Enter a message first."
        return

    actual_model = (model_name or DEFAULT_MODEL).strip()
    actual_voice = (voice or DEFAULT_VOICE).strip()

    user_text = clean_response_text(user_text)

    updated_history = history + [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": ""},
    ]

    yield (
        "",
        updated_history,
        None,
        f"Waiting for {actual_model}...",
    )

    reply = ""

    try:
        for partial_reply in stream_ollama_reply(
            user_text,
            history,
            system_prompt,
            actual_model,
        ):
            reply = ensure_natural_opener(partial_reply)

            updated_history[-1] = {
                "role": "assistant",
                "content": reply,
            }

            yield (
                "",
                updated_history,
                None,
                f"Receiving response from {actual_model}...",
            )

    except Exception as exc:
        error_text = str(exc)

        updated_history[-1] = {
            "role": "assistant",
            "content": f"Ollama error: {error_text}",
        }

        yield (
            "",
            updated_history,
            None,
            f"Ollama error: {error_text}",
        )
        return

    # Final cleaning and opener enforcement.
    reply = ensure_natural_opener(reply)

    if not reply:
        yield (
            "",
            updated_history,
            None,
            "The model returned an empty response.",
        )
        return

    updated_history[-1] = {
        "role": "assistant",
        "content": reply,
    }

    yield (
        "",
        updated_history,
        None,
        f"Text ready. Generating speech with Kokoro voice {actual_voice}...",
    )

    try:
        audio_path = make_speech(reply, actual_voice)

        yield (
            "",
            updated_history,
            audio_path,
            f"Reply generated using {actual_model} and Kokoro voice {actual_voice}.",
        )

    except Exception as exc:
        yield (
            "",
            updated_history,
            None,
            f"Text is ready, but Kokoro speech generation failed: {exc}",
        )


# ============================================================
# OTHER ACTIONS
# ============================================================

def clear_conversation():
    return "", [], None, "Conversation cleared."


def test_ollama(model_name):
    model_name = (model_name or DEFAULT_MODEL).strip()
    url = f"{ollama_native_base_url()}/api/tags"

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()

        data = response.json()
        models = [
            item.get("name", "")
            for item in data.get("models", [])
        ]

        if models and model_name not in models:
            return (
                f"Ollama is reachable, but {model_name} was not listed. "
                f"Available models: {', '.join(models)}"
            )

        if not models:
            return (
                f"Ollama is reachable at {url}, "
                "but it returned no installed models."
            )

        return f"Ollama is reachable. Selected model: {model_name}."

    except Exception as exc:
        return f"Could not reach Ollama at {url}: {exc}"


# ============================================================
# GRADIO INTERFACE
# ============================================================

with gr.Blocks(
    title="Local Text + Voice AI Test",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown(
        "# Local AI Text + Voice Test\n"
        "Chat with your local Ollama model and hear responses using Kokoro.\n\n"
        "**Text input only. No microphone or Vonage connection is used.**"
    )

    with gr.Row():
        model_box = gr.Textbox(
            label="Ollama model",
            value=DEFAULT_MODEL,
            scale=2,
        )

        voice_box = gr.Dropdown(
            label="Kokoro voice",
            choices=[
                "af_heart",
                "af_bella",
                "af_nicole",
                "af_sarah",
                "am_adam",
                "am_michael",
                "am_echo",
            ],
            value=DEFAULT_VOICE,
            allow_custom_value=True,
            scale=1,
        )

    with gr.Row():
        check_button = gr.Button("Test Ollama connection")

        connection_status = gr.Markdown(
            f"Ollama endpoint: `{ollama_native_base_url()}/api/chat`"
        )

    system_box = gr.Textbox(
        label="System instructions",
        value=DEFAULT_SYSTEM_PROMPT,
        lines=6,
    )

    chat = gr.Chatbot(
        label="Conversation",
        type="messages",
        height=380,
        allow_tags=False,
    )

    with gr.Row():
        message_box = gr.Textbox(
            label="Your message",
            placeholder="Type a message and press Enter...",
            lines=2,
            scale=5,
        )

        send_button = gr.Button(
            "Send",
            variant="primary",
            scale=1,
        )

    audio_output = gr.Audio(
        label="Kokoro spoken reply",
        type="filepath",
    )

    status = gr.Markdown("Ready. Enter a message to start testing.")

    clear_button = gr.Button("Clear conversation")

    inputs = [
        message_box,
        chat,
        system_box,
        model_box,
        voice_box,
    ]

    outputs = [
        message_box,
        chat,
        audio_output,
        status,
    ]

    send_button.click(
        send_message,
        inputs=inputs,
        outputs=outputs,
    )

    message_box.submit(
        send_message,
        inputs=inputs,
        outputs=outputs,
    )

    clear_button.click(
        clear_conversation,
        inputs=[],
        outputs=outputs,
    )

    check_button.click(
        test_ollama,
        inputs=[model_box],
        outputs=[connection_status],
    )


if __name__ == "__main__":
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        share=False,
    )
