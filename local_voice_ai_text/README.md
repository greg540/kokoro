# Local Text + Voice AI Test (Windows)

A small local web app for testing **typed input → Ollama → Kokoro TTS**. This first version deliberately uses text input only—no microphone and no Vonage connection.

## Requirements

- 64-bit Python 3.11 recommended.
- Ollama running on this computer at `http://localhost:11434`.
- A model already downloaded in Ollama (the default in the app is `qwen2.5-coder:14b`).
- `espeak-ng` installed on Windows for Kokoro's pronunciation/phonemization support.

## Setup

1. Install `espeak-ng` from its official releases: https://github.com/espeak-ng/espeak-ng/releases. Run the Windows x64 `.msi` installer and restart your terminal afterwards.
2. Extract this folder anywhere, for example to `C:\Users\gmitchell\kokoro\local_voice_ai_text`. It has its own `.venv` and won't replace your existing Kokoro environment.
3. Double-click `start.bat`.
4. On first launch, the script creates a virtual environment and installs requirements. Kokoro may also download its model assets the first time it is initialized, so the first run needs internet access. Packages and model assets are cached for subsequent runs.
5. The browser should open `http://127.0.0.1:7860`. If it doesn't, open that address manually.
6. Click **Test Ollama connection**, type a message, and press **Send**. The local model response appears in the conversation and a WAV player provides the Kokoro speech.

## Settings

- **Ollama model:** Any model name shown by `ollama list`; default `qwen2.5-coder:14b`.
- **Kokoro voice:** Default `af_heart`; a few English voices are included in the dropdown.
- **System instructions:** Change how the model responds.

Environment variables can override defaults before launch:

- `OLLAMA_BASE_URL` (default `http://localhost:11434/v1`)
- `OLLAMA_MODEL` (default `qwen2.5-coder:14b`)
- `KOKORO_VOICE` (default `af_heart`)
- `KOKORO_LANG` (default `a`, American English)

## Troubleshooting

- **Ollama won't connect:** Start Ollama, then run `ollama list`. The app uses Ollama's OpenAI-compatible `/v1/chat/completions` endpoint.
- **Model isn't found:** In PowerShell, run `ollama list` and copy the exact model name into the app's model box.
- **`espeak-ng` or phonemizer error:** Install the Windows x64 `espeak-ng` installer from the official release page, restart the terminal, and launch again.
- **First Kokoro launch is slow or fails offline:** The library may need to download model assets on first use. Run once while connected, then cached assets can be used locally.
- **Port 7860 is busy:** Close the other app using that port, or change `server_port=7860` near the bottom of `app.py`.
- **Privacy:** Gradio binds to `127.0.0.1` with sharing disabled; the UI is intended to be accessible only on the same computer.

This is a turn-based test harness. It does not yet record microphone audio, continuously listen, place telephone calls, or implement real-time barge-in.
