import json
import os
import re
import random
import tempfile
import threading
import time
import unicodedata
import queue
from collections import deque
from pathlib import Path

import gradio as gr
import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
import torch
from faster_whisper import WhisperModel


def env_bool(name, default=True):
    """Parse a boolean env var properly ('0', 'false', 'no', 'off', '' are False)."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


# ---------------- Configuration ----------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1").rstrip("/")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")

# Fish Audio TTS (https://docs.fish.audio/features/text-to-speech)
FISH_API_KEY = os.getenv("FISH_API_KEY", "sk-fish-C26H09jJCvkMEi_zPk0C0Ps_mM1iGTd5LvcGxJNxUfg").strip()
FISH_MODEL = os.getenv("FISH_MODEL", "s2.1-pro-free").strip()
FISH_TTS_URL = os.getenv("FISH_TTS_URL", "https://api.fish.audio/v1/tts")
FISH_SAMPLE_RATE = int(os.getenv("FISH_SAMPLE_RATE", "44100"))
FISH_LATENCY = os.getenv("FISH_LATENCY", "balanced")  # "balanced" (fast) or "normal" (most stable)
FISH_SPEED = float(os.getenv("FISH_SPEED", "1.0"))    # 0.5 - 2.0
# Optional voice model id (reference_id) from the Fish Audio voice library. Empty = default voice.
DEFAULT_VOICE = os.getenv("FISH_VOICE_ID", "").strip()
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "base.en")

# Tune these if needed. RMS is measured on float audio in the range -1.0 to 1.0.
VAD_RMS_THRESHOLD = float(os.getenv("VAD_RMS_THRESHOLD", "0.035"))
VAD_START_FRAMES = int(os.getenv("VAD_START_FRAMES", "6"))
VAD_END_SILENCE_SECONDS = float(os.getenv("VAD_END_SILENCE_SECONDS", "0.70"))
VAD_MIN_UTTERANCE_SECONDS = float(os.getenv("VAD_MIN_UTTERANCE_SECONDS", "0.30"))

# Automatic barge-in threshold settings
AUTO_THRESHOLD_MIN = float(os.getenv("AUTO_THRESHOLD_MIN", "0.020"))
AUTO_THRESHOLD_MAX = float(os.getenv("AUTO_THRESHOLD_MAX", "0.080"))
AUTO_THRESHOLD_SMOOTHING = 0.10

# Automatic barge-in threshold is derived from the live noise floor * multiplier.
THRESHOLD_MULTIPLIER = float(os.getenv("THRESHOLD_MULTIPLIER", "3.0"))
# Seconds of processing silence before a random "thinking" filler is spoken.
THINKING_FILLER_DELAY = float(os.getenv("THINKING_FILLER_DELAY", "2.0"))

# Silero VAD (pip install silero-vad): a block counts as speech only when it is loud
# enough (RMS) AND Silero rates it as human speech, so machinery/clicks/static are rejected.
USE_SILERO_VAD = env_bool("USE_SILERO_VAD", True)
VAD_PROB_THRESHOLD = float(os.getenv("VAD_PROB_THRESHOLD", "0.5"))
SILERO_WINDOW = 512        # Silero requires exactly 512 samples per call at 16 kHz
SILERO_RATE = 16000

MAX_EVENTS = 5000          # events retained in memory (oldest dropped first)
LIVE_LOG_LINES = 300       # lines shown in the live log view
NOISE_LOG_MIN_FRAMES = 2   # ignored noise bursts shorter than this are not logged
NOISE_LOG_INTERVAL = 0.5   # seconds between "noise ignored" log lines

TEMP_DIR = Path(tempfile.gettempdir()) / "local_voice_ai_text"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SYSTEM_PROMPT = """You are a natural, helpful phone assistant speaking to a customer.
Always begin with one natural conversational opener when it fits, such as Hmm, Okay, I see, Well, or Right.
Do not describe your thinking or say that you are processing.
Use plain text, no emoji, no markdown, and no stage directions.
Keep spoken replies concise: usually 1–3 short sentences and under 35 words.
Ask one clear question at a time when you need information.
Sound calm, attentive, and professional. Do not claim to be human."""

# Short natural phrases spoken to fill the processing gap so there is no dead air.
THINKING_FILLERS = [
    "Hmm, let me think about that for a moment.",
    "Let me consider that for a second.",
    "Just give me a moment to check that.",
    "Let me get that straight for you.",
    "One moment while I look into that.",
    "Okay, let me think through that.",
    "Let me pull that together for you.",
    "Hold on a moment while I check.",
]

# Canned first-call introduction played once when the caller's first words (e.g. "hello")
# are detected, before the normal Ollama-generated reply.
FIRST_INTRODUCTION = os.getenv(
    "FIRST_INTRODUCTION",
    "Hi, thanks for calling. I'm the automated assistant here. How can I help you today?",
)
PLAY_FIRST_INTRODUCTION = True

# Words/phrases Whisper should favor during recognition. A curated default list of
# wood-splitter related terms is always used; add your own via WHISPER_HOTWORDS.
DEFAULT_WHISPER_HOTWORDS = [
    # machine & parts
    "splitter", "splitting", "split", "log", "logs", "firewood", "wood",
    "wedge", "wedges", "ram", "cylinder", "cylinders", "hydraulic", "hydraulics",
    "pump", "motor", "engine", "deck", "axle", "trailer", "tow", "blade",
    "concave", "cutting", "gearbox", "chain", "valve", "hose", "frame",
    # action & feedback
    "crack", "cracked", "cracking", "crushes", "splits", "jam", "stuck",
    "grain", "knob", "knot", "ton", "tons", "pressure", "throttle", "choke",
]
WHISPER_HOTWORDS = sorted(
    set(
        DEFAULT_WHISPER_HOTWORDS
        + [w.strip() for w in os.getenv("WHISPER_HOTWORDS", "").split(",") if w.strip()]
    )
)

_speech_cache = {}  # (text, voice, model, rate) -> wav path, for fixed phrases
_whisper_model = None
_silero_model = None
_model_lock = threading.Lock()

# Audio capture continues independently from transcription, Ollama, and playback.
_audio_running = threading.Event()
_interrupt_playback = threading.Event()
_cancel_response = threading.Event()
_bot_active = threading.Event()  # set while the bot is generating / synthesizing / speaking
_audio_queue = queue.Queue(maxsize=120)
_turn_queue = queue.Queue(maxsize=8)
_audio_stream = None
_monitor_thread = None
_turn_thread = None
_state_lock = threading.RLock()
_input_device_index = None
_output_device_index = None
_input_sample_rate = 48000
_voice_model = DEFAULT_MODEL
_voice_name = DEFAULT_VOICE
_voice_system_prompt = DEFAULT_SYSTEM_PROMPT
_voice_history = []
_voice_status = "Voice mode stopped."
_input_device_name = "-"
_output_device_name = "-"

# Automatic threshold + thinking-filler toggles (live-settable from the UI).
AUTO_BARGE_THRESHOLD = False
PLAY_THINKING_FILLERS = True
_first_intro_played = False  # reset per session in start_voice_mode


# ---------------- Event log (transcript + diagnostics) ----------------
# kinds: CALLER (caller speech), BOT (bot replies), SYSTEM (debug/info),
#        BARGE (barge-in diagnostics), ERROR (mic / Whisper / Ollama / Fish Audio / playback errors)
_events = []
_events_lock = threading.Lock()

# Rolling input-level history (one RMS value per ~21 ms audio block) for the live meter.
_level_history = deque(maxlen=300)

# Live audio metrics, written by the monitor thread and read by the UI.
_metrics = {
    "rms_now": 0.0,
    "vad_now": 0.0,
    "vad_true_frames": 0,
    "rms_peak_recent": 0.0,
    "rms_peak_session": 0.0,
    "noise_floor": 0.0,
    "speech_active": False,
    "frames_over": 0,
    "blocks_dropped": 0,
    "mic_status": "",
    "utterances": 0,
}


def log_event(kind, text, source="voice"):
    with _events_lock:
        _events.append({"time": time.time(), "kind": kind, "source": source, "text": str(text)})
        if len(_events) > MAX_EVENTS:
            del _events[: len(_events) - MAX_EVENTS]


def format_event(event, include_kind=True):
    t = event["time"]
    stamp = time.strftime("%H:%M:%S", time.localtime(t)) + f".{int(t * 1000) % 1000:03d}"
    kind = event["kind"]
    if event["source"] == "text" and kind in ("CALLER", "BOT"):
        kind = f"{kind} (text)"
    return f"[{stamp}] {kind}: {event['text']}" if include_kind else f"[{stamp}] {event['text']}"


def snapshot_events(kinds=None, limit=None):
    with _events_lock:
        items = [e for e in _events if kinds is None or e["kind"] in kinds]
    if limit:
        items = items[-limit:]
    return items


def transcript_text():
    """Full conversation only: every caller utterance and bot reply, in order."""
    return "\n".join(format_event(e) for e in snapshot_events({"CALLER", "BOT"}))


def live_log_text():
    """Everything (conversation + system + barge-in + errors) as it happens."""
    return "\n".join(format_event(e) for e in snapshot_events(limit=LIVE_LOG_LINES))


def barge_log_text():
    return "\n".join(format_event(e, include_kind=False) for e in snapshot_events({"BARGE"}, limit=200))


def error_log_text():
    return "\n".join(format_event(e, include_kind=False) for e in snapshot_events({"ERROR"}, limit=200))


def audio_diagnostics_text():
    m = dict(_metrics)
    running = _audio_running.is_set()
    floor = m["noise_floor"]
    suggested = max(0.005, min(0.08, floor * 3.0)) if floor > 0 else None
    lines = [
        f"Voice mode:            {'RUNNING' if running else 'stopped'}",
        f"Input device:          {_input_device_name}  ({_input_sample_rate} Hz)",
        f"Output device:         {_output_device_name}",
        f"Current level (RMS):   {m['rms_now']:.4f}   {'#' * min(40, int(m['rms_now'] * 400))}",
        f"Silero VAD (prob):     {m['vad_now']:.3f}   {'#' * min(40, int(m['vad_now'] * 40))}",
        f"Recent peak:           {m['rms_peak_recent']:.4f}",
        f"Session peak:          {m['rms_peak_session']:.4f}",
        f"Noise floor (est.):    {floor:.4f}" + (f"   (suggested threshold ≈ {suggested:.4f})" if suggested else ""),
        f"Barge-in threshold:    {VAD_RMS_THRESHOLD:.4f}" + ("   (automatic)" if AUTO_BARGE_THRESHOLD else ""),
        f"Auto noise multiplier: {THRESHOLD_MULTIPLIER:.1f}x" + ("   ← used" if AUTO_BARGE_THRESHOLD else "   (only when automatic)"),
        f"Silero VAD:            {'enabled' if USE_SILERO_VAD else 'RMS-only'}" + ("   (prob ≥ {:.2f})".format(VAD_PROB_THRESHOLD) if USE_SILERO_VAD else ""),
        f"Thinking fillers:      {'on' if PLAY_THINKING_FILLERS else 'off'}   (fills the processing gap)",
        f"First introduction:    {'on' if PLAY_FIRST_INTRODUCTION else 'off'}" + ("  (played)" if _first_intro_played else "  (pending)"),
        f"Hotwords:              {len(WHISPER_HOTWORDS)} term(s) favored by Whisper",
        f"Confirmation frames:   {VAD_START_FRAMES}   (currently {m['frames_over']} consecutive above threshold)",
        f"Caller speaking now:   {'yes' if m['speech_active'] else 'no'}",
        f"Bot active:            {'yes' if _bot_active.is_set() else 'no'}",
        f"Utterances captured:   {m['utterances']}",
        f"Audio blocks dropped:  {m['blocks_dropped']}",
        f"Last mic status:       {m['mic_status'] or 'OK'}",
    ]
    return "\n".join(lines)


def render_level_meter(threshold=None):
    """Rolling input-volume graph with the barge-in threshold drawn over it."""
    try:
        thr = float(threshold) if threshold is not None else VAD_RMS_THRESHOLD
    except (TypeError, ValueError):
        thr = VAD_RMS_THRESHOLD
    # Always show the threshold actually in use (it moves in automatic mode).
    if AUTO_BARGE_THRESHOLD:
        thr = VAD_RMS_THRESHOLD
    levels = list(_level_history)
    n = _level_history.maxlen
    W, H = 1000, 200
    peak = max(levels) if levels else 0.0
    ymax = max(0.10, thr * 1.5, peak * 1.15)
    bw = W / n
    offset = n - len(levels)
    bars = []
    for i, v in enumerate(levels):
        h = min(H, v / ymax * H)
        color = "#c9605f" if v >= thr else "#4a9d78"
        bars.append(
            f'<rect x="{(offset + i) * bw:.2f}" y="{H - h:.2f}" width="{bw + 0.3:.2f}" height="{h:.2f}" fill="{color}"/>'
        )
    ty = H - thr / ymax * H
    ty_pct = ty / H * 100.0
    now = levels[-1] if levels else 0.0
    state = "ABOVE threshold" if now >= thr else "below threshold"
    state_color = "#c9605f" if now >= thr else "#4a9d78"
    return (
        '<div style="font-family:monospace;font-size:13px;margin-bottom:4px;">'
        f'Input level: <b>{now:.4f}</b> &nbsp;|&nbsp; Barge-in threshold: <b>{thr:.4f}</b> &nbsp;|&nbsp; '
        f'<b style="color:{state_color}">{state}</b> '
        '&nbsp;<span style="opacity:.7">(green = below, red = above; last ~6 s)</span></div>'
        '<div style="position:relative;width:100%;height:200px;background:rgba(127,127,127,.12);'
        'border:1px solid rgba(127,127,127,.35);border-radius:6px;overflow:hidden;">'
        f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="none" width="100%" height="100%" '
        'xmlns="http://www.w3.org/2000/svg" style="display:block;">'
        + "".join(bars)
        + f'<line x1="0" x2="{W}" y1="{ty:.2f}" y2="{ty:.2f}" stroke="#f5a623" stroke-width="2" '
        'stroke-dasharray="8 5" vector-effect="non-scaling-stroke"/>'
        '</svg>'
        f'<span style="position:absolute;right:6px;top:calc({ty_pct:.2f}% - 18px);font-family:monospace;'
        'font-size:12px;font-weight:bold;color:#f5a623;background:rgba(0,0,0,.55);padding:0 5px;'
        f'border-radius:3px;">threshold {thr:.4f}</span>'
        f'<span style="position:absolute;left:6px;top:4px;font-family:monospace;font-size:11px;opacity:.7;">'
        f'{ymax:.3f}</span>'
        '<span style="position:absolute;left:6px;bottom:4px;font-family:monospace;font-size:11px;opacity:.7;">0</span>'
        '</div>'
    )


def apply_live_settings(threshold, frames):
    """Apply slider changes immediately so the threshold can be tuned while listening."""
    global VAD_RMS_THRESHOLD, VAD_START_FRAMES
    VAD_RMS_THRESHOLD = float(threshold)
    VAD_START_FRAMES = int(frames)
    log_event("SYSTEM", f"Settings changed live: threshold {VAD_RMS_THRESHOLD:.4f}, confirmation frames {VAD_START_FRAMES}.")


def set_auto_threshold(auto):
    """Toggle automatic barge-in threshold; lock the manual threshold slider when on."""
    global AUTO_BARGE_THRESHOLD
    AUTO_BARGE_THRESHOLD = bool(auto)
    log_event("SYSTEM", f"Automatic barge-in threshold: {'ON' if AUTO_BARGE_THRESHOLD else 'off'}.")
    # Disable the manual threshold slider while automatic is on, enable the multiplier instead.
    return gr.update(interactive=not AUTO_BARGE_THRESHOLD), gr.update(interactive=bool(AUTO_BARGE_THRESHOLD))


def set_threshold_multiplier(value):
    """Apply the automatic-threshold noise multiplier live."""
    global THRESHOLD_MULTIPLIER
    THRESHOLD_MULTIPLIER = float(value)
    log_event("SYSTEM", f"Auto threshold multiplier set to {THRESHOLD_MULTIPLIER:.1f}x noise floor.")


def set_thinking_fillers(on):
    global PLAY_THINKING_FILLERS
    PLAY_THINKING_FILLERS = bool(on)
    log_event("SYSTEM", f"Thinking fillers: {'ON' if PLAY_THINKING_FILLERS else 'off'}.")


def set_first_intro(enabled, text):
    """Toggle and edit the canned first-call introduction (live-settable)."""
    global PLAY_FIRST_INTRODUCTION, FIRST_INTRODUCTION
    PLAY_FIRST_INTRODUCTION = bool(enabled)
    cleaned = (text or "").strip()
    if cleaned:
        FIRST_INTRODUCTION = cleaned
    log_event(
        "SYSTEM",
        f"First-call introduction: {'enabled' if PLAY_FIRST_INTRODUCTION else 'disabled'}.",
    )


def set_silero_vad(on):
    """Toggle Silero VAD speech detection live (falls back to RMS-only if unavailable)."""
    global USE_SILERO_VAD
    USE_SILERO_VAD = bool(on)
    log_event(
        "SYSTEM",
        f"Silero VAD speech detection: {'enabled' if USE_SILERO_VAD else 'disabled (RMS-only)'}.",
    )


def set_fish_settings(api_key, model):
    """Apply the Fish Audio API key / model live (empty fields keep the current value)."""
    global FISH_API_KEY, FISH_MODEL
    if (api_key or "").strip():
        FISH_API_KEY = api_key.strip()
    if (model or "").strip():
        FISH_MODEL = model.strip()


def refresh_views():
    return audio_diagnostics_text(), transcript_text(), live_log_text(), barge_log_text(), error_log_text(), get_voice_status()


def clear_logs():
    with _events_lock:
        _events.clear()
    _metrics.update(rms_peak_session=0.0, rms_peak_recent=0.0, utterances=0, blocks_dropped=0, mic_status="")
    log_event("SYSTEM", "Logs cleared.")
    return refresh_views()[1:5]


def export_transcript():
    """Write a troubleshooting file: settings, transcript, full event log, errors."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = TEMP_DIR / f"conversation_{stamp}.txt"
    sections = [
        "LOCAL VOICE AI — CONVERSATION EXPORT",
        f"Exported:   {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model:      {_voice_model}",
        f"Voice:      {_voice_name}",
        f"Input:      {_input_device_name} ({_input_sample_rate} Hz)",
        f"Output:     {_output_device_name}",
        f"Threshold:  {VAD_RMS_THRESHOLD:.4f}   Confirmation frames: {VAD_START_FRAMES}",
        "",
        "=== TRANSCRIPT (caller + bot) ===",
        transcript_text() or "(empty)",
        "",
        "=== FULL EVENT LOG (conversation, system, barge-in, errors) ===",
        "\n".join(format_event(e) for e in snapshot_events()) or "(empty)",
        "",
        "=== ERRORS ===",
        error_log_text() or "(none)",
        "",
    ]
    path.write_text("\n".join(sections), encoding="utf-8")
    return str(path)


# ---------------- LLM / speech helpers ----------------
def ollama_native_base_url():
    return OLLAMA_BASE_URL[:-3] if OLLAMA_BASE_URL.endswith("/v1") else OLLAMA_BASE_URL


def clean_response_text(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.I | re.S)
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s*[-*•]\s+", "", text)
    text = re.sub(r"[\U0001F000-\U0001FAFF\U00002600-\U000027BF]", "", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+([,.!?;:])", r"\1", text)


def ensure_natural_opener(text):
    text = clean_response_text(text)
    if not text:
        return ""
    if re.match(r"^(hmm|okay|ok|i see|well|right)\b", text, flags=re.I):
        return text
    if len(text) > 1:
        return "Okay, " + text[0].lower() + text[1:]
    return "Okay, " + text.lower()


def build_messages(history, user_text, system_prompt):
    messages = [{"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT}]
    for item in (history or [])[-12:]:
        if isinstance(item, dict) and item.get("role") in ("user", "assistant") and item.get("content"):
            messages.append({"role": item["role"], "content": str(item["content"])})
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            if item[0]:
                messages.append({"role": "user", "content": str(item[0])})
            if item[1]:
                messages.append({"role": "assistant", "content": str(item[1])})
    messages.append({"role": "user", "content": user_text})
    return messages


def stream_ollama_reply(history, user_text, model, system_prompt, cancel_event=None):
    url = f"{ollama_native_base_url()}/api/chat"
    payload = {
        "model": (model or DEFAULT_MODEL).strip(),
        "messages": build_messages(history, user_text, system_prompt),
        "stream": True,
        "think": False,
        "keep_alive": "30m",
        "options": {"temperature": 0.5, "num_predict": 100, "num_ctx": 2048},
    }
    with requests.post(url, json=payload, stream=True, timeout=(10, 180)) as response:
        response.raise_for_status()
        accumulated = ""
        for line in response.iter_lines(decode_unicode=True):
            if cancel_event is not None and cancel_event.is_set():
                break
            if not line:
                continue
            try:
                packet = json.loads(line)
            except json.JSONDecodeError:
                continue
            message = packet.get("message", {})
            accumulated += message.get("content", "") or ""
            if accumulated:
                yield ensure_natural_opener(accumulated)
            if packet.get("done"):
                break


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        with _model_lock:
            if _whisper_model is None:
                # CPU int8 is a reliable Windows default. Whisper downloads its model on first use.
                _whisper_model = WhisperModel(WHISPER_MODEL_NAME, device="cpu", compute_type="int8")
    return _whisper_model


# ---------------- Silero VAD (silero-vad package) ----------------
def get_silero_model():
    """Load Silero VAD from the `silero-vad` pip package. Returns None if unavailable."""
    global _silero_model
    if _silero_model is None:
        with _model_lock:
            if _silero_model is None:
                try:
                    from silero_vad import load_silero_vad

                    torch.set_num_threads(1)
                    _silero_model = load_silero_vad()
                    log_event("SYSTEM", "Silero VAD model loaded.")
                except Exception as exc:
                    log_event("ERROR", f"Silero VAD unavailable (pip install silero-vad): {exc}")
                    _silero_model = None
    return _silero_model


class StreamResampler:
    """Streaming mono resampler to 16 kHz that keeps state between blocks."""

    def __init__(self, src_rate, dst_rate=SILERO_RATE):
        self.src = int(src_rate)
        self.dst = int(dst_rate)
        self.buf = np.zeros(0, dtype=np.float32)
        self.pos = 0.0  # fractional read position (non-integer-ratio path)
        self.factor = self.src // self.dst if self.src % self.dst == 0 else None

    def process(self, x):
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if self.src == self.dst:
            return x
        self.buf = np.concatenate([self.buf, x])
        if self.factor:
            # Integer ratio (e.g. 48000 -> 16000): average groups of samples (cheap low-pass).
            usable = (len(self.buf) // self.factor) * self.factor
            if usable == 0:
                return np.zeros(0, dtype=np.float32)
            out = self.buf[:usable].reshape(-1, self.factor).mean(axis=1)
            self.buf = self.buf[usable:]
            return out.astype(np.float32)
        # Non-integer ratio (e.g. 44100 -> 16000): linear interpolation with carried position.
        step = self.src / float(self.dst)
        n = len(self.buf)
        if n < 2:
            return np.zeros(0, dtype=np.float32)
        count = int(np.floor((n - 1 - self.pos) / step)) + 1
        if count <= 0:
            return np.zeros(0, dtype=np.float32)
        idx = self.pos + step * np.arange(count)
        out = np.interp(idx, np.arange(n), self.buf)
        next_pos = self.pos + step * count
        drop = int(np.floor(next_pos))
        self.buf = self.buf[drop:]
        self.pos = next_pos - drop
        return out.astype(np.float32)


class SileroGate:
    """Feeds exact 512-sample 16 kHz windows to Silero and holds the latest probability."""

    def __init__(self, model, src_rate):
        self.model = model
        self.resampler = StreamResampler(src_rate)
        self.pending = np.zeros(0, dtype=np.float32)
        self.last_prob = 0.0
        self.reset()

    def reset(self):
        try:
            self.model.reset_states()
        except Exception:
            pass
        self.resampler = StreamResampler(self.resampler.src)
        self.pending = np.zeros(0, dtype=np.float32)
        self.last_prob = 0.0

    def process(self, block):
        """Return the speech probability after consuming this audio block."""
        self.pending = np.concatenate([self.pending, self.resampler.process(block)])
        probs = []
        while len(self.pending) >= SILERO_WINDOW:
            chunk = self.pending[:SILERO_WINDOW].copy()
            self.pending = self.pending[SILERO_WINDOW:]
            with torch.no_grad():
                probs.append(float(self.model(torch.from_numpy(chunk), SILERO_RATE).item()))
        if probs:
            self.last_prob = max(probs)
        return self.last_prob


def make_speech(text, voice, apply_opener=True, cache=False):
    """Synthesize `text` with the Fish Audio API and return the path of a WAV file.

    `voice` is an optional Fish voice model id (reference_id); empty uses the default voice.
    `cache=True` reuses audio for fixed phrases (thinking fillers, first introduction) so
    they play instantly and don't spend API calls.
    """
    if apply_opener:
        text = ensure_natural_opener(text)
    if not text:
        raise ValueError("There is no response text to synthesize.")
    if not FISH_API_KEY:
        raise RuntimeError("Fish Audio API key missing. Set FISH_API_KEY or enter it in the app.")

    voice = (voice or "").strip()
    cache_key = (text, voice, FISH_MODEL, FISH_SAMPLE_RATE)
    if cache:
        cached = _speech_cache.get(cache_key)
        if cached and Path(cached).exists():
            return cached

    payload = {
        "text": text,
        "format": "wav",
        "sample_rate": FISH_SAMPLE_RATE,
        "latency": FISH_LATENCY,
        "prosody": {"speed": FISH_SPEED, "volume": 0},
    }
    if voice:
        payload["reference_id"] = voice
    headers = {
        "Authorization": f"Bearer {FISH_API_KEY}",
        "Content-Type": "application/json",
        "model": FISH_MODEL,
    }
    try:
        response = requests.post(FISH_TTS_URL, json=payload, headers=headers, timeout=(10, 60))
    except requests.RequestException as exc:
        raise RuntimeError(f"Fish Audio request failed: {exc}") from exc
    if response.status_code != 200:
        hints = {
            401: " (check your API key)",
            402: " (payment/credit problem)",
            429: " (rate limit or fair-use limit reached)",
        }
        raise RuntimeError(
            f"Fish Audio HTTP {response.status_code}{hints.get(response.status_code, '')}: {response.text[:300]}"
        )
    if not response.content:
        raise RuntimeError("Fish Audio returned no audio.")
    path = TEMP_DIR / f"speech_{int(time.time() * 1000)}.wav"
    path.write_bytes(response.content)
    if cache:
        _speech_cache[cache_key] = str(path)
    return str(path)


def play_thinking_filler_if_needed(done_event, running_event, delay=None):
    """If generation is still running after `delay` seconds, speak a random natural filler.

    Interruptible: caller speech sets _interrupt_playback/_cancel_response and stops it.
    `done_event` is set by the worker once generation finishes so the filler is not spoken late.
    """
    if delay is None:
        delay = THINKING_FILLER_DELAY

    def _worker():
        time.sleep(delay)
        if done_event.is_set() or not running_event.is_set():
            return
        try:
            filler = random.choice(THINKING_FILLERS)
            path = make_speech(filler, _voice_name, apply_opener=False, cache=True)
            if done_event.is_set() or not running_event.is_set():
                return
            play_audio_interruptibly(path)
        except Exception as exc:
            log_event("SYSTEM", f"Thinking filler error: {exc}")

    threading.Thread(target=_worker, daemon=True).start()


def device_choices():
    labels = []
    for index, device in enumerate(sd.query_devices()):
        ins = int(device.get("max_input_channels", 0))
        outs = int(device.get("max_output_channels", 0))
        if ins or outs:
            labels.append(f"{index}: {device['name']} [in {ins}, out {outs}]")
    return labels


def parse_device_index(label):
    match = re.match(r"\s*(\d+):", str(label or ""))
    return int(match.group(1)) if match else None


def default_device_label(direction, preferred_terms):
    labels = device_choices()
    for label in labels:
        if not any(term in label.lower() for term in preferred_terms):
            continue
        index = parse_device_index(label)
        info = sd.query_devices(index)
        if direction == "input" and info["max_input_channels"] > 0:
            return label
        if direction == "output" and info["max_output_channels"] > 0:
            return label
    for label in labels:
        index = parse_device_index(label)
        info = sd.query_devices(index)
        if direction == "input" and info["max_input_channels"] > 0:
            return label
        if direction == "output" and info["max_output_channels"] > 0:
            return label
    return None


def resample_to_16k(audio, source_rate):
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if source_rate == 16000 or not len(audio):
        return audio
    new_length = max(1, int(round(len(audio) * 16000 / float(source_rate))))
    return np.interp(
        np.linspace(0, len(audio) - 1, new_length, dtype=np.float32),
        np.arange(len(audio), dtype=np.float32), audio,
    ).astype(np.float32)


def transcribe_audio(audio, sample_rate):
    audio16 = resample_to_16k(audio, sample_rate)
    transcribe_kwargs = dict(
        language="en", beam_size=1, vad_filter=True, condition_on_previous_text=False
    )
    if WHISPER_HOTWORDS:
        transcribe_kwargs["hotwords"] = " ".join(WHISPER_HOTWORDS)
    segments, _ = get_whisper_model().transcribe(audio16, **transcribe_kwargs)
    return clean_response_text(" ".join(segment.text.strip() for segment in segments))


def set_voice_status(message):
    global _voice_status
    with _state_lock:
        _voice_status = message


def get_voice_status():
    with _state_lock:
        return _voice_status


def stop_playback():
    # Also cancels a response currently being generated by Ollama.
    _interrupt_playback.set()
    _cancel_response.set()


def play_audio_interruptibly(path):
    """Play small blocks so a confirmed customer interruption can stop output quickly."""
    if _output_device_index is None or not _audio_running.is_set():
        return
    try:
        audio, rate = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        block = max(1, int(rate * 0.025))  # about 25 ms per block
        _interrupt_playback.clear()
        total = len(audio)
        with sd.OutputStream(
            device=_output_device_index, samplerate=rate, channels=1, dtype="float32", latency="low"
        ) as stream:
            for start in range(0, total, block):
                if not _audio_running.is_set() or _interrupt_playback.is_set():
                    pct = int(100 * start / max(1, total))
                    log_event("SYSTEM", f"Playback stopped early at {pct}% of the bot's audio.")
                    return
                stream.write(audio[start:start + block])
        log_event("SYSTEM", "Playback finished normally.")
    except Exception as exc:
        set_voice_status(f"Audio playback error: {exc}")
        log_event("ERROR", f"Audio playback error (output device): {exc}")


def audio_input_callback(indata, frames, time_info, status):
    if not _audio_running.is_set():
        return
    if status:
        # Keep the callback light; the monitor thread logs this.
        _metrics["mic_status"] = str(status)
    block = np.asarray(indata[:, 0], dtype=np.float32).copy()
    try:
        _audio_queue.put_nowait(block)
    except queue.Full:
        _metrics["blocks_dropped"] += 1
        try:
            _audio_queue.get_nowait()
            _audio_queue.put_nowait(block)
        except (queue.Empty, queue.Full):
            pass


def monitor_audio_loop():
    """Only detects speech and queues turns; never waits for Whisper/Ollama/TTS."""
    global _input_sample_rate, VAD_RMS_THRESHOLD
    speech_chunks = []
    pre_roll = []
    speech_active = False
    consecutive_voice = 0
    silence_frames = 0
    burst_peak = 0.0
    last_noise_log = 0.0
    last_mic_status_logged = ""
    last_peak_decay = time.time()
    block_seconds = 1024 / float(_input_sample_rate)
    end_silence_frames = max(1, int(VAD_END_SILENCE_SECONDS / block_seconds))

    # Silero is created lazily so it can also be switched on live from the UI.
    gate = None
    gate_failed = False
    gate_was_enabled = False

    while _audio_running.is_set():
        try:
            block = _audio_queue.get(timeout=0.15)
        except queue.Empty:
            continue
        if not len(block):
            continue
        rms = float(np.sqrt(np.mean(np.square(block, dtype=np.float64)) + 1e-12))

        # Automatic thresholding: derive the target from the live noise floor, clamp it to
        # [MIN, MAX], and move toward it gradually so the threshold doesn't jitter.
        if AUTO_BARGE_THRESHOLD and _metrics["noise_floor"] > 0:
            target_threshold = _metrics["noise_floor"] * THRESHOLD_MULTIPLIER
            target_threshold = max(AUTO_THRESHOLD_MIN, min(AUTO_THRESHOLD_MAX, target_threshold))
            VAD_RMS_THRESHOLD += AUTO_THRESHOLD_SMOOTHING * (target_threshold - VAD_RMS_THRESHOLD)

        # Silero VAD gate: require human speech, not just loudness.
        if USE_SILERO_VAD and gate is None and not gate_failed:
            model = get_silero_model()
            if model is None:
                gate_failed = True
            else:
                gate = SileroGate(model, _input_sample_rate)
        silero_on = USE_SILERO_VAD and gate is not None
        if silero_on and not gate_was_enabled:
            gate.reset()  # fresh state when (re)enabled
        gate_was_enabled = silero_on

        if silero_on:
            try:
                vad_prob = gate.process(block)
            except Exception as exc:
                log_event("ERROR", f"Silero VAD processing failed, falling back to RMS-only: {exc}")
                gate = None
                gate_failed = True
                vad_prob = 1.0
        else:
            vad_prob = 1.0

        loud = rms >= VAD_RMS_THRESHOLD
        voiced = loud and vad_prob >= VAD_PROB_THRESHOLD
        now = time.time()

        # ---- live metrics ----
        _metrics["rms_now"] = rms
        _metrics["vad_now"] = vad_prob
        _level_history.append(rms)
        _metrics["rms_peak_session"] = max(_metrics["rms_peak_session"], rms)
        if now - last_peak_decay > 2.0:
            _metrics["rms_peak_recent"] = 0.0
            last_peak_decay = now
        _metrics["rms_peak_recent"] = max(_metrics["rms_peak_recent"], rms)
        if not voiced and not speech_active:
            floor = _metrics["noise_floor"]
            _metrics["noise_floor"] = rms if floor == 0.0 else 0.98 * floor + 0.02 * rms
        mic_status = _metrics["mic_status"]
        if mic_status and mic_status != last_mic_status_logged:
            log_event("ERROR", f"Microphone stream status: {mic_status}")
            last_mic_status_logged = mic_status

        if not speech_active:
            pre_roll.append(block)
            if len(pre_roll) > 8:
                pre_roll.pop(0)
            if voiced:
                consecutive_voice += 1
                burst_peak = max(burst_peak, rms)
            else:
                # A short burst that never reached the confirmation count = background noise.
                if consecutive_voice >= NOISE_LOG_MIN_FRAMES and now - last_noise_log > NOISE_LOG_INTERVAL:
                    log_event(
                        "BARGE",
                        f"Noise ignored: {consecutive_voice} frame(s) above threshold, needs "
                        f"{VAD_START_FRAMES} (peak RMS {burst_peak:.4f}, threshold {VAD_RMS_THRESHOLD:.4f}).",
                    )
                    last_noise_log = now
                consecutive_voice = 0
                burst_peak = 0.0
            _metrics["frames_over"] = consecutive_voice

            if consecutive_voice >= VAD_START_FRAMES:
                speech_active = True
                _metrics["speech_active"] = True
                speech_chunks = list(pre_roll)
                pre_roll.clear()
                silence_frames = 0
                bot_was_active = _bot_active.is_set()
                if bot_was_active:
                    log_event(
                        "BARGE",
                        f"Speech confirmed: {consecutive_voice} frames ≥ {VAD_RMS_THRESHOLD:.4f} "
                        f"(peak RMS {burst_peak:.4f}, Silero {vad_prob:.2f}). Bot was active -> playback "
                        "interrupted and in-flight reply cancelled.",
                    )
                else:
                    log_event(
                        "BARGE",
                        f"Speech confirmed: {consecutive_voice} frames ≥ {VAD_RMS_THRESHOLD:.4f} "
                        f"(peak RMS {burst_peak:.4f}, Silero {vad_prob:.2f}). Bot idle -> nothing to interrupt.",
                    )
                stop_playback()  # Barge-in: stop speech and cancel any in-flight reply.
                set_voice_status("Customer speaking — AI audio interrupted.")
                burst_peak = 0.0
            continue

        speech_chunks.append(block)
        silence_frames = 0 if voiced else silence_frames + 1
        if silence_frames >= end_silence_frames:
            utterance = speech_chunks
            speech_chunks = []
            speech_active = False
            _metrics["speech_active"] = False
            consecutive_voice = 0
            silence_frames = 0
            pre_roll = []
            if silero_on:
                gate.reset()  # reset Silero's recurrent state between utterances
            duration = sum(len(part) for part in utterance) / float(_input_sample_rate)
            if duration >= VAD_MIN_UTTERANCE_SECONDS:
                _metrics["utterances"] += 1
                log_event("SYSTEM", f"Caller utterance captured ({duration:.2f}s); queued for transcription.")
                try:
                    _turn_queue.put_nowait(utterance)
                except queue.Full:
                    log_event("SYSTEM", "Turn queue full; dropped the oldest utterance.")
                    # Keep the newest utterance if the worker is backed up.
                    try:
                        _turn_queue.get_nowait()
                        _turn_queue.put_nowait(utterance)
                    except (queue.Empty, queue.Full):
                        pass
            else:
                log_event(
                    "BARGE",
                    f"Utterance discarded: {duration:.2f}s is shorter than {VAD_MIN_UTTERANCE_SECONDS:.2f}s minimum.",
                )


def customer_turn_worker():
    """Processes complete utterances while the independent monitor keeps listening."""
    global _voice_history, _first_intro_played
    while _audio_running.is_set():
        try:
            utterance = _turn_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        if not _audio_running.is_set():
            break
        # Clear the previous turn's interruption flags before processing this utterance.
        # Any new speech detected during transcription/generation will set them again.
        _interrupt_playback.clear()
        _cancel_response.clear()
        audio = np.concatenate(utterance)
        stage = "transcription (Whisper)"
        _bot_active.set()
        try:
            set_voice_status("Transcribing customer...")
            t0 = time.time()
            transcript = transcribe_audio(audio, _input_sample_rate)
            log_event("SYSTEM", f"Whisper finished in {time.time() - t0:.2f}s.")
            if not transcript:
                log_event("SYSTEM", "Whisper returned no text (silence or noise); no reply generated.")
                set_voice_status("Listening for customer...")
                _bot_active.clear()
                continue

            log_event("CALLER", transcript)

            # Play the canned first-call introduction once per session, before the normal reply.
            if PLAY_FIRST_INTRODUCTION and not _first_intro_played:
                _first_intro_played = True
                set_voice_status("Greeting caller...")
                try:
                    intro_path = make_speech(FIRST_INTRODUCTION, _voice_name, apply_opener=False, cache=True)
                    if not _interrupt_playback.is_set() and _audio_running.is_set():
                        play_audio_interruptibly(intro_path)
                        log_event("BOT", FIRST_INTRODUCTION)
                except Exception as exc:
                    log_event("ERROR", f"First introduction error: {exc}")

            with _state_lock:
                # The latest transcript becomes the new user turn.
                _voice_history.append({"role": "user", "content": transcript})
                history = list(_voice_history[:-1])
                model, voice, prompt = _voice_model, _voice_name, _voice_system_prompt
            set_voice_status(f"Customer: {transcript}")

            stage = "Ollama"
            answer = ""
            t0 = time.time()
            _generation_done = threading.Event()
            if PLAY_THINKING_FILLERS:
                play_thinking_filler_if_needed(_generation_done, _audio_running, delay=THINKING_FILLER_DELAY)
            try:
                for partial in stream_ollama_reply(history, transcript, model, prompt, _cancel_response):
                    if _cancel_response.is_set() or not _audio_running.is_set():
                        break
                    answer = partial
            finally:
                _generation_done.set()
            log_event("SYSTEM", f"Ollama finished in {time.time() - t0:.2f}s (model {model}).")

            answer = ensure_natural_opener(answer)
            if not answer:
                log_event("ERROR", "Ollama returned an empty response.")
                set_voice_status("Ollama returned an empty response; listening.")
                _bot_active.clear()
                continue
            with _state_lock:
                _voice_history.append({"role": "assistant", "content": answer})
            log_event("BOT", answer)
            set_voice_status(f"Speaking: {answer}")

            stage = "text-to-speech (Fish Audio)"
            t0 = time.time()
            speech_path = make_speech(answer, voice)
            log_event("SYSTEM", f"Fish Audio synthesized audio in {time.time() - t0:.2f}s.")
            # Do not clear an interrupt that arrived during synthesis.
            if not _interrupt_playback.is_set() and _audio_running.is_set():
                stage = "playback"
                play_audio_interruptibly(speech_path)
            else:
                log_event("SYSTEM", "Skipped playback: caller spoke during synthesis.")
            set_voice_status("Listening for customer...")
        except requests.RequestException as exc:
            set_voice_status(f"Ollama connection error: {exc}")
            log_event("ERROR", f"Ollama connection error: {exc}")
        except Exception as exc:
            set_voice_status(f"Voice pipeline error ({stage}): {exc}")
            log_event("ERROR", f"Pipeline error during {stage}: {exc}")
        finally:
            _bot_active.clear()


def start_voice_mode(input_label, output_label, model, voice, system_prompt, barge_threshold, confirmation_frames):
    global _audio_stream, _monitor_thread, _turn_thread, _input_device_index, _output_device_index
    global _input_sample_rate, _voice_model, _voice_name, _voice_system_prompt, _voice_history
    global VAD_RMS_THRESHOLD, VAD_START_FRAMES, _input_device_name, _output_device_name
    global _first_intro_played
    if _audio_running.is_set():
        return "Voice mode is already running."
    input_index, output_index = parse_device_index(input_label), parse_device_index(output_label)
    if input_index is None or output_index is None:
        return "Select both an audio input and an audio output device."
    try:
        input_info = sd.query_devices(input_index)
        output_info = sd.query_devices(output_index)
        if input_info["max_input_channels"] < 1:
            return "The selected input device has no input channels."
        if output_info["max_output_channels"] < 1:
            return "The selected output device has no output channels."
        _input_device_index, _output_device_index = input_index, output_index
        _input_device_name, _output_device_name = input_info["name"], output_info["name"]
        VAD_RMS_THRESHOLD = float(barge_threshold)
        VAD_START_FRAMES = int(confirmation_frames)
        _input_sample_rate = int(round(input_info.get("default_samplerate", 48000)))
        _voice_model = (model or DEFAULT_MODEL).strip()
        _voice_name = voice or DEFAULT_VOICE
        _voice_system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        _voice_history = []
        _first_intro_played = False
        while not _audio_queue.empty():
            try:
                _audio_queue.get_nowait()
            except queue.Empty:
                break
        while not _turn_queue.empty():
            try:
                _turn_queue.get_nowait()
            except queue.Empty:
                break
        _interrupt_playback.clear()
        _cancel_response.clear()
        _bot_active.clear()
        _metrics.update(
            rms_now=0.0, vad_now=0.0, vad_true_frames=0, rms_peak_recent=0.0, rms_peak_session=0.0,
            noise_floor=0.0, speech_active=False, frames_over=0, mic_status="",
        )
        log_event("SYSTEM", "=== Voice session started ===")
        log_event(
            "SYSTEM",
            f"Input: {_input_device_name} ({_input_sample_rate} Hz) | Output: {_output_device_name} | "
            f"Model: {_voice_model} | Voice: {_voice_name}",
        )
        log_event("SYSTEM", f"Barge-in threshold {VAD_RMS_THRESHOLD:.4f}, confirmation frames {VAD_START_FRAMES}.")
        _audio_running.set()
        _audio_stream = sd.InputStream(
            device=input_index, samplerate=_input_sample_rate, channels=1,
            dtype="float32", blocksize=1024, callback=audio_input_callback,
        )
        _audio_stream.start()
        _monitor_thread = threading.Thread(target=monitor_audio_loop, daemon=True)
        _turn_thread = threading.Thread(target=customer_turn_worker, daemon=True)
        _monitor_thread.start()
        _turn_thread.start()
        set_voice_status("Listening for customer...")
        return (
            f"Voice mode running. Input: {input_info['name']} | Output: {output_info['name']}. "
            "Customer speech interrupts AI playback."
        )
    except Exception as exc:
        _audio_running.clear()
        log_event("ERROR", f"Could not start voice mode (microphone/audio device): {exc}")
        try:
            if _audio_stream:
                _audio_stream.stop()
                _audio_stream.close()
        except Exception:
            pass
        _audio_stream = None
        return f"Could not start voice mode: {exc}"


def stop_voice_mode():
    global _audio_stream
    _audio_running.clear()
    stop_playback()
    try:
        if _audio_stream is not None:
            _audio_stream.stop()
            _audio_stream.close()
    except Exception as exc:
        log_event("ERROR", f"Error closing microphone stream: {exc}")
    _audio_stream = None
    _bot_active.clear()
    _metrics["speech_active"] = False
    log_event("SYSTEM", "=== Voice session stopped ===")
    set_voice_status("Voice mode stopped.")
    return "Voice mode stopped."


def send_message(message, history, model, voice, system_prompt):
    history = list(history or [])
    message = (message or "").strip()
    if not message:
        yield "", history, None, "Enter a message first."
        return
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})
    log_event("CALLER", message, source="text")
    yield "", history, None, "Thinking locally..."
    stage = "Ollama"
    try:
        answer = ""
        for partial in stream_ollama_reply(history[:-2], message, model, system_prompt):
            answer = partial
            history[-1]["content"] = answer
            yield "", history, None, "Generating response..."
        answer = ensure_natural_opener(answer)
        history[-1]["content"] = answer
        log_event("BOT", answer, source="text")
        stage = "text-to-speech (Fish Audio)"
        speech_path = make_speech(answer, voice)
        yield "", history, speech_path, "Response ready."
    except Exception as exc:
        history[-1]["content"] = f"Error: {exc}"
        log_event("ERROR", f"Text chat error during {stage}: {exc}")
        yield "", history, None, f"Error: {exc}"


def clear_conversation():
    return "", [], None, "Conversation cleared."


def test_fish(voice):
    """Synthesize a short phrase to verify the API key, model, and voice."""
    try:
        t0 = time.time()
        path = make_speech("Testing Fish Audio.", voice, apply_opener=False)
        return f"Fish Audio OK: model {FISH_MODEL}, {os.path.getsize(path)} bytes in {time.time() - t0:.2f}s."
    except Exception as exc:
        log_event("ERROR", f"Fish Audio test failed: {exc}")
        return f"Fish Audio test failed: {exc}"


def test_ollama():
    try:
        response = requests.get(f"{ollama_native_base_url()}/api/tags", timeout=5)
        response.raise_for_status()
        names = [item.get("name", "") for item in response.json().get("models", [])]
        return "Ollama is reachable. Models: " + (", ".join(names) if names else "(none listed)")
    except Exception as exc:
        log_event("ERROR", f"Ollama connection test failed: {exc}")
        return f"Ollama connection failed: {exc}"


try:
    INPUT_CHOICES = device_choices()
    DEFAULT_INPUT_DEVICE = default_device_label("input", ["voicemeeter aux output", "voice meeter aux output"])
    OUTPUT_CHOICES = INPUT_CHOICES
    DEFAULT_OUTPUT_DEVICE = default_device_label("output", ["voicemeeter input", "voice meeter input"])
except Exception as _exc:
    INPUT_CHOICES, OUTPUT_CHOICES = [], []
    DEFAULT_INPUT_DEVICE, DEFAULT_OUTPUT_DEVICE = None, None
    log_event("ERROR", f"Could not list audio devices: {_exc}")

NO_FLASH_CSS = """
/* Stop Gradio's "updating" pulse/dim effect on auto-refreshing components */
.pending, .generating, .translucent { opacity: 1 !important; animation: none !important; transition: none !important; }
.generating { border-color: transparent !important; }
.wrap.default, .wrap.center.default, .progress-text, .eta-bar { display: none !important; }
"""

with gr.Blocks(title="Local Text + Voice AI Test", theme=gr.themes.Soft(), css=NO_FLASH_CSS) as demo:
    gr.Markdown("# Local Text + Voice AI")
    gr.Markdown(
        "Local Ollama + Fish Audio with a continuous VoiceMeeter audio path. For Vonage, the input should "
        "contain caller audio only; do not route the AI's own voice back into the customer-audio input."
    )
    with gr.Row():
        model_box = gr.Textbox(label="Ollama model", value=DEFAULT_MODEL)
        voice_box = gr.Textbox(
            label="Fish voice ID (reference_id, optional)", value=DEFAULT_VOICE,
            info="Leave empty for Fish's default voice. Copy an ID from the Fish Audio voice library.",
        )
    with gr.Row():
        fish_model_box = gr.Textbox(label="Fish Audio model", value=FISH_MODEL, info="s2.1-pro-free, s2.1-pro, s2-pro")
        fish_key_box = gr.Textbox(
            label="Fish Audio API key", type="password", value="",
            placeholder="Using FISH_API_KEY from environment" if FISH_API_KEY else "Paste your Fish Audio API key",
        )
    fish_model_box.change(set_fish_settings, inputs=[fish_key_box, fish_model_box])
    fish_key_box.change(set_fish_settings, inputs=[fish_key_box, fish_model_box])
    system_prompt_box = gr.Textbox(label="System prompt", value=DEFAULT_SYSTEM_PROMPT, lines=5)

    with gr.Tab("Voice / Vonage"):
        gr.Markdown(
            "**Barge-in enabled:** raise the RMS threshold and/or require more consecutive audio frames "
            "to prevent background noise from interrupting the AI. The sliders apply live (when you release them); the graph below shows your input volume with the threshold line, so you can tune it while watching. "
            "Turn on **Automatic barge-in threshold** to derive it from the live noise floor instead of a fixed value, enable **Silero VAD** to require human speech (not just loudness) before interrupting, and turn on **thinking fillers** to speak a short natural phrase during processing so there is no dead air. A **first-call introduction** plays when the caller's first words (e.g. 'hello') are detected."
        )
        with gr.Row():
            input_device_box = gr.Dropdown(label="Customer audio input", choices=INPUT_CHOICES, value=DEFAULT_INPUT_DEVICE, allow_custom_value=True)
            output_device_box = gr.Dropdown(label="AI speech output", choices=OUTPUT_CHOICES, value=DEFAULT_OUTPUT_DEVICE, allow_custom_value=True)
        with gr.Row():
            barge_threshold_box = gr.Slider(
                minimum=0.005, maximum=0.080, value=VAD_RMS_THRESHOLD, step=0.0025,
                label="Barge-in threshold (RMS)", info="Higher = ignores more background noise; too high may miss quiet callers."
            )
            confirmation_frames_box = gr.Slider(
                minimum=2, maximum=12, value=VAD_START_FRAMES, step=1,
                label="Speech confirmation frames", info="Higher = sound must stay above threshold longer before interrupting."
            )
        with gr.Row():
            auto_threshold_box = gr.Checkbox(
                value=False,
                label="Automatic barge-in threshold",
                info="Derive the threshold from the live noise floor (× multiplier) instead of a fixed value.",
            )
            threshold_multiplier_box = gr.Slider(
                minimum=1.5, maximum=6.0, value=THRESHOLD_MULTIPLIER, step=0.5,
                label="Auto threshold multiplier (× noise floor)",
                info="Higher = needs louder speech to interrupt. Only used when automatic threshold is on.",
                interactive=False,
            )
        with gr.Row():
            silero_vad_box = gr.Checkbox(
                value=USE_SILERO_VAD,
                label="Silero VAD speech detection",
                info="Require human speech, not just loudness, before interrupting — rejects machinery, clicks, static.",
            )
        with gr.Row():
            fillers_box = gr.Checkbox(
                value=True,
                label="Play natural 'thinking' fillers while the AI responds",
                info="Speaks a short filler (e.g. 'Hmm, let me think') during processing so there is no dead air.",
            )
        with gr.Row():
            first_intro_enable = gr.Checkbox(
                value=True,
                label="Play first-call introduction",
                info="Canned greeting played once per session when the caller's first words (e.g. 'hello') are detected.",
            )
        first_intro_box = gr.Textbox(
            label="First-call introduction text",
            value=FIRST_INTRODUCTION,
            lines=2,
            info="Spoken once at the start of the call. Only used when the toggle above is on.",
        )
        with gr.Row():
            start_voice_button = gr.Button("Start Voice Mode", variant="primary")
            stop_voice_button = gr.Button("Stop Voice Mode")
        voice_status = gr.Textbox(label="Voice status", value="Voice mode stopped.", interactive=False)

        level_meter = gr.HTML(value=render_level_meter(), label="Input volume vs. barge-in threshold")

        audio_diag_box = gr.Textbox(
            label="Audio troubleshooting (live)", value=audio_diagnostics_text(), lines=16,
            interactive=False, elem_classes=["mono"],
        )

        with gr.Tabs():
            with gr.Tab("Full transcript"):
                transcript_box = gr.Textbox(
                    label="Caller and bot turns (chronological, all turns retained)",
                    lines=16, max_lines=16, interactive=False, autoscroll=True,
                )
            with gr.Tab("Live conversation log"):
                live_box = gr.Textbox(
                    label="Caller / BOT / SYSTEM / BARGE / ERROR events as they happen",
                    lines=16, max_lines=16, interactive=False, autoscroll=True,
                )
            with gr.Tab("Barge-in diagnostics"):
                barge_box = gr.Textbox(
                    label="Noise ignored, speech confirmed, and why playback was interrupted",
                    lines=12, max_lines=12, interactive=False, autoscroll=True,
                )
            with gr.Tab("Error log"):
                error_box = gr.Textbox(
                    label="Microphone, transcription, Ollama, Fish Audio TTS, and playback errors",
                    lines=12, max_lines=12, interactive=False, autoscroll=True,
                )

        with gr.Row():
            export_button = gr.Button("Export transcript to text file", variant="secondary")
            clear_logs_button = gr.Button("Clear logs")
        export_file = gr.File(label="Exported transcript (click to download)", interactive=False)

        start_voice_button.click(
            start_voice_mode,
            inputs=[input_device_box, output_device_box, model_box, voice_box, system_prompt_box, barge_threshold_box, confirmation_frames_box],
            outputs=[voice_status],
        )
        stop_voice_button.click(stop_voice_mode, outputs=[voice_status])
        export_button.click(export_transcript, outputs=[export_file])
        clear_logs_button.click(clear_logs, outputs=[transcript_box, live_box, barge_box, error_box])

        meter_timer = gr.Timer(0.3)
        meter_timer.tick(
            render_level_meter, inputs=[barge_threshold_box], outputs=[level_meter],
            show_progress="hidden", queue=False,
        )
        barge_threshold_box.release(apply_live_settings, inputs=[barge_threshold_box, confirmation_frames_box])
        confirmation_frames_box.release(apply_live_settings, inputs=[barge_threshold_box, confirmation_frames_box])
        auto_threshold_box.change(
            set_auto_threshold,
            inputs=[auto_threshold_box],
            outputs=[barge_threshold_box, threshold_multiplier_box],
        )
        threshold_multiplier_box.release(set_threshold_multiplier, inputs=[threshold_multiplier_box])
        silero_vad_box.change(set_silero_vad, inputs=[silero_vad_box])
        fillers_box.change(set_thinking_fillers, inputs=[fillers_box])
        first_intro_enable.change(set_first_intro, inputs=[first_intro_enable, first_intro_box])
        first_intro_box.change(set_first_intro, inputs=[first_intro_enable, first_intro_box])

        refresh_timer = gr.Timer(1.0)
        refresh_timer.tick(
            refresh_views,
            outputs=[audio_diag_box, transcript_box, live_box, barge_box, error_box, voice_status],
            show_progress="hidden", queue=False,
        )

    with gr.Tab("Text Chat"):
        chatbot = gr.Chatbot(label="Conversation", type="messages", height=420, allow_tags=False)
        message_box = gr.Textbox(label="Message", placeholder="Type a message and press Enter", lines=2)
        with gr.Row():
            send_button = gr.Button("Send", variant="primary")
            clear_button = gr.Button("Clear conversation")
        audio_output = gr.Audio(label="Generated speech", type="filepath", autoplay=False)
        text_status = gr.Textbox(label="Status", value="Ready", interactive=False)
        send_button.click(send_message, inputs=[message_box, chatbot, model_box, voice_box, system_prompt_box], outputs=[message_box, chatbot, audio_output, text_status])
        message_box.submit(send_message, inputs=[message_box, chatbot, model_box, voice_box, system_prompt_box], outputs=[message_box, chatbot, audio_output, text_status])
        clear_button.click(clear_conversation, outputs=[message_box, chatbot, audio_output, text_status])

    with gr.Accordion("Diagnostics", open=False):
        test_button = gr.Button("Test Ollama connection")
        test_status = gr.Textbox(label="Ollama status", interactive=False)
        test_button.click(test_ollama, outputs=[test_status])
        fish_test_button = gr.Button("Test Fish Audio")
        fish_test_status = gr.Textbox(label="Fish Audio status", interactive=False)
        fish_test_button.click(test_fish, inputs=[voice_box], outputs=[fish_test_status])
        gr.Markdown(
            "If VoiceMeeter devices are missing, restart this app after VoiceMeeter is running. "
            "Silero VAD requires `pip install silero-vad`; if it cannot load, detection falls back "
            "to RMS-only. faster-whisper downloads its model on first use unless it is already "
            "cached. Install required packages with: `pip install faster-whisper sounddevice silero-vad`."
        )

if __name__ == "__main__":
    import warnings

    # Silence Gradio 6.0 deprecation warnings (theme/css moved to launch(); chatbot allow_tags).
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        share=False,
    )
