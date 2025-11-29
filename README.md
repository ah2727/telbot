# Doctor Voice Assistant

A local microphone-based doctor reservation assistant that relies on OpenAI for speech-to-text and conversation. The agent converses in Persian, collects the caller's name and address, confirms them, and stores the captured data in `data/last_session.json` for later use.

## Setup

1. **Python environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **PortAudio dependency** – `sounddevice` uses PortAudio. On macOS use `brew install portaudio`; on Linux install `portaudio19-dev`.
3. **Voices for pyttsx3** – macOS/Windows ship with voices. Linux may require `espeak`.
4. **Environment variables** – Copy `.env.example` to `.env` and keep the provided OpenAI key or set your own `OPENAI_API_KEY`. Optional overrides:
   - `OPENAI_RESPONSE_MODEL` (default `gpt-4o-mini`) for the reasoning model.
   - `OPENAI_TRANSCRIBE_MODEL` (default `gpt-4o-mini-transcribe`) for speech-to-text. If you set this to an advanced multimodal model, the bot sends the audio as base64 text within the prompt and instructs that model to decode/transcribe it; on failure, it falls back to `OPENAI_TRANSCRIBE_FALLBACK`.
   - `OPENAI_TRANSCRIBE_FALLBACK` (default `gpt-4o-mini-transcribe`) used whenever the chosen transcription model cannot accept raw audio.
   - `OPENAI_TTS_MODEL` / `OPENAI_TTS_VOICE` for text-to-speech selection.

## Running the bot

```bash
python voice_bot.py
```

- Press `Enter` to capture up to 8 seconds of audio (or pass `--record-seconds` to adjust).
- Speak naturally; recording auto-stops a fraction of a second after you fall silent, so responses arrive quickly. The configured transcription model handles speech-to-text (the bot automatically embeds the audio as base64 text when targeting multimodal models, and will warn only if it must fall back to the Whisper model).
- The assistant reasons with the model in `OPENAI_RESPONSE_MODEL` (defaults to `gpt-4o-mini`) and always replies in Persian JSON that includes the latest name/address if available.
- Replies are spoken aloud through `pyttsx3` and logged to `data/session_log.txt`.
- Updated profile data is persisted to `data/last_session.json` (with the active session name) so you can see the captured name/address record after each conversation.

### Sessions & client recall

- Every run starts a new session ID like `drx-YYYYMMDD-HHMMSS-xxxx`. The latest ID is stored in `data/session_meta.json`, and each log entry is tagged with `[session-id]` for easy traceability.
- Known caller names are saved to `data/clients.json`. When a transcript mentions a familiar name, the assistant is nudged to greet them as a returning caller and confirm their details politely.
- `data/last_session.json` is kept only as reference—each new session starts with empty contact info, and the assistant is explicitly told to reconfirm any prior names or addresses before reusing them.
- Keep the session ID handy if you need to archive or reference a specific call.

## Training the prompt from voice

Record a new system prompt just by speaking:

```bash
python voice_bot.py --train-prompt
```

- Speak for up to `--record-seconds` (default 8 seconds) to describe how the assistant should behave.
- The transcription is saved to `data/custom_prompt.txt`, and future sessions automatically use it instead of the default built-in prompt.

## Realtime mode

Hands-free listening is also supported:

```bash
python voice_bot.py --realtime
```

- The bot listens continuously, detects speech using audio energy, and auto-sends utterances once you pause; every response is spoken in Persian.
- Use `--chunk-seconds`, `--silence-timeout`, and `--energy-threshold` to fine-tune how quickly it reacts after you finish speaking (defaults are tuned for snappy handoffs).
- Pause for roughly a second to let the assistant answer out loud.
- Press `Ctrl+C` to end the session. Conversation logs/profile storage work the same as push-to-talk mode.

## Customization

- Change voices: `pyttsx3` lets you pick any installed voice—call `self.tts_engine.setProperty("voice", voice_id)` in `VoiceDoctorBot.__init__`.
- Longer/shorter utterances: adjust `record_seconds`.
- Additional structured fields: edit `data/custom_prompt.txt` (after running `--train-prompt`) or the built-in `SYSTEM_PROMPT` to include more keys in the JSON payload.
- Update `data/custom_prompt.txt` to tweak the ManaCare/DrX persona or to adjust how returning clients and session IDs should be mentioned.

## Safety

Handle real user data carefully—`data/last_session.json` contains PII. Delete the file when no longer needed.
