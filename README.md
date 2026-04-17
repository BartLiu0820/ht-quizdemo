# AI Assistant Web App

Current Version: v2.0.0

## Changelog

- **v2.0.0**: Major voice interaction upgrade
  - Added dynamic recording button with visual feedback (pulsing animation, expanding ring)
  - Added real-time speech-to-text (ASR) using WebSocket streaming
  - Added text-to-speech (TTS) with Qwen TTS
  - Added voice upload for non-WebRTC browsers
  - Enhanced sidebar with password protection for settings
  - Added challenge speaker button to read questions aloud
  - Added completion animations and celebration effects
  - Added challenge image upload in settings sidebar with responsive display

- **v1.1.0**: Removed `openai` package dependency to bypass local system proxy issues, switched to using `requests` with explicit proxy bypassing (`session.trust_env = False`).

- **v1.0.0**: Initial version with text chat using Flask and OpenAI compatible endpoints.

## Features

- **Text Chat**: Interact with AI using natural language
- **Voice Input**: Record speech and convert to text in real-time
- **Voice Output**: AI responses are read aloud automatically
- **Settings Panel**: Customize system prompt and initial message (password protected)
- **Responsive UI**: Clean, modern interface with visual feedback

## Tech Stack

- **Backend**: Flask, Flask-Sock (WebSocket)
- **ASR**: DashScope Paraformer Real-time ASR
- **TTS**: DashScope Qwen TTS
- **Chat**: OpenAI compatible API via requests
- **Frontend**: Vanilla JavaScript, CSS animations
