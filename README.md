# AI Assistant Web App
Current Version: v1.1.0

## Changelog
- **v1.1.0**: Removed `openai` package dependency to bypass local system proxy issues, switched to using `requests` with explicit proxy bypassing (`session.trust_env = False`).
- **v1.0.0**: Initial version with text chat using Flask and OpenAI compatible endpoints.

## Upcoming Features
- Voice input capability using Web Speech API (SpeechRecognition).
