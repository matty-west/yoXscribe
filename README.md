# Application use-case:
- When capturing audio of multiple speakers with one microphone, subsequent transcripts will not specify the speaker.
- This application will generate a speaker-specific, annotated transcript of an audio file with multiple speakers. 

1. The application will use OpenAI's Whisper model generate a .tsv transcript
2. The .tsv transcript can used for global conversation anaylsis
3. Further processing with pyannote.audio speaker diarization toolkit will annotate individual speakers
4. Annotated tsv can be used for high-resolution conversation analysis

