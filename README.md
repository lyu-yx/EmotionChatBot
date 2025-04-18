# Voice-Interactive Chatbot System

A modular voice-interactive chatbot capable of real-time speech-to-text and text-to-speech communication with a language model backend.

## Features

- **Real-time Voice Interaction**: Speak naturally to the chatbot and hear spoken responses
- **Automatic Speech Recognition (ASR)**: Convert spoken language to text
- **Natural Language Processing**: Process user queries using a Large Language Model
- **Text-to-Speech (TTS)**: Convert text responses to natural-sounding speech
- **Modular Architecture**: Easily extend with new capabilities
- **Configurable**: Adjust settings via configuration files

## System Requirements

- Python 3.8+
- PyAudio dependencies (Windows: Visual C++ Build Tools, Linux: portaudio19-dev)
- **ffmpeg** installed and available in PATH (required for audio streaming)
- Alibaba Cloud account and API credentials (for the default implementations)
- Alternatively, an OpenAI API key (if you prefer to use OpenAI instead of Alibaba)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/lyu-yx/EmotionChatBot.git
cd EmotionChatBot
```

### 2. Set up the Conda environment

```bash
conda create -n emotion_chatbot python=3.10 -y
conda activate emotion_chatbot
```

### 3. Install the required packages

```bash
pip install -r requirements.txt
```

### 4. Install ffmpeg

ffmpeg is required for real-time audio streaming. Install it according to your operating system:

#### Windows
- Download from [ffmpeg.org](https://ffmpeg.org/download.html) or use chocolatey: `choco install ffmpeg`
- Add ffmpeg to your PATH environment variable

#### macOS
```bash
brew install ffmpeg
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install ffmpeg
```

### 5. Set up environment variables

Create a `.env` file in the project root directory:

```
# Alibaba Cloud API Key
ALIBABA_API_KEY=your_api_key_here

# Optional: OpenAI API Key (if using OpenAI instead of Alibaba)
# OPENAI_API_KEY=your_api_key_here
```

## Configuration

The system can be configured using the `config.json` file in the project root. 
The available configuration options include:

- **ASR Settings**: Engine, language, timeout, energy threshold
- **TTS Settings**: Engine, voice, speech rate, volume
- **LLM Settings**: Provider, model, temperature, system prompt
- **System Settings**: Debug mode, exit phrase

Example configuration:

```json
{
  "system": {
    "debug": false,
    "exit_phrase": "exit"
  },
  "asr": {
    "engine": "alibaba",
    "language": "zh-cn",
    "timeout": 5
  },
  "tts": {
    "engine": "alibaba",
    "voice": "xiaoyun",
    "speech_rate": 0,
    "volume": 50
  },
  "llm": {
    "provider": "alibaba",
    "model": "qwen-turbo",
    "temperature": 0.7
  }
}
```

## Usage

### Basic usage

```bash
python src/main.py
```

### With custom configuration file

```bash
python src/main.py --config path/to/custom_config.json
```

### With custom exit phrase

```bash
# Run with default settings (Chinese language)
python -m src.main

# Run with English language
python -m src.main --language en-us

# Run with a specific voice
python -m src.main --voice xiaomo

# Disable emotion detection
python -m src.main --no-emotion

# Use a different exit phrase
python -m src.main --exit-phrase "goodbye"
```

## Architecture

The system is designed with a modular architecture for extensibility:

- **ASR Module**: Converts speech to text using different recognition engines
- **TTS Module**: Converts text to speech using various synthesis engines
- **LLM Module**: Processes text queries and generates responses
- **Core Module**: Coordinates the interaction between components
- **Emotion Module**: (Future extension) Will provide emotion detection capabilities

### Extending the System

The modular design allows for easy extension with new capabilities:

1. **Add New ASR Engine**: Implement the `SpeechRecognizer` interface
2. **Add New TTS Engine**: Implement the `TextToSpeech` interface 
3. **Add New LLM Integration**: Implement the `LanguageModel` interface
4. **Add Emotion Recognition**: Implement the `EmotionDetector` interface

### Available Service Implementations

The system comes with the following implementations:

#### ASR Engines:
- **AlibabaSpeechRecognizer**: Uses Alibaba Cloud's Speech Recognition service
- **GoogleSpeechRecognizer**: Uses Google's Web Speech API

#### TTS Engines:
- **AlibabaSpeechSynthesizer**: Uses Alibaba Cloud's Speech Synthesis service
- **PyttsxSpeechSynthesizer**: Uses local pyttsx3 engine

#### LLM Providers:
- **AlibabaLanguageModel**: Uses Alibaba Cloud's Tongyi Qianwen LLM
- **OpenAILanguageModel**: Uses OpenAI's GPT models

## Future Enhancements

- **Emotion Recognition**: Detect emotions from voice and text
- **Adaptive Responses**: Tailor responses based on detected emotions
- **Multiple Language Support**: Add support for languages beyond English
- **Offline Mode**: Local speech recognition and synthesis
- **Whisper Integration**: Enhanced speech recognition accuracy

## License

MIT

## Extra state

This work account heavily on LLM code generation tools.


## Contributors

- Yixuan - Initial work