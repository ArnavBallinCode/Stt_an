# Customer Service Conversation Analyzer

A real-time conversation analysis system that transcribes, analyzes sentiment, and checks compliance with customer service SOPs. This tool is particularly useful for analyzing customer service calls and ensuring agents follow proper protocols.

## Features

- **Real-time Audio Processing**: Record and analyze live conversations
- **Audio File Analysis**: Process pre-recorded conversations
- **Speech-to-Text**: Accurate transcription using OpenAI's Whisper
- **Sentiment Analysis**: Real-time emotion detection
- **SOP Compliance**: Automatic checking against customer service protocols
- **Speaker Diarization**: Identify different speakers in the conversation
- **Detailed Reporting**: Comprehensive analysis of conversation quality

## Models Used

1. **Whisper (Speech-to-Text)**
   - Model: `whisper-base`
   - Purpose: Transcribes audio to text
   - Features:
     - High accuracy transcription
     - Support for multiple languages
     - Handles various audio formats

2. **BERTweet (Sentiment Analysis)**
   - Model: `finiteautomata/bertweet-base-sentiment-analysis`
   - Purpose: Analyzes text sentiment
   - Output:
     - POS (Positive)
     - NEU (Neutral)
     - NEG (Negative)
   - Includes confidence scores

3. **Custom SOP Analyzer**
   - Purpose: Checks agent compliance with protocols
   - Analyzes:
     - Greeting protocol
     - Problem identification
     - Solution steps
     - Closing protocol
     - Prohibited phrases
     - Required information collection

## Installation

1. **Prerequisites**
   ```bash
   # Install Homebrew (if not already installed)
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

   # Install FFmpeg
   brew install ffmpeg

   # Install PortAudio
   brew install portaudio
   ```

2. **Create and Activate Virtual Environment**
   ```bash
   # Create virtual environment
   conda create -n stt_an python=3.9
   conda activate stt_an
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Live Recording Mode
```bash
python live_analysis.py
```
- Starts recording from your microphone
- Press Ctrl+C to stop recording
- Shows real-time analysis as the conversation progresses

### 2. Audio File Analysis Mode
```bash
python live_analysis.py --file "path/to/your/audio_file.m4a"
```
- Processes the specified audio file
- Shows analysis in real-time as it processes the file

### Output Format

The system provides color-coded output:
- Customer (Speaker 1): Blue
- Agent (Speaker 2): Green
- Positive sentiment: Green
- Neutral sentiment: Yellow
- Negative sentiment: Red

Example output:
```
Speaker 2 (14:30:45):
Text: Hello, how can I help you today?
Sentiment: POS (95.2%)
----------------------------------------
```

### Final Analysis Report

After processing, you'll receive:
1. **Conversation Summary**
   - Total messages per speaker
   - Sentiment distribution
   - Complete conversation flow

2. **SOP Compliance Analysis**
   - Greeting protocol score
   - Problem identification score
   - Solution steps score
   - Closing protocol score
   - Prohibited phrases used
   - Information collection status

## SOP Rules

The system checks for compliance with the following protocols:

### Greeting Protocol
- Must include a greeting
- Must use customer's name if known
- Must ask how they can help

### Problem Identification
- Must ask for specific details
- Must acknowledge concerns
- Must show empathy

### Solution Steps
- Must provide clear next steps
- Must ask for required information
- Must set clear expectations

### Closing Protocol
- Must summarize solution
- Must thank the customer
- Must ask if anything else needed

### Prohibited Phrases
- "I don't know"
- "That's not my job"
- "You'll have to call back"
- "There's nothing I can do"

### Required Information
- Order number
- Product details
- Issue description
- Contact information

## Troubleshooting

1. **Audio File Issues**
   - Ensure the file is not corrupted
   - Try converting to WAV format
   - Check file size (should be reasonable)

2. **Installation Issues**
   - Make sure all prerequisites are installed
   - Check Python version (3.9 recommended)
   - Verify all dependencies are installed

3. **Performance Issues**
   - Use smaller audio files for testing
   - Ensure sufficient system resources
   - Close other resource-intensive applications

## Contributing

Feel free to:
- Add more SOP rules
- Improve speaker detection
- Add visualization features
- Enhance sentiment analysis

## License

This project is licensed under the MIT License - see the LICENSE file for details. 