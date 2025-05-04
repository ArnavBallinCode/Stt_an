# Customer Service Conversation Analyzer - Test Module
# Copyright (C) 2024 Arnav BallinCode
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import whisper
from transformers import pipeline
import torch
import os
import re
import numpy as np
from scipy.spatial.distance import cosine

def test_whisper():
    print("Testing Whisper with Speaker Detection...")
    try:
        # Load the base model for better accuracy
        model = whisper.load_model("base")
        print("✓ Whisper model loaded successfully")
        
        # Test transcription with a small audio file
        print("\nPlease provide a test audio file path (or press Enter to use sample conversation):")
        audio_path = input().strip()
        
        if not audio_path:
            print("\nUsing sample conversation for testing...")
            # Sample conversation with speaker labels
            result = {
                "text": "Speaker 1: Hello, how can I help you today?\nSpeaker 2: I'm having issues with my product.\nSpeaker 1: I'm sorry to hear that. What seems to be the problem?\nSpeaker 2: It's not working properly and I'm really frustrated.",
                "segments": [
                    {"text": "Hello, how can I help you today?", "speaker": "Speaker 1"},
                    {"text": "I'm having issues with my product.", "speaker": "Speaker 2"},
                    {"text": "I'm sorry to hear that. What seems to be the problem?", "speaker": "Speaker 1"},
                    {"text": "It's not working properly and I'm really frustrated.", "speaker": "Speaker 2"}
                ]
            }
        else:
            # Clean up the file path
            audio_path = audio_path.strip()
            if not os.path.exists(audio_path):
                print(f"✗ Error: File not found at {audio_path}")
                return None
                
            print(f"\nProcessing audio file: {audio_path}")
            try:
                result = model.transcribe(audio_path)
            except Exception as e:
                print(f"✗ Error transcribing audio: {str(e)}")
                print("\nTroubleshooting tips:")
                print("1. Make sure the audio file is not corrupted")
                print("2. Try converting the audio to a different format (e.g., WAV)")
                print("3. Make sure the audio file is not too large")
                return None
            
            # Process Whisper output
            segments = []
            for segment in result["segments"]:
                segments.append({
                    "text": segment["text"].strip(),
                    "start": segment["start"],
                    "end": segment["end"]
                })
            
            # Improved speaker detection based on content and timing
            speakers = []
            current_speaker = "Speaker 1"
            
            for i, segment in enumerate(segments):
                # Check if this segment might be a different speaker
                if i > 0:
                    prev_segment = segments[i-1]
                    time_gap = segment["start"] - prev_segment["end"]
                    text_similarity = any(word in segment["text"].lower() for word in ["hello", "hi", "good morning", "good afternoon"])
                    
                    # Switch speakers if:
                    # 1. There's a significant time gap
                    # 2. The text contains greetings (likely a new speaker)
                    # 3. The previous text ended with a question
                    if (time_gap > 1.0 or 
                        text_similarity or 
                        prev_segment["text"].strip().endswith("?") or
                        is_likely_different_speaker(prev_segment["text"], segment["text"])):
                        current_speaker = "Speaker 2" if current_speaker == "Speaker 1" else "Speaker 1"
                
                segment["speaker"] = current_speaker
                speakers.append(current_speaker)
            
            result["segments"] = segments
        
        print("\nTranscription Result:")
        for segment in result["segments"]:
            print(f"{segment['speaker']}: {segment['text']}")
        
        return result["segments"]
    except Exception as e:
        print(f"✗ Whisper test failed: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have FFmpeg installed (run: brew install ffmpeg)")
        print("2. Ensure the audio file is in a supported format (WAV, MP3, etc.)")
        print("3. Check if the file path is correct")
        return None

def is_likely_different_speaker(text1, text2):
    """Simple heuristic to determine if two segments might be from different speakers"""
    # Check for common patterns that might indicate different speakers
    patterns = [
        (r'^[A-Z][a-z]*:', r'^[A-Z][a-z]*:'),  # Names followed by colon
        (r'^[A-Z][a-z]* says', r'^[A-Z][a-z]* says'),  # "X says" pattern
        (r'^[A-Z][a-z]* asked', r'^[A-Z][a-z]* asked'),  # "X asked" pattern
    ]
    
    for pattern1, pattern2 in patterns:
        if re.search(pattern1, text1) and re.search(pattern2, text2):
            return True
    
    return False

def analyze_emotions(text):
    try:
        # Initialize emotion analyzer with a more sophisticated model
        emotion_analyzer = pipeline(
            "text-classification",
            model="finiteautomata/bertweet-base-sentiment-analysis"
        )
        
        # Get emotion analysis
        result = emotion_analyzer(text)[0]
        
        # Map sentiment to emotions
        sentiment_to_emotions = {
            "POS": ["joy", "optimism", "trust"],
            "NEU": ["neutral", "calm", "indifference"],
            "NEG": ["anger", "sadness", "frustration"]
        }
        
        # Get the main sentiment
        main_sentiment = result['label']
        emotions = sentiment_to_emotions.get(main_sentiment, ["neutral"])
        
        return [{"label": emotion, "score": result['score']} for emotion in emotions]
    except Exception as e:
        print(f"✗ Emotion analysis failed: {str(e)}")
        return None

def analyze_sentiment(text):
    try:
        # Initialize sentiment analyzer
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="finiteautomata/bertweet-base-sentiment-analysis"
        )
        
        # Get sentiment analysis
        result = sentiment_analyzer(text)[0]
        return result
    except Exception as e:
        print(f"✗ Sentiment analysis failed: {str(e)}")
        return None

def analyze_conversation(segments):
    print("\nAnalyzing Conversation...")
    
    # Group segments by speaker
    speakers = {}
    for segment in segments:
        speaker = segment['speaker']
        if speaker not in speakers:
            speakers[speaker] = []
        speakers[speaker].append(segment['text'])
    
    # Analyze each speaker's text
    for speaker, texts in speakers.items():
        print(f"\nAnalysis for {speaker}:")
        full_text = " ".join(texts)
        
        # Emotion analysis
        emotions = analyze_emotions(full_text)
        if emotions:
            print("\nDetected Emotions:")
            for i, emotion in enumerate(emotions, 1):
                print(f"{i}. {emotion['label']}: {emotion['score']:.2%}")
        
        # Sentiment analysis
        sentiment = analyze_sentiment(full_text)
        if sentiment:
            sentiment_map = {
                "POS": "Positive",
                "NEU": "Neutral",
                "NEG": "Negative"
            }
            print(f"\nOverall Sentiment: {sentiment_map.get(sentiment['label'], sentiment['label'])}")
            print(f"Confidence: {sentiment['score']:.2%}")
        
        # Print conversation summary
        print("\nConversation Summary:")
        for text in texts:
            print(f"- {text}")

def main():
    print("Starting conversation analysis...")
    print("\nNote: For analysis, you can:")
    print("1. Provide a path to an audio file")
    print("2. Press Enter to use sample conversation")
    print("3. Type 'exit' to quit")
    
    while True:
        choice = input("\nEnter your choice (1/2/3): ").strip()
        
        if choice == '3' or choice.lower() == 'exit':
            print("Exiting...")
            return
            
        if choice in ['1', '2']:
            break
        else:
            print("Invalid choice. Please try again.")
    
    segments = test_whisper()
    if segments:
        analyze_conversation(segments)
    else:
        print("\nFailed to analyze conversation. Please check the error messages above.")

if __name__ == "__main__":
    main() 