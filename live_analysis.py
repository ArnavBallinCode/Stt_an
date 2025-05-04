# Customer Service Conversation Analyzer
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

import pyaudio
import wave
import whisper
import numpy as np
import threading
import queue
import time
from transformers import pipeline
import torch
import os
from sop_analyzer import SOPAnalyzer
import argparse

class LiveConversationAnalyzer:
    def __init__(self):
        # Audio settings
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = 8000  # 0.5 seconds of audio
        self.RECORD_SECONDS = 5  # Process in 5-second chunks
        
        # Initialize Whisper
        print("Loading Whisper model...")
        self.whisper_model = whisper.load_model("base")
        
        # Initialize sentiment analyzer
        print("Loading sentiment analyzer...")
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="finiteautomata/bertweet-base-sentiment-analysis"
        )
        
        # Initialize SOP analyzer
        print("Loading SOP analyzer...")
        self.sop_analyzer = SOPAnalyzer()
        
        # Audio processing objects
        self.audio = pyaudio.PyAudio()
        self.frames_queue = queue.Queue()
        self.is_recording = False
        
        # Analysis results
        self.current_speaker = "Speaker 1"
        self.conversation_history = []
    
    def analyze_audio_file(self, audio_path):
        """Analyze an existing audio file"""
        if not os.path.exists(audio_path):
            print(f"Error: File not found at {audio_path}")
            return
        
        print(f"\nProcessing audio file: {audio_path}")
        try:
            # Transcribe the entire audio file
            result = self.whisper_model.transcribe(audio_path)
            
            # Process each segment
            for segment in result["segments"]:
                text = segment["text"].strip()
                if text:
                    # Detect speaker changes
                    if self._should_switch_speaker(text):
                        self.current_speaker = "Speaker 2" if self.current_speaker == "Speaker 1" else "Speaker 1"
                    
                    # Analyze sentiment
                    sentiment = self.sentiment_analyzer(text)[0]
                    
                    # Analyze against SOP
                    self.sop_analyzer.analyze_message(text, self.current_speaker)
                    
                    # Store results
                    analysis = {
                        "speaker": self.current_speaker,
                        "text": text,
                        "sentiment": sentiment["label"],
                        "confidence": sentiment["score"],
                        "timestamp": time.strftime("%H:%M:%S")
                    }
                    self.conversation_history.append(analysis)
                    
                    # Print analysis
                    self._print_live_analysis(analysis)
            
            # Print final analysis
            self._print_final_analysis()
            
        except Exception as e:
            print(f"Error processing audio file: {str(e)}")
            print("\nTroubleshooting tips:")
            print("1. Make sure the audio file is not corrupted")
            print("2. Try converting the audio to a different format (e.g., WAV)")
            print("3. Make sure the audio file is not too large")
    
    def start_recording(self):
        """Start recording audio"""
        self.is_recording = True
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.analysis_thread = threading.Thread(target=self._process_audio)
        
        print("\nStarting live conversation analysis...")
        print("Press Ctrl+C to stop recording\n")
        
        self.recording_thread.start()
        self.analysis_thread.start()
    
    def stop_recording(self):
        """Stop recording audio"""
        self.is_recording = False
        self.recording_thread.join()
        self.analysis_thread.join()
        self.audio.terminate()
        
        # Print final analysis
        self._print_final_analysis()
    
    def _record_audio(self):
        """Record audio in chunks"""
        stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
        
        while self.is_recording:
            frames = []
            for _ in range(int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
                if not self.is_recording:
                    break
                data = stream.read(self.CHUNK)
                frames.append(data)
            
            if frames:
                self.frames_queue.put(frames)
        
        stream.stop_stream()
        stream.close()
    
    def _process_audio(self):
        """Process audio chunks and perform analysis"""
        while self.is_recording or not self.frames_queue.empty():
            if not self.frames_queue.empty():
                frames = self.frames_queue.get()
                
                # Convert frames to numpy array
                audio_data = np.frombuffer(b''.join(frames), dtype=np.float32)
                
                # Transcribe audio
                result = self.whisper_model.transcribe(audio_data)
                
                if result["text"].strip():
                    # Detect speaker changes
                    text = result["text"].strip()
                    if self._should_switch_speaker(text):
                        self.current_speaker = "Speaker 2" if self.current_speaker == "Speaker 1" else "Speaker 1"
                    
                    # Analyze sentiment
                    sentiment = self.sentiment_analyzer(text)[0]
                    
                    # Analyze against SOP
                    self.sop_analyzer.analyze_message(text, self.current_speaker)
                    
                    # Store results
                    analysis = {
                        "speaker": self.current_speaker,
                        "text": text,
                        "sentiment": sentiment["label"],
                        "confidence": sentiment["score"],
                        "timestamp": time.strftime("%H:%M:%S")
                    }
                    self.conversation_history.append(analysis)
                    
                    # Print live analysis
                    self._print_live_analysis(analysis)
            
            time.sleep(0.1)
    
    def _should_switch_speaker(self, text):
        """Determine if we should switch speakers based on content"""
        if not self.conversation_history:
            return False
        
        # Check for patterns that might indicate a speaker change
        text_lower = text.lower()
        greeting_words = ["hello", "hi", "good morning", "good afternoon"]
        question_response = self.conversation_history[-1]["text"].strip().endswith("?")
        contains_greeting = any(word in text_lower for word in greeting_words)
        
        return question_response or contains_greeting
    
    def _print_live_analysis(self, analysis):
        """Print real-time analysis"""
        speaker_color = "\033[94m" if analysis["speaker"] == "Speaker 1" else "\033[92m"
        sentiment_color = {
            "POS": "\033[92m",  # Green
            "NEU": "\033[93m",  # Yellow
            "NEG": "\033[91m"   # Red
        }.get(analysis["sentiment"], "\033[0m")
        
        print(f"\n{speaker_color}{analysis['speaker']} ({analysis['timestamp']}):\033[0m")
        print(f"Text: {analysis['text']}")
        print(f"Sentiment: {sentiment_color}{analysis['sentiment']}\033[0m ({analysis['confidence']:.2%})")
        print("-" * 50)
    
    def _print_final_analysis(self):
        """Print final conversation analysis"""
        print("\n=== Final Conversation Analysis ===")
        
        # Analyze per speaker
        for speaker in ["Speaker 1", "Speaker 2"]:
            speaker_messages = [msg for msg in self.conversation_history if msg["speaker"] == speaker]
            if speaker_messages:
                print(f"\n{speaker}:")
                print(f"Total messages: {len(speaker_messages)}")
                
                # Calculate average sentiment
                sentiments = [msg["sentiment"] for msg in speaker_messages]
                pos_count = sentiments.count("POS")
                neu_count = sentiments.count("NEU")
                neg_count = sentiments.count("NEG")
                
                print("Sentiment distribution:")
                print(f"- Positive: {pos_count/len(sentiments):.1%}")
                print(f"- Neutral: {neu_count/len(sentiments):.1%}")
                print(f"- Negative: {neg_count/len(sentiments):.1%}")
                
                print("\nConversation flow:")
                for msg in speaker_messages:
                    print(f"[{msg['timestamp']}] {msg['text']}")
        
        # Print SOP compliance analysis
        print("\n" + self.sop_analyzer.get_analysis_report())

def main():
    parser = argparse.ArgumentParser(description='Analyze customer service conversations')
    parser.add_argument('--file', type=str, help='Path to audio file to analyze', nargs='+')
    args = parser.parse_args()
    
    analyzer = LiveConversationAnalyzer()
    
    if args.file:
        # Join the file path parts in case there are spaces
        audio_path = ' '.join(args.file)
        # Analyze audio file
        analyzer.analyze_audio_file(audio_path)
    else:
        # Start live recording
        try:
            analyzer.start_recording()
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping recording...")
            analyzer.stop_recording()

if __name__ == "__main__":
    main() 