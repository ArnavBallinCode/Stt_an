import os
import whisper
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import tempfile
from typing import Dict, Any
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Whisper model (using the 'tiny' model for better performance on 16GB RAM)
whisper_model = whisper.load_model("tiny")

# Initialize lightweight sentiment analysis model
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="finiteautomata/bertweet-base-sentiment-analysis",  # More lightweight model
    device=0 if torch.cuda.is_available() else -1
)

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Transcribe audio file to text using Whisper
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Transcribe audio with memory optimization
        with torch.cuda.amp.autocast():  # Use mixed precision for better memory usage
            result = whisper_model.transcribe(temp_file_path)
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        return {
            "text": result["text"],
            "language": result["language"]
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/analyze")
async def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyze sentiment of text using lightweight BERT model
    """
    try:
        # Get sentiment analysis with memory optimization
        with torch.cuda.amp.autocast():
            result = sentiment_analyzer(text)[0]
        
        # Map sentiment to a 1-5 scale
        sentiment_map = {
            "POS": 5,
            "NEU": 3,
            "NEG": 1
        }
        
        return {
            "sentiment": sentiment_map.get(result['label'], 3),
            "confidence": result['score']
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/transcribe-and-analyze")
async def transcribe_and_analyze(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Transcribe audio and analyze sentiment in one step
    """
    try:
        # First transcribe the audio
        transcription = await transcribe_audio(file)
        if "error" in transcription:
            return transcription
            
        # Then analyze the sentiment
        analysis = await analyze_sentiment(transcription["text"])
        if "error" in analysis:
            return analysis
            
        return {
            "text": transcription["text"],
            "language": transcription["language"],
            "sentiment": analysis["sentiment"],
            "confidence": analysis["confidence"]
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 