from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import uvicorn
import traceback

from services.chat_analyzer import ChatAnalyzer
from services.chat_summarizer import ChatSummarizer
from models.response_models import AnalysisResponse
from config.settings import Settings

app = FastAPI(
    title="Chat Analysis API",
    description="API for analyzing WhatsApp chat exports"
)

settings = Settings()
chat_analyzer = ChatAnalyzer()
chat_summarizer = ChatSummarizer(settings.openai_api_key)

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_chat(
    file: UploadFile = File(...),
    method: Optional[str] = "lda",
    generate_summary: Optional[bool] = False
):
    try:
        if method not in ["lda", "kmeans"]:
            raise ValueError(f"Invalid method: {method}. Must be 'lda' or 'kmeans'")
        
        # Save uploaded file temporarily
        content = await file.read()
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as f:
            f.write(content)

        # Process chat
        results_dir = chat_analyzer.analyze(temp_file_path, method)
        
        # Generate summaries if requested
        summaries = None
        if generate_summary:
            summaries = chat_summarizer.generate_summaries(results_dir)

        return {
            "status": "success",
            "results_directory": results_dir,
            "summaries": summaries
        }

    except Exception as e:
        error_msg = f"Error: {str(e)}\nTraceback: {traceback.format_exc()}"
        print(error_msg)  # Para logging
        raise HTTPException(
            status_code=500,
                detail={
                        "status": "error",
                        "message": str(e),
                        "traceback": traceback.format_exc()
                }
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
