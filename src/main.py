from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional
import uvicorn
import traceback
import os

from services.chat_analyzer import ChatAnalyzer
from services.chat_summarizer import ChatSummarizer
from models.response_models import AnalysisResponse
from config.settings import Settings

app = FastAPI(
    title="Chat Analysis API",
    description="API para analizar los chat exportados"
)

settings = Settings()
chat_analyzer = ChatAnalyzer()
chat_summarizer = ChatSummarizer(settings.openai_api_key)

#Aqui empieza el endpoint para analizar
@app.post("/analyze")
async def analyze_chat(
    file: UploadFile = File(...),
    method: Optional[str] = "lda",
    format: Optional[str] = "json"
):
    try:
        content = await file.read()
        temp_file_path = f"temp_{file.filename}"
        
        with open(temp_file_path, "wb") as f:
            f.write(content)

        # Analizar el chat
        df = chat_analyzer.load_chat(temp_file_path)
        df_processed, keywords = chat_analyzer.cluster_messages(df, method)
        
        # Preparar respuesta JSON
        topics = {}
        for cluster in df_processed["cluster"].unique():
            df_cluster = df_processed[df_processed["cluster"] == cluster]
            topics[int(cluster)] = {
                "keywords": keywords[cluster][:10],
                "message_count": len(df_cluster),
                "messages": df_cluster[["fecha", "hora", "usuario", "mensaje_original"]]
                    .to_dict("records")
            }

        return JSONResponse({
            "status": "success",
            "topics": topics
        })

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": str(e)
            }
        )
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == "__main__":
    # Aqui se puede manejar el puerto y en host
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
