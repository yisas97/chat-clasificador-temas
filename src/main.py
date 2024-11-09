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
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_chat(
    file: UploadFile = File(...),
    method: Optional[str] = "lda",
    generate_summary: Optional[bool] = False
):
    try:
        if method not in ["lda", "kmeans"]:
            raise ValueError(f"Invalid method: {method}. Must be 'lda' or 'kmeans'")
        
        # Guardar de manera temporal
        content = await file.read()
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as f:
            f.write(content)

        # Procesar CHAT y extraer para exportar
        zip_file_path = chat_analyzer.analyze(temp_file_path, method)
        
        # Generate summaries if requested
        if generate_summary:
            # Aquí falta la chicha del resumen con la IA
            pass
        
        # Retornar el archivo ZIP
        return FileResponse(
            path=zip_file_path,
            filename=os.path.basename(zip_file_path),
            media_type='application/zip'
        )

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

    finally:
        # Eliminar el archivo temporal
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == "__main__":
    # Aqui se puede manejar el puerto y en host
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
