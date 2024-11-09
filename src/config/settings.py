from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str = "AQUI_INSERTAR_EL_API_KEY"
    
    class Config:
        env_file = ".env"