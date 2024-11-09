from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str = "your-api-key-here"
    
    class Config:
        env_file = ".env"