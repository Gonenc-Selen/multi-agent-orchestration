from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    google_cloud_project: str = ""
    google_cloud_location: str = "us-central1"
    gemini_model: str = "gemini-2.5-pro"
    random_seed: int = 42

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
