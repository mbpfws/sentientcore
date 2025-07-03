import os
from pydantic_settings import BaseSettings, SettingsConfigDict

# Determine the root directory of the project
# This assumes config.py is in 'sentient-core/core/'
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENV_FILE_PATH = os.path.join(ROOT_DIR, 'sentient-core', '.env')

class Settings(BaseSettings):
    """
    Manages application settings and API keys, loading them from environment
    variables or a .env file.
    """
    # AI Provider Keys
    GROQ_API_KEY: str
    GOOGLE_API_KEY: str
    OPENAI_API_KEY: str

    # Agent & Sandbox Keys
    E2B_API_KEY: str
    AGENTVERSE_API_KEY: str

    # Search Tool Keys
    EXA_API_KEY: str
    TAVILY_API_KEY: str

    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH,
        env_file_encoding='utf-8',
        extra='ignore' # Ignore extra fields from the environment
    )

# Instantiate the settings object
try:
    settings = Settings()
except Exception as e:
    print(f"Error loading settings: {e}")
    print(f"Please ensure a .env file exists at {ENV_FILE_PATH} and contains all required variables.")
    # In a real application, you might exit or use default values
    # For this prototype, we will print an error and continue,
    # which will likely fail later if keys are needed.
    settings = None
