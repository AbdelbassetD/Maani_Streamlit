import os
import logging
from typing import Optional

import google.generativeai as genai
from google.generativeai.types import GenerationConfig, GenerateContentResponse
from dotenv import load_dotenv

# Assuming config/settings.py is accessible
from config.settings import DEFAULT_MODEL_NAME, SAFETY_SETTINGS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LLMClient:
    """Handles interaction with the Google Generative AI API."""

    def __init__(self, model_name: Optional[str] = None):
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name or DEFAULT_MODEL_NAME
        self.model = None

        if not self.api_key:
            logging.warning("GOOGLE_API_KEY environment variable not set. LLMClient will not function.")
            return

        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(
                self.model_name,
                safety_settings=SAFETY_SETTINGS
            )
            logging.info(f"Initialized Google Generative AI model: {self.model_name}")
        except Exception as e:
            logging.error(f"Failed to initialize Google Generative AI model shell: {e}")
            self.model = None # Ensure model is None if initialization fails

    def generate_text(self, prompt: str, temperature: float = 0.2, max_output_tokens: Optional[int] = None) -> Optional[str]:
        """Calls the Google Generative AI API synchronously to generate text."""
        if not self.model:
            logging.warning("Google AI model shell not initialized. Skipping API call.")
            return None

        generation_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens
        )

        try:
            # Use synchronous generate_content
            response: GenerateContentResponse = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )

            # Accessing text safely using response.text getter
            # The .text getter handles potential errors/blocks internally in recent versions
            # However, checking parts can still be useful for debugging
            if not response.parts:
                 # Handle potential blocks or empty responses explicitly if needed
                finish_reason = response.candidates[0].finish_reason if response.candidates else 'UNKNOWN'
                safety_ratings = response.candidates[0].safety_ratings if response.candidates else None
                logging.warning(f"Google API response has no parts. Finish reason: {finish_reason}")
                if safety_ratings:
                    logging.warning(f"Safety Ratings: {safety_ratings}")
                return None # Indicate failure or empty response

            return response.text # Safest way to get text output

        except Exception as e:
            logging.error(f"Error calling Google API (model: {self.model_name}): {e}")
            # Check for specific API errors if possible (e.g., quota, key invalid)
            if "API key not valid" in str(e):
                 logging.error("Google API key is invalid. Please check your .env file.")
                 # Consider disabling the client if the key is definitively bad
                 # self.model = None
            # Add more specific error handling here if needed (e.g., ResourceExhaustedError)
            return None 