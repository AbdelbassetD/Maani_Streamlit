import logging
import time
from typing import Optional

from core.llm_client import LLMClient
from config.prompts import get_initial_translation_prompt
from config.settings import TEMP_INITIAL_TRANSLATION, MAX_TOKENS_INITIAL_TRANSLATION

def generate_initial_translation(llm_client: LLMClient, arabic_text: str) -> Optional[str]:
    """Generates a basic initial translation using the LLM."""
    if not llm_client or not llm_client.model:
        logging.warning("LLMClient not available for initial translation.")
        # Basic simulation fallback if no LLM
        return "[Simulation] Praise be to Allah." if 'الحمد لله' in arabic_text else "[Simulation] Initial translation unavailable."

    prompt = get_initial_translation_prompt(arabic_text)
    start_time = time.time()

    translation = llm_client.generate_text(
        prompt,
        temperature=TEMP_INITIAL_TRANSLATION,
        max_output_tokens=MAX_TOKENS_INITIAL_TRANSLATION
    )

    end_time = time.time()
    logging.info(f"Initial translation generated in {end_time - start_time:.2f}s")

    if not translation:
        logging.error("Failed to generate initial translation from LLM.")
        return "Error: Could not generate initial translation."

    return translation.strip() 