import logging
import time
from typing import Optional

from core.llm_client import LLMClient
from config.prompts import get_refinement_prompt
from config.settings import TEMP_REFINEMENT, MAX_TOKENS_REFINEMENT
from shared_types import ContextAnalysis

def generate_refined_translation(
    llm_client: LLMClient,
    arabic_text: str,
    initial_translation: str,
    context: ContextAnalysis
) -> str:
    """Generates a refined translation using the LLM and context."""
    if not llm_client or not llm_client.model:
        logging.warning("LLMClient not available for refinement.")
        return initial_translation + " [Refinement Simulated - No LLM]"

    prompt = get_refinement_prompt(arabic_text, initial_translation, context)
    start_time = time.time()

    refined_text = llm_client.generate_text(
        prompt,
        temperature=TEMP_REFINEMENT,
        max_output_tokens=MAX_TOKENS_REFINEMENT
    )

    end_time = time.time()
    logging.info(f"Refined translation generated in {end_time - start_time:.2f}s")

    if not refined_text:
        logging.error("Failed to generate refined translation from LLM.")
        return initial_translation + " [Refinement Error - LLM Failed]"

    return refined_text.strip() 