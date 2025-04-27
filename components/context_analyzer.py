import logging
import time
from typing import Optional

from core.llm_client import LLMClient
from config.prompts import get_context_analysis_prompt
from config.settings import TEMP_CONTEXT_ANALYSIS, MAX_TOKENS_CONTEXT_ANALYSIS
from shared_types import ContextAnalysis
from utils.helpers import extract_json

# Define the default context analysis structure here or import from a dedicated defaults file
DEFAULT_CONTEXT_ANALYSIS = ContextAnalysis(
    genre='Classical Arabic Prose',
    timePeriod='Classical Era (8th-13th centuries CE)',
    tone='Formal, reflective, scholarly',
    keyTerminology=[
        'كتب التاريخ (Books of history)',
        'الحروب (Wars)',
        'أهل التدبير (Men of administration/statesmanship/strategy)',
        'رؤساءها (Their leaders)',
        'مشايخ الحرب (Veterans/experts of war)',
        'علمائها (Scholars/experts of war)',
    ],
    historicalContext='Default context: Appears scholarly, possibly historical.',
    generatedTime=0,
)

async def generate_context_analysis(llm_client: LLMClient, arabic_text: str) -> ContextAnalysis:
    """Performs dynamic context analysis using the LLM."""
    if not llm_client or not llm_client.model:
        logging.warning("LLMClient not available for context analysis. Using default.")
        return DEFAULT_CONTEXT_ANALYSIS

    prompt = get_context_analysis_prompt(arabic_text)
    start_time = time.time()

    response_text = await llm_client.generate_text(
        prompt,
        temperature=TEMP_CONTEXT_ANALYSIS,
        max_output_tokens=MAX_TOKENS_CONTEXT_ANALYSIS
    )

    end_time = time.time()
    generated_time_ms = int((end_time - start_time) * 1000)

    if not response_text:
        logging.warning("Failed to get context analysis from LLM. Using default.")
        return DEFAULT_CONTEXT_ANALYSIS

    context_json = extract_json(response_text)
    if not context_json or not isinstance(context_json, dict):
        logging.warning("Failed to parse context analysis JSON from LLM response. Using default.")
        return DEFAULT_CONTEXT_ANALYSIS

    # Basic validation
    required_keys = ["genre", "timePeriod", "tone", "keyTerminology", "historicalContext"]
    if not all(key in context_json for key in required_keys):
         logging.warning(f"Context analysis JSON missing required keys ({required_keys}). Using default. JSON: {context_json}")
         return DEFAULT_CONTEXT_ANALYSIS
    if not isinstance(context_json.get("keyTerminology"), list):
         logging.warning("Context analysis 'keyTerminology' is not a list. Using default terms.")
         context_json["keyTerminology"] = DEFAULT_CONTEXT_ANALYSIS.keyTerminology

    try:
        return ContextAnalysis(
            genre=str(context_json.get('genre', DEFAULT_CONTEXT_ANALYSIS.genre)),
            timePeriod=str(context_json.get('timePeriod', DEFAULT_CONTEXT_ANALYSIS.timePeriod)),
            tone=str(context_json.get('tone', DEFAULT_CONTEXT_ANALYSIS.tone)),
            keyTerminology=list(context_json.get('keyTerminology', DEFAULT_CONTEXT_ANALYSIS.keyTerminology)),
            historicalContext=str(context_json.get('historicalContext', DEFAULT_CONTEXT_ANALYSIS.historicalContext)),
            generatedTime=generated_time_ms
        )
    except Exception as e:
        logging.error(f"Error creating ContextAnalysis object from JSON: {e}. JSON: {context_json}")
        return DEFAULT_CONTEXT_ANALYSIS 