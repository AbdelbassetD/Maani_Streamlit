import logging
import time
from typing import List, Optional
import streamlit as st

from core.llm_client import LLMClient
from config.prompts import get_linguistic_nuance_prompt
from config.settings import TEMP_NUANCE_ANALYSIS, MAX_TOKENS_NUANCE_ANALYSIS
from shared_types import ContextAnalysis, LinguisticNuance, TextLocation
from utils.helpers import extract_json, find_best_match

# Define fallback nuances
FALLBACK_LINGUISTIC_NUANCES = [
    LinguisticNuance(
        text="Fallback Nuance 1",
        explanation='Fallback: Phrase cultural significance.',
        category='cultural-reference',
        targetLocation=TextLocation(start=0, end=20),
    ),
    LinguisticNuance(
        text="Fallback Nuance 2",
        explanation='Fallback: Formal rhetorical style.',
        category='formality',
        targetLocation=TextLocation(start=50, end=70),
    ),
]

def generate_linguistic_nuances(
    llm_client: LLMClient,
    arabic_text: str, # Keep arabic_text in case prompts need it later
    refined_translation: str,
    context: ContextAnalysis
) -> List[LinguisticNuance]:
    """Generates linguistic nuance explanations using the LLM."""
    if not llm_client or not llm_client.model:
        logging.warning("LLMClient not available for linguistic nuances. Using fallback.")
        return _get_adjusted_fallback_nuances(refined_translation)

    # Return empty list if refined translation is empty or too short
    if not refined_translation or len(refined_translation) < 10:
        logging.info("Refined translation too short for nuance analysis.")
        return []

    prompt = get_linguistic_nuance_prompt(arabic_text, refined_translation, context)
    start_time = time.time()

    response_text = llm_client.generate_text(
        prompt,
        temperature=TEMP_NUANCE_ANALYSIS,
        max_output_tokens=MAX_TOKENS_NUANCE_ANALYSIS
    )

    end_time = time.time()
    logging.info(f"Linguistic nuance analysis generated in {end_time - start_time:.2f}s")

    # --- START DEBUGGING --- #
    st.text(f"DEBUG: LLM Raw Response (Nuance Length): {len(response_text) if response_text else 0}")
    # --- END DEBUGGING --- #

    if not response_text:
        logging.warning("Failed to get linguistic nuances from LLM. Using fallback.")
        return _get_adjusted_fallback_nuances(refined_translation)

    nuances_json = extract_json(response_text)
    # --- START DEBUGGING --- #
    st.text(f"DEBUG: Extracted JSON Data (Nuance): {nuances_json}")
    # --- END DEBUGGING --- #

    if not nuances_json or not isinstance(nuances_json, list):
        logging.warning(f"Failed to parse linguistic nuances JSON or not a list. Using fallback. Raw: {response_text[:200]}")
        return _get_adjusted_fallback_nuances(refined_translation)

    processed_nuances: List[LinguisticNuance] = []
    try:
        st.text(f"DEBUG: Starting Nuance loop. JSON List Length: {len(nuances_json)}")
        for i, nuance_data in enumerate(nuances_json):
            st.text(f"DEBUG: Processing Nuance {i}: {nuance_data}")
            if not isinstance(nuance_data, dict) or not all(k in nuance_data for k in ['text', 'explanation', 'category', 'targetLocation']):
                 logging.warning(f"Nuance data {i} is invalid or missing keys. Skipping: {nuance_data}")
                 continue

            text_to_find = str(nuance_data.get('text', '')).strip()
            if not text_to_find: # Skip if text is empty
                logging.warning(f"Nuance data {i} has empty text. Skipping.")
                continue

            target_loc_data = nuance_data.get('targetLocation')
            target_loc: Optional[TextLocation] = None

            st.text(f"DEBUG: Finding nuance target match for: '{text_to_find}'")
            # Simplified logic: Use find_best_match first. LLM location is secondary.
            match_loc = find_best_match(text_to_find, refined_translation)
            st.text(f"DEBUG: Nuance match result: {match_loc}")

            if match_loc:
                 target_loc = match_loc
                 # Optional: Could add logging here comparing match_loc to target_loc_data if needed
            else:
                 # If fuzzy match failed, maybe try the LLM's provided indices directly?
                 # For now, we prioritize fuzzy match. If it fails, we skip.
                 logging.warning(f"Could not confidently locate target text '{text_to_find}' for nuance {i}. Skipping.")
                 continue # Skip if fuzzy match failed

            # Only add if we found a location via find_best_match
            # (The condition 'if target_loc:' below handles this)
            if target_loc:
                st.text(f"DEBUG: Appending processed nuance {i}")
                processed_nuances.append(LinguisticNuance(
                    text=text_to_find,
                    explanation=str(nuance_data.get('explanation', 'N/A')),
                    category=str(nuance_data.get('category', 'Unknown')),
                    targetLocation=target_loc,
                    # Source location is not typically generated in this step
                ))
            else:
                logging.warning(f"Could not determine target location for nuance: '{text_to_find}'. Skipping.")

        st.text(f"DEBUG: Finished Nuance loop. Processed Nuances Count: {len(processed_nuances)}")
        return processed_nuances

    except Exception as e:
        logging.error(f"Error processing linguistic nuance data: {e}. Using fallback. Raw JSON: {nuances_json}")
        return _get_adjusted_fallback_nuances(refined_translation)

def _get_adjusted_fallback_nuances(refined_translation: str) -> List[LinguisticNuance]:
    """Creates fallback nuances adjusted to the length of the translation."""
    adjusted_fallbacks = []
    if refined_translation:
        len_trans = len(refined_translation)
        # First fallback nuance
        fb1_end = min(20, len_trans)
        fb1_text = refined_translation[0:fb1_end]
        if fb1_text:
            adjusted_fallbacks.append(
                LinguisticNuance(
                    text=fb1_text,
                    explanation=FALLBACK_LINGUISTIC_NUANCES[0].explanation,
                    category=FALLBACK_LINGUISTIC_NUANCES[0].category,
                    targetLocation=TextLocation(start=0, end=fb1_end),
                )
            )
        # Second fallback nuance (if text is long enough)
        if len_trans > 40:
             fb2_start = min(len_trans // 2, len_trans - 1)
             fb2_end = min(fb2_start + 20, len_trans)
             fb2_text = refined_translation[fb2_start:fb2_end]
             if fb2_text and len(adjusted_fallbacks) < 2:
                 adjusted_fallbacks.append(
                     LinguisticNuance(
                         text=fb2_text,
                         explanation=FALLBACK_LINGUISTIC_NUANCES[1].explanation,
                         category=FALLBACK_LINGUISTIC_NUANCES[1].category,
                         targetLocation=TextLocation(start=fb2_start, end=fb2_end),
                     )
                 )
    return adjusted_fallbacks 