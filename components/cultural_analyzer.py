import logging
import time
from typing import List, Optional
import streamlit as st

from core.llm_client import LLMClient
from config.prompts import get_cultural_gap_analysis_prompt
from config.settings import TEMP_CULTURAL_ANALYSIS, MAX_TOKENS_CULTURAL_ANALYSIS, DEFAULT_CULTURAL_EFFECTIVENESS, DEFAULT_STRATEGY
from shared_types import ContextAnalysis, CulturalGapAnalysis, CulturalGap, TextLocation
from utils.helpers import extract_json, find_best_match

# Define fallback cultural gaps
FALLBACK_CULTURAL_GAPS = [
    CulturalGap(
        name='Fallback Cultural Reference',
        category='Historical Context',
        description='Fallback: Specific cultural reference requiring background knowledge.',
        translationStrategy='Contextual explanation (fallback)',
        sourceLocation=TextLocation(start=0, end=20),
        targetLocation=TextLocation(start=0, end=30),
    ),
    CulturalGap(
        name='Fallback Idiomatic Expression',
        category='Linguistic',
        description='Fallback: Arabic expression with no direct English equivalent.',
        translationStrategy='Functional equivalent (fallback)',
        sourceLocation=TextLocation(start=50, end=65),
        targetLocation=TextLocation(start=60, end=80),
    ),
    CulturalGap(
        name='Placeholder Gap',
        category='Placeholder',
        description='Placeholder description for missing gap.',
        translationStrategy='Placeholder strategy.',
        sourceLocation=TextLocation(start=100, end=110),
        targetLocation=TextLocation(start=120, end=135)
    )
]

def generate_cultural_gap_analysis(
    llm_client: LLMClient,
    arabic_text: str,
    refined_translation: str,
    context: ContextAnalysis
) -> CulturalGapAnalysis:
    """Generates cultural gap analysis using the LLM."""
    if not llm_client or not llm_client.model:
        logging.warning("LLMClient not available for cultural gap analysis. Using fallback.")
        return CulturalGapAnalysis(gaps=FALLBACK_CULTURAL_GAPS, overallStrategy=DEFAULT_STRATEGY, effectivenessRating=DEFAULT_CULTURAL_EFFECTIVENESS)

    prompt = get_cultural_gap_analysis_prompt(arabic_text, refined_translation, context)
    start_time = time.time()

    response_text = llm_client.generate_text(
        prompt,
        temperature=TEMP_CULTURAL_ANALYSIS,
        max_output_tokens=MAX_TOKENS_CULTURAL_ANALYSIS
    )

    end_time = time.time()
    generated_time_ms = int((end_time - start_time) * 1000)

    # --- START DEBUGGING --- #
    st.text(f"DEBUG: LLM Raw Response (Cultural Gap Length): {len(response_text) if response_text else 0}")
    # --- END DEBUGGING --- #

    if not response_text:
        logging.warning("Failed to get cultural gap analysis from LLM. Using fallback.")
        return CulturalGapAnalysis(gaps=FALLBACK_CULTURAL_GAPS, overallStrategy=DEFAULT_STRATEGY, effectivenessRating=DEFAULT_CULTURAL_EFFECTIVENESS)

    raw_json_data = extract_json(response_text)
    # --- START DEBUGGING --- #
    st.text(f"DEBUG: Extracted JSON Data (Cultural Gap): {raw_json_data}")
    # --- END DEBUGGING --- #

    gap_list = []
    overall_strategy = DEFAULT_STRATEGY
    effectiveness_rating = DEFAULT_CULTURAL_EFFECTIVENESS

    if isinstance(raw_json_data, dict):
        # Case 1: Correct structure { "gaps": [...], "overallStrategy": ..., ... }
        if 'gaps' in raw_json_data and isinstance(raw_json_data['gaps'], list):
            # --- START DEBUGGING --- #
            st.text("DEBUG: Parsing Case 1 - Correct Structure")
            # --- END DEBUGGING --- #
            gap_list = raw_json_data.get('gaps', [])
            overall_strategy = str(raw_json_data.get('overallStrategy', DEFAULT_STRATEGY))
            eff_rating_str = raw_json_data.get('effectivenessRating', str(DEFAULT_CULTURAL_EFFECTIVENESS))
            try:
                eff_rating = int(eff_rating_str)
                effectiveness_rating = min(10, max(1, eff_rating))
            except (ValueError, TypeError):
                logging.warning(f"Could not parse effectiveness rating ('{eff_rating_str}') as int. Using default {DEFAULT_CULTURAL_EFFECTIVENESS}.")
                effectiveness_rating = DEFAULT_CULTURAL_EFFECTIVENESS
        # Case 2: Single gap object returned directly {...}
        elif 'name' in raw_json_data: # Heuristic: If it looks like a gap object itself
            # --- START DEBUGGING --- #
            st.text("DEBUG: Parsing Case 2 - Single Gap Object")
            # --- END DEBUGGING --- #
            logging.warning("LLM returned a single gap object instead of the expected structure. Processing as a single gap.")
            gap_list = [raw_json_data]
            # Cannot get overall strategy/rating, use defaults
            overall_strategy = DEFAULT_STRATEGY
            effectiveness_rating = DEFAULT_CULTURAL_EFFECTIVENESS
        else:
            # Dictionary format is wrong
            # --- START DEBUGGING --- #
            st.text("DEBUG: Parsing Case Failed - Dictionary format wrong")
            # --- END DEBUGGING --- #
            logging.warning(f"Failed to parse cultural gap JSON structure. Using fallback. JSON: {raw_json_data}")
            return CulturalGapAnalysis(gaps=FALLBACK_CULTURAL_GAPS, overallStrategy=DEFAULT_STRATEGY, effectivenessRating=DEFAULT_CULTURAL_EFFECTIVENESS)

    elif isinstance(raw_json_data, list):
        # Case 3: LLM returned just a list of gaps [...] (less likely but possible)
        # --- START DEBUGGING --- #
        st.text("DEBUG: Parsing Case 3 - List of Gaps")
        # --- END DEBUGGING --- #
        logging.warning("LLM returned a list of gaps instead of the expected structure. Processing list directly.")
        gap_list = raw_json_data
        # Cannot get overall strategy/rating, use defaults
        overall_strategy = DEFAULT_STRATEGY
        effectiveness_rating = DEFAULT_CULTURAL_EFFECTIVENESS
    else:
        # Not a dict or list, parsing failed
        # --- START DEBUGGING --- #
        st.text("DEBUG: Parsing Case Failed - Not Dict or List")
        # --- END DEBUGGING --- #
        logging.warning(f"Failed to parse cultural gap JSON response (not dict or list). Using fallback. Raw Response: {response_text[:200]}...")
        return CulturalGapAnalysis(gaps=FALLBACK_CULTURAL_GAPS, overallStrategy=DEFAULT_STRATEGY, effectivenessRating=DEFAULT_CULTURAL_EFFECTIVENESS)

    processed_gaps: List[CulturalGap] = []
    try:
        # Validate and process gaps from the extracted gap_list
        st.text(f"DEBUG: Starting loop. Gap List Length: {len(gap_list)}")
        for i, gap_data in enumerate(gap_list):
            st.text(f"DEBUG: Processing Gap {i}: {gap_data}")
            if not isinstance(gap_data, dict) or not all(k in gap_data for k in ['name', 'category', 'description', 'translationStrategy', 'sourceText', 'targetText']):
                logging.warning(f"Gap data {i} is invalid or missing keys. Skipping.")
                continue

            source_text = str(gap_data.get('sourceText', '')).strip()
            target_text = str(gap_data.get('targetText', '')).strip()

            st.text(f"DEBUG: Finding source match for: '{source_text}'")
            source_loc = find_best_match(source_text, arabic_text) if source_text else None
            st.text(f"DEBUG: Finding target match for: '{target_text}'")
            target_loc = find_best_match(target_text, refined_translation) if target_text else None
            st.text(f"DEBUG: Match results - Source: {source_loc}, Target: {target_loc}")

            # If the LLM provided text but we couldn't find the TARGET text, skip the gap.
            # It's okay if source_loc is None, we just won't display the source snippet.
            if target_text and not target_loc:
                 logging.warning(f"Could not confidently locate target text '{target_text}' provided by LLM for gap '{gap_data.get('name', 'N/A')}'. Skipping gap.")
                 continue # Skip this gap

            # Only add if we successfully processed (target_loc might be None if target_text was empty)
            st.text(f"DEBUG: Appending processed gap {i} (SourceLoc: {source_loc}, TargetLoc: {target_loc})")
            processed_gaps.append(CulturalGap(
                name=str(gap_data.get('name', f'Unknown Gap {i+1}')),
                category=str(gap_data.get('category', 'Unknown')),
                description=str(gap_data.get('description', 'N/A')),
                translationStrategy=str(gap_data.get('translationStrategy', 'N/A')),
                sourceLocation=source_loc,
                targetLocation=target_loc,
            ))

        # Note: overallStrategy and effectiveness_rating are handled during initial parsing
        st.text(f"DEBUG: Finished loop. Processed Gaps Count: {len(processed_gaps)}")

        return CulturalGapAnalysis(
            gaps=processed_gaps, # Return the processed gaps without forcing count
            overallStrategy=overall_strategy,
            effectivenessRating=effectiveness_rating,
            generatedTime=generated_time_ms
        )

    except Exception as e:
         logging.error(f"Error processing cultural gap data: {e}. Using fallback. Raw JSON: {raw_json_data}")
         return CulturalGapAnalysis(gaps=FALLBACK_CULTURAL_GAPS, overallStrategy=f"{DEFAULT_STRATEGY} (Error Fallback)", effectivenessRating=DEFAULT_CULTURAL_EFFECTIVENESS - 2)