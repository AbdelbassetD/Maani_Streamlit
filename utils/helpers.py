import re
import json
import logging
from typing import Optional, Dict, Any
from thefuzz import process, fuzz # Import fuzzy matching

from shared_types import TextLocation # Assuming shared_types is in root or PYTHONPATH

def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extracts the first valid JSON object or array from a string."""
    if not text: return None

    # Try to find JSON object {} first
    obj_match = re.search(r'\{(?:[^{}]|\{[^{}]*\})*\}', text, re.DOTALL)
    # Try to find JSON array [] next
    arr_match = re.search(r'\[(?:[^\[\]]|\[[^\[\]]*\])*\]', text, re.DOTALL)

    match = None
    if obj_match and arr_match:
        # If both found, take the one that starts earlier
        match = obj_match if obj_match.start() < arr_match.start() else arr_match
    elif obj_match:
        match = obj_match
    elif arr_match:
        match = arr_match

    if not match:
        logging.warning("No JSON object or array found in text.")
        return None

    try:
        # Use loads on the matched string segment
        return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse extracted JSON: {e}\nMatched text: {match.group(0)}\nOriginal text: {text[:500]}...")
        return None

def find_best_match(term: str, text: str, score_cutoff: int = 75) -> Optional[TextLocation]:
    """Finds the best fuzzy match for a term in a text using thefuzz."""
    if not term or not text or len(term) < 3: # Basic checks
        return None

    try:
        # Use process.extractOne to find the best match above a certain score
        # We are essentially looking for the term *within* the text.
        # This requires a slightly different approach than finding the best choice from a list.
        # Let's iterate through possible substrings of text that are similar in length to the term.

        term_len = len(term)
        best_match_info = {"score": 0, "start": -1, "end": -1}

        # Iterate through the text to find potential matches
        # Only compare substrings of similar length to the term for efficiency
        # Allow some variation in length (e.g., +/- 20% or a fixed window)
        min_len = max(1, term_len - 5) # Allow slightly shorter matches
        max_len = term_len + 10 # Allow slightly longer matches

        for length in range(min_len, max_len + 1):
             for i in range(len(text) - length + 1):
                 substring = text[i : i + length]
                 # Calculate similarity score
                 score = fuzz.ratio(term.lower(), substring.lower())

                 # Prefer WRatio for more complex comparisons (optional)
                 # score = fuzz.WRatio(term.lower(), substring.lower())

                 if score > best_match_info["score"]:
                     best_match_info = {"score": score, "start": i, "end": i + length}

        # Check if the best score found meets the cutoff
        if best_match_info["score"] >= score_cutoff and best_match_info["start"] != -1:
            logging.debug(f"Fuzzy match found for '{term}': '{text[best_match_info['start']:best_match_info['end']]}' (Score: {best_match_info['score']})")
            return TextLocation(start=best_match_info["start"], end=best_match_info["end"])
        else:
            logging.debug(f"No suitable fuzzy match found for '{term}' (Best score: {best_match_info['score']} < {score_cutoff})")
            return None # No match above cutoff

    except Exception as e:
        logging.error(f"Error during fuzzy matching for term '{term}': {e}")
        return None

    # --- Old Logic ---
    # try:
    #     # Exact match first
    #     exact_index = text.find(term)
    #     if exact_index != -1:
    #         return TextLocation(start=exact_index, end=exact_index + len(term))
    #
    #     # Case-insensitive match
    #     lower_text = text.lower()
    #     lower_term = term.lower()
    #     lower_match_index = lower_text.find(lower_term)
    #     if lower_match_index != -1:
    #         return TextLocation(start=lower_match_index, end=lower_match_index + len(term))
    #
    #     # Fuzzy matching for Arabic (simple diacritic removal)
    #     if re.search(r'[\u0600-\u06FF]', term):
    #         normalized_text = re.sub(r'[\u064B-\u065F]', '', text)
    #         normalized_term = re.sub(r'[\u064B-\u065F]', '', term)
    #         if not normalized_term: # Skip if term becomes empty after normalization
    #             return None
    #         normalized_match_index = normalized_text.find(normalized_term)
    #         if normalized_match_index != -1:
    #             # Estimate original position (can be inaccurate)
    #             # This is a simplification; proper alignment is complex
    #             return TextLocation(start=normalized_match_index, end=normalized_match_index + len(normalized_term))
    #
    #     # Partial match (at least 80%)
    #     # Avoid division by zero if term length is 0 (though checked earlier)
    #     if len(term) == 0: return None
    #     min_length = max(3, int(len(term) * 0.8))
    #     if len(text) < min_length: return None # Text too short for partial match
    #
    #     for i in range(len(text) - min_length + 1):
    #         sub_text = text[i : min(i + len(term), len(text))] # Ensure sub_text index is valid
    #         match_count = 0
    #         for j in range(min(len(sub_text), len(term))):
    #             if sub_text[j].lower() == term[j].lower():
    #                 match_count += 1
    #         match_ratio = match_count / len(term)
    #         if match_ratio >= 0.8:
    #             return TextLocation(start=i, end=i + len(sub_text))
    #
    # except Exception as e:
    #     logging.error(f"Error during find_best_match for term '{term}': {e}")
    #
    # # No reliable match found
    # return None