import re
import random
import logging
from typing import List

from shared_types import TranslationConfidence, ConfidenceSegment
from config.settings import DEFAULT_CONFIDENCE

def generate_reliable_confidence(
    original_text: str,
    translated_text: str,
    is_refined: bool = False
) -> TranslationConfidence:
    """Generates a reliable confidence analysis for a translation (non-LLM)."""
    if not original_text or not translated_text:
         logging.warning("Cannot calculate confidence for empty text.")
         return TranslationConfidence(overall=DEFAULT_CONFIDENCE, segments=[])

    sentences = [s.strip() for s in re.split(r'[.!?]+', translated_text) if s.strip()]
    if not sentences:
        # If text exists but no sentences split, assign slightly higher base confidence
        logging.warning("Translated text contains no sentences for segment confidence.")
        return TranslationConfidence(overall=max(DEFAULT_CONFIDENCE, 0.6), segments=[])

    base_confidence = 0.78 if is_refined else 0.72
    overall_confidence = base_confidence

    # Factors affecting confidence
    try:
        # 1. Length ratio analysis
        actual_length_ratio = len(translated_text) / len(original_text) if len(original_text) > 0 else 1
        ideal_length_ratio = 1.45
        length_ratio_deviation = abs(actual_length_ratio - ideal_length_ratio) / ideal_length_ratio
        # Penalizes significant deviation, small bonus for being close
        length_ratio_factor = max(-0.08, min(0.05, 0.05 - length_ratio_deviation * 0.2))

        # 2. Untranslated terms (simple check for sequences of Arabic characters)
        arabic_term_regex = r'[\u0600-\u06FF]{4,}' # 4 or more consecutive Arabic chars
        arabic_matches = re.findall(arabic_term_regex, translated_text)
        untranslated_term_factor = max(-0.07, -0.01 * len(arabic_matches))

        # 3. Structural complexity analysis
        # a. Sentence length variance
        avg_sentence_length = sum(len(s) for s in sentences) / len(sentences)
        # Normalize variance by avg length, handle division by zero if avg is 0
        sentence_length_variances = [
            abs(len(s) - avg_sentence_length) / avg_sentence_length if avg_sentence_length > 0 else 0
            for s in sentences
        ]
        avg_variance = sum(sentence_length_variances) / len(sentence_length_variances)
        # Penalize high variance, small bonus for low variance
        variance_factor = max(-0.05, min(0.02, 0.02 - avg_variance * 0.1))

        # b. Sentence complexity (based on punctuation and clause indicators)
        # Regex to find complexity markers (punctuation and common subordinating conjunctions/relative pronouns)
        complexity_markers_regex = r'[,;:()\"\'\\u2014\[\]]|\b(which|that|who|whom|whose|where|when|because|although|while|since|if|unless|until|so that)\b'
        total_complexity_markers = sum(len(re.findall(complexity_markers_regex, s, re.IGNORECASE)) for s in sentences)
        avg_complexity_per_sentence = total_complexity_markers / len(sentences)
        # More complex sentences might be harder, slight penalty
        complexity_factor = max(-0.06, min(0.03, 0.03 - avg_complexity_per_sentence * 0.01))

        # 4. Source text complexity (word count)
        arabic_words = len(original_text.split())
        # Slight bonus for very short, slight penalty for very long
        word_count_factor = 0.02 if arabic_words < 10 else -0.04 if arabic_words > 100 else max(-0.04, min(0.02, 0.02 - arabic_words * 0.0002))

        # 5. Refinement bonus
        refinement_bonus = 0.06 if is_refined else 0

        # Calculate overall confidence
        calculated_confidence = (
            base_confidence +
            length_ratio_factor +
            untranslated_term_factor +
            variance_factor * 0.7 + # Weighted less heavily
            complexity_factor +
            word_count_factor * 0.5 + # Minor factor
            refinement_bonus
        )
        # Clamp confidence between reasonable bounds (e.g., 0.65 to 0.97)
        overall_confidence = min(0.97, max(0.65, calculated_confidence))

    except ZeroDivisionError:
        logging.warning("Division by zero encountered during confidence calculation. Using base confidence.")
        overall_confidence = base_confidence # Fallback if calculation fails (e.g., empty sentences list)
    except Exception as e:
        logging.error(f"Error during overall confidence calculation: {e}")
        overall_confidence = base_confidence # General fallback

    # Generate segment confidences
    segments_data: List[ConfidenceSegment] = []
    position = 0
    for sentence in sentences:
        # Ensure sentence ends with punctuation for splitting consistency
        sentence_with_period = sentence + ('' if re.search(r'[.!?]$', sentence) else '.')
        segment_length = len(sentence_with_period)

        # Segment-specific factors
        # Length factor: very short/long segments slightly less confident
        length_factor = -0.03 if segment_length < 15 else -0.05 if segment_length > 150 else 0.01
        # Complexity factor: more markers = slightly less confident
        complexity_count = len(re.findall(complexity_markers_regex, sentence_with_period, re.IGNORECASE))
        segment_complexity_factor = -0.04 if complexity_count > 5 else -0.02 if complexity_count > 3 else 0.01

        # Calculate segment confidence based on overall + segment factors + tiny random noise
        segment_base = overall_confidence + length_factor + segment_complexity_factor
        segment_confidence = min(0.98, max(0.60, segment_base + (random.random() * 0.02 - 0.01)))

        segments_data.append(ConfidenceSegment(
            text=sentence_with_period,
            start=position,
            end=position + segment_length,
            confidence=round(segment_confidence, 2)
        ))
        position += segment_length + 1 # Account for space after period/punctuation

    return TranslationConfidence(overall=round(overall_confidence, 2), segments=segments_data)