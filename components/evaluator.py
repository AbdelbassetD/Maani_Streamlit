import logging
import time
from typing import Optional

from core.llm_client import LLMClient
from config.prompts import get_evaluation_prompt
from config.settings import TEMP_EVALUATION, MAX_TOKENS_EVALUATION
from shared_types import ContextAnalysis, TranslationEvaluation, EvaluationScore
from utils.helpers import extract_json

# Define the fallback evaluation structure
FALLBACK_EVALUATION = TranslationEvaluation(
    initialTranslation=EvaluationScore(accuracy=7, fluency=7, nuance=6, culturalFidelity=6),
    refinedTranslation=EvaluationScore(accuracy=8, fluency=8, nuance=8, culturalFidelity=8),
    preferredTranslation='refined',
    preferenceConfidence=80,
    accuracyAssessment='Fallback: Refined likely more accurate.',
    fluencyAssessment='Fallback: Refined likely more fluent.',
    culturalFidelityAssessment='Fallback: Refined likely has better cultural fidelity.',
    generatedTime=0,
)

async def generate_translation_evaluation(
    llm_client: LLMClient,
    arabic_text: str,
    initial_translation: str,
    refined_translation: str,
    context: ContextAnalysis
) -> TranslationEvaluation:
    """Generates a detailed evaluation comparing translations using the LLM."""
    if not llm_client or not llm_client.model:
        logging.warning("LLMClient not available for evaluation. Using fallback.")
        return FALLBACK_EVALUATION

    prompt = get_evaluation_prompt(arabic_text, initial_translation, refined_translation, context)
    start_time = time.time()

    response_text = await llm_client.generate_text(
        prompt,
        temperature=TEMP_EVALUATION,
        max_output_tokens=MAX_TOKENS_EVALUATION
    )

    end_time = time.time()
    generated_time_ms = int((end_time - start_time) * 1000)

    if not response_text:
        logging.warning("Failed to get evaluation from LLM. Using fallback.")
        return FALLBACK_EVALUATION

    eval_json = extract_json(response_text)
    if not eval_json or not isinstance(eval_json, dict):
        logging.warning("Failed to parse evaluation JSON from LLM response. Using fallback.")
        return FALLBACK_EVALUATION

    # Validate structure before creating object
    try:
        required_top_keys = ['initialTranslation', 'refinedTranslation', 'preferredTranslation', 'preferenceConfidence']
        required_score_keys = ['accuracy', 'fluency', 'nuance', 'culturalFidelity']
        if not (
            all(k in eval_json for k in required_top_keys)
            and isinstance(eval_json.get('initialTranslation'), dict)
            and all(sk in eval_json['initialTranslation'] for sk in required_score_keys)
            and isinstance(eval_json.get('refinedTranslation'), dict)
            and all(sk in eval_json['refinedTranslation'] for sk in required_score_keys)
        ):
             raise ValueError("Evaluation JSON missing required keys or structure.")

        # Ensure scores are integers within range
        for t_key in ['initialTranslation', 'refinedTranslation']:
             for s_key in required_score_keys:
                  # Use .get with default to avoid KeyError, then cast
                  score_str = eval_json[t_key].get(s_key, '0')
                  try:
                      score = int(score_str)
                      if not 1 <= score <= 10:
                           logging.warning(f"Score {t_key}.{s_key} ({score}) out of range (1-10). Clamping to 1-10.")
                           score = max(1, min(10, score))
                      eval_json[t_key][s_key] = score # Store validated int
                  except (ValueError, TypeError):
                       logging.warning(f"Could not parse score {t_key}.{s_key} ('{score_str}') as int. Using 1.")
                       eval_json[t_key][s_key] = 1 # Default score on parse error

        pref_conf_str = eval_json.get('preferenceConfidence', '0')
        try:
            pref_conf = int(pref_conf_str)
            if not 0 <= pref_conf <= 100:
                 logging.warning(f"Preference confidence ({pref_conf}) out of range (0-100). Clamping.")
                 pref_conf = max(0, min(100, pref_conf))
            eval_json['preferenceConfidence'] = pref_conf
        except (ValueError, TypeError):
             logging.warning(f"Could not parse preference confidence ('{pref_conf_str}') as int. Using 0.")
             eval_json['preferenceConfidence'] = 0


        pref_trans = str(eval_json.get('preferredTranslation', 'initial')).lower()
        if pref_trans not in ['initial', 'refined']:
             logging.warning(f"Invalid preferred translation '{pref_trans}'. Defaulting to 'initial'.")
             pref_trans = 'initial'
        eval_json['preferredTranslation'] = pref_trans

        return TranslationEvaluation(
            initialTranslation=EvaluationScore(**eval_json['initialTranslation']),
            refinedTranslation=EvaluationScore(**eval_json['refinedTranslation']),
            preferredTranslation=eval_json['preferredTranslation'],
            preferenceConfidence=eval_json['preferenceConfidence'],
            accuracyAssessment=str(eval_json.get('accuracyAssessment', 'N/A')),
            fluencyAssessment=str(eval_json.get('fluencyAssessment', 'N/A')),
            culturalFidelityAssessment=str(eval_json.get('culturalFidelityAssessment', 'N/A')),
            generatedTime=generated_time_ms
        )
    except (ValueError, TypeError, KeyError) as e:
        logging.error(f"Error creating TranslationEvaluation object: {e}. Using fallback. Raw JSON: {eval_json}")
        return FALLBACK_EVALUATION 