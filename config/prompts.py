"""Stores the prompts used for interacting with the Generative AI model."""
import json
from shared_types import ContextAnalysis # Assuming shared_types is accessible
import logging

def get_initial_translation_prompt(arabic_text: str) -> str:
    return f"""Translate the following Classical Arabic text to English. Provide only the English translation.

Arabic Text:
{arabic_text}

English Translation:"""

def get_context_analysis_prompt(arabic_text: str) -> str:
    return f"""
You are a scholar of Classical Arabic literature and linguistics with expertise in historical and cultural analysis.

Analyze the provided Arabic text and extract key contextual information:

1. The genre/type of the text
2. The time period it represents
3. The tone and register
4. Key terminology (provide 4-5 Arabic terms with English meanings)
5. Historical and cultural context (1-2 sentences)

ARABIC TEXT:
{arabic_text}

Provide your analysis ONLY in the following JSON format:
{{
  "genre": "The literary genre (NO quotes or colons at beginning)",
  "timePeriod": "The historical period or era (NO quotes or colons at beginning)",
  "tone": "The tone/voice of the text (NO quotes or colons at beginning)",
  "keyTerminology": [
    "Arabic term 1 (English meaning 1)",
    "Arabic term 2 (English meaning 2)",
    "Arabic term 3 (English meaning 3)",
    "Arabic term 4 (English meaning 4)"
  ],
  "historicalContext": "A concise paragraph explaining the historical/cultural context necessary for understanding."
}}
"""

def get_refinement_prompt(arabic_text: str, initial_translation: str, context: ContextAnalysis) -> str:
    # Safely format key terminology for the prompt
    key_terms_str = json.dumps(context.keyTerminology) if context and context.keyTerminology else "[]"
    genre = context.genre if context else "Unknown"
    time_period = context.timePeriod if context else "Unknown"
    tone = context.tone if context else "Unknown"
    hist_context = context.historicalContext if context else "Not available"

    return f"""
You are an expert translator of Classical Arabic to English with extensive knowledge of historical texts, religious concepts, and cultural nuances.

Your task is to provide the **highest quality English translation** of the original Arabic text, potentially improving upon the provided initial translation by incorporating deeper cultural insights, using more appropriate period-specific terminology, and creating a more fluent and idiomatic translation. The goal is the best possible English rendering.

Original Arabic Text:
{arabic_text}

Context Analysis:
- Genre: {genre}
- Time Period: {time_period}
- Tone: {tone}
- Key Terminology: {key_terms_str}
- Historical Context: {hist_context}

Initial Translation (for reference only):
{initial_translation}

**REFINEMENT INSTRUCTIONS:**
1. **Prioritize Clarity and Quality:** Produce the most accurate, fluent, and contextually appropriate English translation possible. Do not attempt to necessarily change the initial translation or a part of it, if it is already high quality.
2. **Use English Equivalents:** Translate all concepts, including culturally specific ones, into their best English equivalents. **DO NOT** use direct transliterations (e.g., 'Fitna') as the primary translation.
3. **Optional Supplementation:** If a culturally specific term (like 'Fitna' or 'Jahiliyyah') has been translated to English but you believe providing the original term adds crucial context, you may append the transliterated term and a brief, succinct explanation within square brackets immediately after the English equivalent. Example: "...civil discord [Fitna: internal strife/sedition]..." or "...the pre-Islamic era [Jahiliyyah]...". Use this sparingly and only when it aids understanding.
4. **Maintain Semantic Accuracy:** Ensure the meaning of the original Arabic is fully preserved.
5. **Enhance Nuance:** Capture cultural nuances, register, and tone appropriately through careful English word choice.
6. **Improve Fluency:** Ensure the final translation reads naturally in English.

Provide ONLY the refined English translation text.
"""

def get_evaluation_prompt(arabic_text: str, initial_translation: str, refined_translation: str, context: ContextAnalysis) -> str:
    # Safely format key terminology for the prompt
    key_terms_str = json.dumps(context.keyTerminology) if context and context.keyTerminology else "[]"
    genre = context.genre if context else "Unknown"
    time_period = context.timePeriod if context else "Unknown"
    tone = context.tone if context else "Unknown"
    hist_context = context.historicalContext if context else "Not available"

    return f"""
You are an expert in Arabic-English translation quality assessment with deep knowledge of Classical Arabic literature, linguistic nuance, and cultural contexts.

Carefully evaluate two translations of the same Arabic text OBJECTIVELY. The refined translation should only receive higher scores if it genuinely improves upon the initial translation in measurable ways. Do not automatically favor the refined translation.

Original Arabic Text:
{arabic_text}

Context Analysis:
- Genre: {genre}
- Time Period: {time_period}
- Tone: {tone}
- Key Terminology: {key_terms_str}
- Historical Context: {hist_context}

Initial Translation:
{initial_translation}

Refined Translation:
{refined_translation}

Evaluate both translations using these criteria (score each from 1-10):
1. ACCURACY (Semantic correctness, terminology)
2. FLUENCY (Readability, grammar, naturalness)
3. NUANCE (Subtlety, tone, author's voice)
4. CULTURAL FIDELITY (Context, references, concepts)

Provide:
- Scores for each translation on all 4 criteria.
- Which translation is preferred overall (initial or refined).
- Preference confidence (0-100%).
- Brief assessment (1 sentence each) comparing accuracy, fluency, and cultural fidelity.

Provide ONLY the evaluation in the following JSON format:
{{
  "initialTranslation": {{ "accuracy": <1-10>, "fluency": <1-10>, "nuance": <1-10>, "culturalFidelity": <1-10> }},
  "refinedTranslation": {{ "accuracy": <1-10>, "fluency": <1-10>, "nuance": <1-10>, "culturalFidelity": <1-10> }},
  "preferredTranslation": "initial" or "refined",
  "preferenceConfidence": <0-100>,
  "accuracyAssessment": "Brief comparison of accuracy.",
  "fluencyAssessment": "Brief comparison of fluency.",
  "culturalFidelityAssessment": "Brief comparison of cultural fidelity."
}}
"""

def get_cultural_gap_analysis_prompt(arabic_text: str, refined_translation: str, context: ContextAnalysis) -> str:
    key_terms_str = json.dumps(context.keyTerminology) if context and context.keyTerminology else "[]"
    genre = context.genre if context else "Unknown"
    time_period = context.timePeriod if context else "Unknown"
    tone = context.tone if context else "Unknown"
    hist_context = context.historicalContext if context else "Not available"

    return f"""You are a specialist in cross-cultural communication and translation studies, focusing on Classical Arabic to modern English challenges.

Your task is to analyze the provided Arabic text and its English translation to identify **ALL significant cultural translation gaps**. A cultural gap occurs where concepts, references, values, social norms, idioms, or material culture from the source text do not have a direct or easily understood equivalent in the target language and culture (modern English). Focus on elements requiring cultural context beyond simple lexical equivalence.

Original Arabic Text:
```
{arabic_text}
```

Context Analysis:
- Genre: {genre}
- Time Period: {time_period}
- Tone: {tone}
- Key Terminology: {key_terms_str}
- Historical Context: {hist_context}

Refined English Translation:
```
{refined_translation}
```

**INSTRUCTIONS FOR ANALYSIS:**
1.  **Identify All Gaps:** Thoroughly scan the original text and translation to find **all** instances representing a cultural gap as defined above.
2.  **Extract Verbatim Segments:** For each identified gap, you **MUST** extract the corresponding text segments *exactly* as they appear:
    *   `sourceText`: The **exact VERBATIM (no transliteration) phrase or term** from the *Original Arabic Text* that represents the gap.
    *   `targetText`: The **exact corresponding phrase or term** from the *Refined English Translation* where the gap is addressed or translated.
    *   **Accuracy is critical.** Ensure these segments semantically match and are copied **VERBATIM (character-for-character)**. Do **NOT** transliterate, summarize, or paraphrase the `sourceText` or `targetText`. Example: If the Arabic text is `الحَمْدُ للهِ`, the `sourceText` MUST be `الحَمْدُ للهِ`, not `Alhamdulillah` or `Praise Allah`.
3.  **Describe Each Gap:** For each gap, provide:
    *   `name`: A concise, descriptive name for the gap (e.g., 'Concept of Adab', 'Historical Event Reference', 'Islamic Legal Term').
    *   `category`: Classify the gap (e.g., 'Religious Concept', 'Historical Reference', 'Social Norm', 'Idiom', 'Material Culture', 'Linguistic-Cultural').
    *   `description`: Briefly explain *why* this specific element presents a cultural translation challenge (1-2 sentences).
    *   `translationStrategy`: Describe the strategy used in the refined translation to bridge the gap (e.g., 'Literal Translation', 'Functional Equivalent', 'Explanatory Translation', 'Cultural Substitution', 'Omission', 'Compensation', 'Transliteration with explanation').
4.  **Summarize:**
    *   `overallStrategy`: Briefly describe the main approach(es) observed in handling cultural gaps across the entire translation (1 sentence).
    *   `effectivenessRating`: Rate the overall success (1-10) of the refined translation in bridging the identified cultural gaps.

**OUTPUT FORMAT:**
Provide ONLY the analysis in the following JSON format. Ensure `sourceText` and `targetText` are exact verbatim copies.

{{
  "gaps": [
    {{ "name": "...", "category": "...", "description": "...", "translationStrategy": "...", "sourceText": "VERBATIM Arabic segment", "targetText": "VERBATIM English segment" }},
    {{ "name": "...", "category": "...", "description": "...", "translationStrategy": "...", "sourceText": "VERBATIM Arabic segment", "targetText": "VERBATIM English segment" }}
    // ... include ALL identified gaps ...
  ],
  "overallStrategy": "description of overall approach",
  "effectivenessRating": <1-10>
}}
"""

def get_linguistic_nuance_prompt(arabic_text: str, refined_translation: str, context: ContextAnalysis) -> str:
    # Safely format key terminology for the prompt
    genre = context.genre if context else "Unknown"
    time_period = context.timePeriod if context else "Unknown"
    tone = context.tone if context else "Unknown"
    hist_context = context.historicalContext if context else "Not available"

    return f"""
You are a linguistics expert specialized in Arabic-English translations.

Identify linguistic nuances in the following translation that would benefit from additional explanation for an English reader.

Original Arabic Text:
{arabic_text}

English Translation:
{refined_translation}

Context Information:
- Genre: {genre}
- Time Period: {time_period}
- Tone: {tone}
- Historical Context: {hist_context}

Identify 3-7 specific phrases, terms, or expressions in the English translation that:
1. Have cultural or contextual significance not immediately clear.
2. Contain subtle wordplay, metaphors, idioms, or rhetorical devices.
3. Have ambiguous meanings or connotations worth explaining.
4. Represent formal or stylistic elements specific to the genre.

For each nuance, provide:
1. text: The EXACT phrase from the English translation.
2. explanation: Concise explanation (30-60 words).
3. category: One of: idiom, metaphor, cultural-reference, wordplay, ambiguity, rhetorical-device, formality, connotation.
4. targetLocation: {{ "start": <char_start_index>, "end": <char_end_index> }} in the English text.

Format your response ONLY as a JSON array:
[
  {{
    "text": "exact phrase from English translation",
    "explanation": "concise explanation of the nuance",
    "category": "category_label",
    "targetLocation": {{"start": number, "end": number}}
  }}
  ...
]
"""