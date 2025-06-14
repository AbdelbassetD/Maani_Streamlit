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
4. Key terminology (provide 2-10 (depending on the length of the text) Arabic terms with English meanings. ONLY provide key terms (given the Classical Arabic context), not ordinary or unimportant terms.)
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
1. **Prioritize Clarity and Quality:** Produce the most accurate, fluent, and contextually appropriate English translation possible. Do not attempt to necessarily change the initial translation or a part of it, if it is already high quality. DO NOT over-refine the translation.
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
You are an expert in Classical Arabic-to-English translation quality assessment with deep knowledge of Classical Arabic literature, linguistic nuance, and cultural contexts.

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

Your task is to analyze the provided ClassicalArabic text and its English translation to identify **ALL key cultural translation gaps/nuances**.

A cultural gap is a specific word, phrase, or expression in the source text that:
- Represents a concept, reference, value, social norm, idiom, or material culture unique to the source culture, and
- Lacks a direct or easily understood equivalent in modern English or the target culture, and
- Requires additional cultural, historical, or contextual knowledge for a modern English reader to fully understand.

**Important:**
- The gap MUST be a small, specific segment (word, phrase, or expression), NOT the entire sentence or a generic/overly broad segment.
- NEVER return the whole sentence or a segment that simply repeats the entire translation.
- DO NOT include ordinary words or phrases that have clear, direct equivalents in English.
- Focus on elements that would likely confuse, mislead, or lose meaning for a modern English reader without extra explanation.
- If in doubt, err on the side of including more specific, challenging segments rather than fewer.

**Examples of cultural gaps (NON-EXHAUSTIVE):**
1. The phrase "الحَمْدُ للهِ" is a reference to a common Islamic expression. In the translation, it is rendered as "Praise be to Allah". The gap is the specific reference to Allah, which is not present in the English translation.
2. The term "عصر الجاهلية" refers to the pre-Islamic era, a concept with deep cultural and historical meaning in Arabic. The gap is the phrase "عصر الجاهلية" and its translation "the Age of Ignorance".
3. The phrase "أهل الفتن" refers to people who incite civil strife, a concept with specific historical resonance. The gap is "أهل الفتن" and its translation "people of discord".
4. The idiom "ضرب في الأرض" literally means "struck in the land" but culturally means "to travel extensively". The gap is the idiom and its translation.
5. The term "ديوان" refers to a specific type of administrative office or collection of poetry, depending on context. The gap is the term and its translation.

**Non-examples (do NOT mark as gaps):**
- Generic words like "man", "house", "walk" unless used in a culturally loaded way.
- Segments that are simply long sentences or the entire translation.
- Phrases that have a clear, direct English equivalent with no cultural or historical ambiguity.

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
1.  **Identify All Gaps:** Carefully scan the original text and translation to find **all** instances representing a cultural gap as defined above. Be thorough and err on the side of including more specific, challenging segments.
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
    *   `overallStrategy`: Briefly describe the main approach(es) observed in handling cultural gaps across the entire translation (1 sentence). This should be informed by translation studies theory.
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