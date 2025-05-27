# Configuration settings for the translation service

# --- Google AI Model Configuration ---
DEFAULT_MODEL_NAME = 'gemini-2.5-flash-preview-05-20'

# Safety settings to minimize refusals for translation/analysis tasks
# Options: BLOCK_NONE, BLOCK_ONLY_HIGH, BLOCK_MEDIUM_AND_ABOVE, BLOCK_LOW_AND_ABOVE
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# --- Generation Parameters (Temperatures) ---
# Lower values = more deterministic; Higher values = more creative
TEMP_INITIAL_TRANSLATION = 0.5
TEMP_CONTEXT_ANALYSIS = 0.3
TEMP_REFINEMENT = 0.5
TEMP_EVALUATION = 0.2
TEMP_CULTURAL_ANALYSIS = 0.25
TEMP_NUANCE_ANALYSIS = 0.3

# --- Generation Parameters (Max Output Tokens) ---
# Adjust based on expected output lengths and model limits
MAX_TOKENS_INITIAL_TRANSLATION = 1500
MAX_TOKENS_CONTEXT_ANALYSIS = 500
MAX_TOKENS_REFINEMENT = 1500
MAX_TOKENS_EVALUATION = 800
MAX_TOKENS_CULTURAL_ANALYSIS = 2000
MAX_TOKENS_NUANCE_ANALYSIS = 1500

# --- Fallback/Default Values (can be moved elsewhere if they grow large) ---
# These are used if the API fails or doesn't return expected data.
# Consider moving detailed fallbacks (like specific gaps/nuances) to a separate file if needed.

DEFAULT_CONFIDENCE = 0.5 # Default overall confidence if calculation fails
DEFAULT_EVALUATION_RATING = 7
DEFAULT_CULTURAL_EFFECTIVENESS = 5
DEFAULT_STRATEGY = "Strategy not specified."