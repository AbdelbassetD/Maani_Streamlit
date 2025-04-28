import streamlit as st
import asyncio
import colorsys # Import the colorsys module
import json # For feedback logging and download
import os # For feedback logging path
from datetime import datetime # For feedback timestamp
from typing import List, Tuple, Optional, Dict
import pandas as pd # For evaluation bar chart
import dataclasses # Import dataclasses module
import logging # Added for logging

# Import genai here for configuration
import google.generativeai as genai

# Updated imports for modular structure
from core.llm_client import LLMClient
from core.translation_orchestrator import TranslationOrchestrator
from shared_types import TranslationResult, InputText, TranslationStep, LinguisticNuance, CulturalGap, TextLocation
from utils.helpers import extract_json, find_best_match

# --- Feedback Logging Setup ---
FEEDBACK_LOG_FILE = "feedback_log.jsonl" # Use JSON Lines format

def log_feedback(input_text: str, initial_translation: str, refined_translation: str, feedback_type: str):
    """Logs user feedback to a file."""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "input_text": input_text,
        "initial_translation": initial_translation,
        "refined_translation": refined_translation,
        "feedback": feedback_type # "positive" or "negative"
    }
    try:
        with open(FEEDBACK_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        print(f"Logged {feedback_type} feedback.") # Log to console as well
    except Exception as e:
        print(f"Error logging feedback: {e}") # Print error to console

# --- Streamlit Page Configuration (MUST BE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="Ma'ani - Classical Arabic to English Translator",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
    # The 'theme' argument is removed as it's not supported in this version
    # Theme settings are now in .streamlit/config.toml
)

# --- Color Generation Helpers ---
def hsl_to_hex(h, s, l) -> str:
    """Converts HSL color values to a HEX string."""
    try:
        r, g, b = colorsys.hls_to_rgb(h / 360.0, l, s) # Note: colorsys uses HLS
        r_int, g_int, b_int = int(r * 255), int(g * 255), int(b * 255)
        # Ensure values are within 0-255 range
        r_int = max(0, min(255, r_int))
        g_int = max(0, min(255, g_int))
        b_int = max(0, min(255, b_int))
        return f"#{r_int:02x}{g_int:02x}{b_int:02x}"
    except Exception:
        # Fallback color in case of error
        return "#D3D3D3" # Light Gray

def generate_distinct_color(index: int, base_hue: int, saturation: float = 0.8, lightness: float = 0.85) -> str:
    """Generates a distinct color based on index using HSL hue rotation.

    Args:
        index: The index of the item (0, 1, 2...).
        base_hue: The starting hue (0-360) for the category (e.g., 0 for gaps, 200 for nuances).
        saturation: Saturation value (0.0-1.0). Higher is more vibrant.
        lightness: Lightness value (0.0-1.0). Higher is lighter.

    Returns:
        A HEX color string.
    """
    # Use golden ratio conjugate for hue distribution - tends to produce distinct hues
    hue_step = 137.507764
    hue = (base_hue + index * hue_step) % 360
    return hsl_to_hex(hue, saturation, lightness)

# Initialize the LLM client and orchestrator once using secrets
@st.cache_resource # Cache the initialized services
def get_services():
    print("Initializing services using Streamlit Secrets...") # Add print to see when this runs
    # Configure genai with the key from Streamlit secrets
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        print("Google AI configured successfully.")
    except Exception as e:
        st.error(f"Failed to configure Google AI. Have you set the GOOGLE_API_KEY secret in Streamlit Cloud? Error: {e}")
        print(f"Failed to configure Google AI: {e}")
        return None # Return None if configuration fails

    # Initialize LLMClient *after* configuration
    try:
        llm_client = LLMClient()
        orchestrator = TranslationOrchestrator(llm_client)
        print("LLMClient and Orchestrator initialized.")
        return orchestrator
    except Exception as e:
        st.error(f"Failed to initialize LLM Client/Orchestrator after configuration. Error: {e}")
        print(f"Failed to initialize LLM Client/Orchestrator: {e}")
        return None # Return None if initialization fails

# Get the orchestrator instance (will be cached)
translation_orchestrator = get_services()

# --- Helper Functions for Display ---

def display_confidence(label: str, confidence_data):
    if confidence_data:
        st.metric(f"{label} Overall Confidence", f"{confidence_data.overall * 100:.1f}%")
        # Optionally display segment confidences if needed
        # with st.expander(f"Show {label} Segment Confidences"):
        #     st.dataframe(confidence_data.segments)
    else:
        st.metric(f"{label} Overall Confidence", "N/A")

def display_evaluation_scores(label: str, scores):
    cols = st.columns(4)
    cols[0].metric(f"{label} Accuracy", f"{scores.accuracy}/10")
    cols[1].metric(f"{label} Fluency", f"{scores.fluency}/10")
    cols[2].metric(f"{label} Nuance", f"{scores.nuance}/10")
    cols[3].metric(f"{label} Cultural Fidelity", f"{scores.culturalFidelity}/10")

# Simple highlighter function (replace with more robust solution if needed)
def highlight_text(text: str, locations: List[Tuple[TextLocation, str, str]]) -> str:
    """Highlights text segments with tooltips. locations is List[(location, color, tooltip)]"""

    # --- START DEBUGGING: Simple Output --- #
    if locations and isinstance(text, str):
        # If there are locations, just prepend a simple colored block to the text
        # Use the color from the first location for simplicity
        debug_color = locations[0][1] if locations and len(locations[0]) > 1 else "#FFCC00" # Fallback yellow
        print(f"DEBUG highlight_text: Received {len(locations)} locations. Applying debug marker.")
        return f'<span style="background-color:{debug_color}; padding: 2px 6px; border-radius: 3px; color: black; font-weight: bold;">HIGHLIGHTED</span> {text}'
    else:
        print(f"DEBUG highlight_text: No locations received or text invalid.")
        return text if isinstance(text, str) else ""
    # --- END DEBUGGING: Simple Output --- #

    # --- Original Logic (Commented Out for Debugging) --- #
    # if not locations or not isinstance(text, str) or not text:
    #     return text if isinstance(text, str) else ""
    #
    # text_len = len(text)
    # highlighted_parts = []
    # last_end = 0
    #
    # try:
    #     locations.sort(key=lambda x: x[0].start if isinstance(x[0], TextLocation) and hasattr(x[0], 'start') else 0)
    # except Exception as e:
    #     print(f"Error sorting highlight locations: {e}. Skipping highlighting.")
    #     return text
    #
    # for loc_tuple in locations:
    #     if not isinstance(loc_tuple, tuple) or len(loc_tuple) != 3:
    #         print(f"Skipping invalid location tuple format: {loc_tuple}")
    #         continue
    #     loc, color, tooltip = loc_tuple
    #     if not isinstance(loc, TextLocation) or not hasattr(loc, 'start') or not hasattr(loc, 'end'):
    #         print(f"Skipping invalid/incomplete TextLocation object: {loc}")
    #         continue
    #     start, end = loc.start, loc.end
    #     if not (isinstance(start, int) and isinstance(end, int)):
    #         print(f"Skipping non-integer indices: start={start} ({type(start)}), end={end} ({type(end)}) TextLen={text_len}")
    #         continue
    #     if not (0 <= start <= text_len and 0 <= end <= text_len):
    #         print(f"Skipping out-of-bounds indices: start={start}, end={end}, TextLen={text_len}")
    #         continue
    #     if start > end:
    #         print(f"Skipping invalid range (start > end): start={start}, end={end}, TextLen={text_len}")
    #         continue
    #     start = max(start, last_end)
    #     if start >= end:
    #          continue
    #     highlighted_parts.append(text[last_end:start])
    #     highlighted_segment = text[start:end]
    #     safe_tooltip = str(tooltip).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\"", "&quot;").replace("'", "&#39;").replace("\n", " ")
    #     highlighted_parts.append(
    #         f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px;" title="{safe_tooltip}">{highlighted_segment}</span>'
    #     )
    #     last_end = end
    # highlighted_parts.append(text[last_end:])
    # return "".join(highlighted_parts)
    # --- End Original Logic --- #

# --- Example Texts ---
EXAMPLE_TEXTS: Dict[str, str] = {
    "Select an Example": "", # Placeholder - THIS MUST BE THE FIRST KEY
    "Example 1": "ربما طرقها العدو أو أسرع الجند إليها، وتداعوا نحوها حتى يكاد يترامى ذلك بهم إلى انتهاب العسكر ثوران الفتنة. فإن أهل الفتن، وسيئ السيرة، ومن همته الشرك كثير، ومسارعتهم إلى الخير بعيدة.",
    "Example 2": "فلما صح عنده بفطرته الفائقة التي تنبهت لمثل هذه الحجة؛ أن جسم السماء متناهٍ، أراد أن يعرف على أي شكل هو وكيفية انقطاعه بالسطوح التي تحده.",
    "Example 3": "قَالَ يَا قَوْمِ أَرَأَيْتُمْ إِن كُنتُ عَلَىٰ بَيِّنَةٍ مِّن رَّبِّي وَآتَانِي رَحْمَةً مِّنْ عِندِهِ فَعُمِّيَتْ عَلَيْكُمْ أَنُلْزِمُكُمُوهَا وَأَنتُمْ لَهَا كَارِهُونَ",
    # ...
}

# --- Streamlit App UI ---
st.title("📖 Ma'ani: Classical Arabic-to-English Translation & Analysis")
st.caption("By Abdelbasset Djamai")

# Initialize session state for input if it doesn't exist
if 'arabic_input' not in st.session_state:
    st.session_state.arabic_input = "" # Start empty
if 'selected_example' not in st.session_state:
    # Default to the placeholder key
    st.session_state.selected_example = list(EXAMPLE_TEXTS.keys())[0]
# Add session state for highlight type selection
if 'highlight_display_type' not in st.session_state:
    st.session_state.highlight_display_type = "None" # Default to showing no highlights
# Add session state for feedback buttons
if 'feedback_submitted' not in st.session_state:
     st.session_state.feedback_submitted = False

# Callback to update text area when example changes
def update_text_from_example():
    selected_key = st.session_state.example_selector # Get value from widget's key
    # ONLY update if a real example (not the placeholder) is selected
    if selected_key != list(EXAMPLE_TEXTS.keys())[0]:
        st.session_state.arabic_input = EXAMPLE_TEXTS.get(selected_key, "")
    # If placeholder is selected, do nothing to the text area

# --- Sidebar ---
with st.sidebar:
    st.header("Options")
    # Moved Example Selector Here
    example_options = list(EXAMPLE_TEXTS.keys())
    if 'selected_example' not in st.session_state:
        st.session_state.selected_example = example_options[0] # Default to placeholder

    # Callback to update text area when example changes
    def update_text_from_example():
        selected_key = st.session_state.example_selector
        if selected_key != example_options[0]: # Check against placeholder
            st.session_state.arabic_input = EXAMPLE_TEXTS.get(selected_key, "")
            st.session_state.selected_example = selected_key # Update selected state too
        else:
             # If placeholder selected, maybe clear the text area? Or do nothing.
             # Doing nothing allows user to select it without losing custom text.
             st.session_state.selected_example = selected_key

    st.selectbox(
        "Select Example",
        options=example_options,
        key="example_selector",
        on_change=update_text_from_example,
        index=example_options.index(st.session_state.selected_example) # Use state for index
    )

    # Clear Input Button
    if st.button("Clear Input Text"):
        st.session_state.arabic_input = ""
        # Optionally reset example selector to placeholder
        st.session_state.selected_example = example_options[0]
        st.rerun() # Rerun to reflect cleared input immediately

    st.divider()
    st.header("About")
    st.info("""
        This tool performs a multi-stage translation and analysis of Classical Arabic text.
        It leverages Generative AI (LLMs) to provide:
        - Initial & Refined Translations
        - Contextual Analysis (Genre, Tone, etc.)
        - Comparative Evaluation
        - Cultural Gap & Linguistic Nuance Identification
    """)

# --- Placeholders for dynamic updates (Moved Up) ---
# progress_area = st.empty()
# results_area = st.container()

# --- Input Area ---
# Check if orchestrator initialized successfully before showing the main app area
if translation_orchestrator:
    with st.container(border=True):
        st.subheader("Source Text")
        # available_datasets = translation_orchestrator.get_datasets() # Not needed anymore for examples
        example_options = list(EXAMPLE_TEXTS.keys())

        # Remove the col1, col2 layout here as the selector is moved
        # Inject CSS to force RTL direction for the text area
        st.markdown("""
        <style>
        /* Target the specific textarea element generated by st.text_area */
        div[data-testid="stTextArea"] textarea {
            direction: rtl !important;
            text-align: right !important; /* Add text alignment too */
        }
        </style>
        """, unsafe_allow_html=True)

        # Use the session state key for the text_area value
        st.text_area(
            "Enter Classical Arabic text here OR select an example from the sidebar:",
            height=150,
            key="arabic_input", # Bind widget to session state
            label_visibility="collapsed", # Hide default label
            placeholder="Enter the Classical Arabic source text here..."
        )
        # Display Character/Word Count (Moved here after removing columns)
        input_text_for_analysis = st.session_state.arabic_input
        char_count = len(input_text_for_analysis)
        word_count = len(input_text_for_analysis.split())
        st.caption(f"{char_count} characters | {word_count} words")

        # It reads directly from the session state bound to the text_area
        arabic_text_input = st.session_state.arabic_input

        # Only set state on button click
        translate_button_clicked = st.button("Translate & Analyze", type="primary", use_container_width=True, disabled=st.session_state.get('is_translating', False))
        if translate_button_clicked:
            if arabic_text_input:
                # Set state to trigger translation run on rerun
                st.session_state.is_translating = True
                st.session_state.text_to_process = arabic_text_input # Store the text
                st.session_state.feedback_submitted = False
                st.session_state.translation_result = None
                st.session_state.current_step = "not_started"
                st.session_state.progress_message = "Starting translation..."
                # Don't clear placeholders here, define them after this block
                st.rerun() # Trigger rerun to start translation
            else:
                st.warning("Please enter text to translate.")

    # --- Placeholders (Define AFTER input container) ---
    progress_area = st.empty()
    results_area = st.container()

else:
    # Display a clear error if services couldn't initialize
    st.error("Translation services could not be initialized. Please ensure the GOOGLE_API_KEY secret is correctly set in the Streamlit Cloud app settings and refresh the page.")
    # Define placeholders even on error to prevent NameError later if state is checked
    progress_area = st.empty()
    results_area = st.container()

# --- Translation Process and Results --- #

# Initialize session state variables
if 'translation_result' not in st.session_state:
    st.session_state.translation_result = None
if 'current_step' not in st.session_state:
    st.session_state.current_step = "not_started"
if 'progress_message' not in st.session_state:
    st.session_state.progress_message = ""
if 'is_translating' not in st.session_state:
    st.session_state.is_translating = False

# --- Progress Callback Handler ---
def handle_progress(step: TranslationStep, partial_result: Optional[TranslationResult]):
    step_messages = {
        "initial_translation": "Generating initial translation...",
        "context_analysis": "Analyzing context...",
        "refinement": "Refining translation...",
        "evaluation": "Evaluating translations...",
        "cultural_gap_analysis": "Analyzing cultural gaps...",
        "nuance_analysis": "Analyzing linguistic nuances...", # Added nuance step message
        "completed": "Translation complete!",
        "not_started": "Waiting to start...",
        "error": "Translation failed." # Add error message
    }
    print(f"Progress Update: Step={step}") # Add logging
    st.session_state.current_step = step
    st.session_state.progress_message = step_messages.get(step, "Processing...")
    if partial_result:
        st.session_state.translation_result = partial_result

# --- Translation Execution (Triggered by State) --- #
if st.session_state.get('is_translating', False):
    if not translation_orchestrator:
        st.error("Translation services unavailable. Cannot proceed.")
        st.session_state.is_translating = False # Reset state
        st.rerun()
    else:
        input_text = st.session_state.get('text_to_process', '')
        if input_text:
            input_data = InputText(arabicText=input_text)
            # Display spinner AND call sync func directly within it
            with st.spinner("Translating and analyzing..."):
                try:
                    print("Attempting translation_orchestrator.translate_with_progress...")
                    # Call the *synchronous* method directly
                    result = translation_orchestrator.translate_with_progress(input_data, handle_progress)
                    st.session_state.translation_result = result # Store the result
                    # Check result state explicitly if needed
                    if result and result.currentStep == 'completed':
                         st.session_state.current_step = "completed"
                         st.session_state.progress_message = "Translation complete!"
                         print("Translation completed successfully.")
                    else:
                         # Handle cases where orchestration finished but might have internal errors
                         st.session_state.current_step = result.currentStep if result else "error"
                         st.session_state.progress_message = f"Translation finished with status: {st.session_state.current_step}"
                         print(f"Translation finished with status: {st.session_state.current_step}")
                         if st.session_state.current_step != 'completed':
                              st.warning(f"Translation process ended with status: {st.session_state.current_step}")

                except Exception as e:
                    print(f"Error during translation execution: {e}")
                    st.error(f"Translation Error: {e}")
                    st.session_state.current_step = "error"
                    st.session_state.translation_result = None
                    st.session_state.progress_message = f"Translation failed: {e}"

            # Update state *after* the spinner context finishes
            st.session_state.is_translating = False
            st.session_state.text_to_process = None # Clear processed text
            st.rerun() # Trigger rerun to display results/errors
        else:
            st.warning("No text found to process.")
            st.session_state.is_translating = False
            st.rerun()

# --- Display Updates and Final Results --- #

# Display progress bar/spinner (Refined Logic)
# Use the state set by the callback or the execution block
current_state = st.session_state.get('current_step', 'not_started')
if st.session_state.get('is_translating', False): # Show progress if actively translating
    # (Existing progress bar logic based on current_step can remain here)
    step_list = [s for s in TranslationStep.__args__ if s not in ['completed', 'not_started', 'error']] if hasattr(TranslationStep, '__args__') else []
    max_index = len(step_list)

    progress_value = 0.0
    if current_state == "completed":
        progress_value = 1.0
    elif current_state == "error":
        progress_value = 0.0 # Or keep the last known value? Setting to 0 for error.
    elif current_state in step_list:
        try:
            # +1 because index is 0-based, steps are 1-based for progress
            current_index = step_list.index(current_state) + 1
            progress_value = current_index / max_index if max_index > 0 else 0.0
        except ValueError:
            progress_value = 0.0 # Step not found in list, should not happen
    else: # e.g., "not_started"
        progress_value = 0.0

    # Ensure progress is within bounds [0.0, 1.0]
    progress_value = max(0.0, min(1.0, progress_value))
    # Display uses the message set by the trigger or callback
    progress_area.progress(progress_value, text=st.session_state.get('progress_message', 'Processing...'))

elif current_state == "error":
    # Keep showing error message if the last attempt failed
    progress_area.error(st.session_state.get('progress_message', 'An error occurred.'))
else:
    progress_area.empty() # Clear progress bar when done or idle

# Display results progressively or when completed
# --- DEBUG: Show the raw result object ---
if st.session_state.get("translation_result"):
    st.subheader("DEBUG: Raw Translation Result Object")
    try:
        # Convert dataclass to dict before passing to st.json
        result_dict = dataclasses.asdict(st.session_state.translation_result)
        st.json(result_dict)
    except Exception as e:
        st.error(f"DEBUG: Could not serialize result object to JSON: {e}")
        # Fallback: print the object representation
        st.text(repr(st.session_state.translation_result))
    st.divider()
# --- END DEBUG ---

if st.session_state.translation_result:
    result = st.session_state.translation_result

    with results_area:
        st.divider()
        st.subheader("Translations")
        col1, col2 = st.columns(2)
        with col1:
            st.info("**Initial Translation**")
            if result.initialTranslation and result.initialTranslation.text:
                # Prepare highlights (if any) for initial text - assuming only target highlights apply to refined
                # If highlights *can* apply to initial, they need separate processing
                highlighted_initial = highlight_text(result.initialTranslation.text, []) # Pass empty list if no highlights for initial
                # Display without blockquote
                st.markdown(highlighted_initial, unsafe_allow_html=True)
                display_confidence("Initial", result.initialTranslation.confidence)
            else:
                 st.markdown("_Processing or N/A_")

        with col2:
            st.success("**Refined Translation**")
            # Add radio button for selecting highlight type
            highlight_type = st.radio(
                "Show Highlights For:",
                options=["None", "Cultural Gaps", "Linguistic Nuances"],
                key="highlight_display_type", # Bind to session state
                horizontal=True,
                label_visibility="collapsed" # Hide the label "Show Highlights For:"
            )

            if result.refinedTranslation and result.refinedTranslation.text:
                # --- Prepare highlights based on selection ---
                highlights_to_show = []
                if highlight_type == "Cultural Gaps":
                    if result.culturalGapAnalysis and result.culturalGapAnalysis.gaps:
                        for i, gap in enumerate(result.culturalGapAnalysis.gaps):
                            if gap.targetLocation: # Only use if target location exists
                                color = generate_distinct_color(i, base_hue=0)
                                tooltip = f"CULTURAL GAP ({gap.category.upper()}): {gap.description} | Strategy: {gap.translationStrategy}"
                                highlights_to_show.append((gap.targetLocation, color, tooltip))
                elif highlight_type == "Linguistic Nuances":
                    if result.refinedTranslation.linguisticNuances:
                         for i, nuance in enumerate(result.refinedTranslation.linguisticNuances):
                             if nuance.targetLocation: # Only use if target location exists
                                 color = generate_distinct_color(i, base_hue=200)
                                 tooltip = f"NUANCE ({nuance.category.upper()}): {nuance.explanation}"
                                 highlights_to_show.append((nuance.targetLocation, color, tooltip))
                # --- End highlight preparation ---

                # Sort highlights before applying
                highlights_to_show.sort(key=lambda x: x[0].start)

                # --- START DEBUGGING --- #
                # Ensure this uses st.text or st.write for live app debugging
                # print(f"DEBUG: Highlights being passed to highlight_text: {highlights_to_show}")
                st.text(f"DEBUG (Highlight Input): {highlights_to_show}")
                # --- END DEBUGGING --- #

                highlighted_refined_text = highlight_text(result.refinedTranslation.text, highlights_to_show)

                # Display without blockquote
                st.markdown(highlighted_refined_text, unsafe_allow_html=True)
                display_confidence("Refined", result.refinedTranslation.confidence)
            else:
                st.markdown("_Processing or N/A_")

        # --- Feedback Section ---
        st.divider()

        # --- NEW Row: Display Details Based on Selection --- #
        # Use the pre-processed displayable_gaps/nuances lists for details
        highlight_details_type = st.session_state.highlight_display_type

        if highlight_details_type == "Cultural Gaps":
             # Display details using the processed 'displayable_gaps'
             if result.culturalGapAnalysis and result.culturalGapAnalysis.gaps: # Check if the list was populated
                 with st.container(border=True):
                     st.markdown("##### Cultural Gap Analysis (Details)")
                     # Display overall strategy etc. from the analysis object if needed
                     if result.culturalGapAnalysis:
                          st.markdown(f"**Overall Strategy:** {result.culturalGapAnalysis.overallStrategy}")
                          st.metric("Overall Effectiveness", f"{result.culturalGapAnalysis.effectivenessRating}/10")

                     st.markdown("**Identified Gaps (with color key):**")
                     for i, gap in enumerate(result.culturalGapAnalysis.gaps):
                         gap_color = generate_distinct_color(i, base_hue=0)
                         st.markdown(f'<span style="display:inline-block; width: 12px; height: 12px; background-color:{gap_color}; border-radius: 50%; margin-right: 8px;"></span>'
                                     f'**{i+1}. {gap.name} ({gap.category.capitalize()})**', unsafe_allow_html=True)

                         # Display Source Text if location exists
                         if gap.sourceLocation and result.inputText and result.inputText.arabicText:
                             try:
                                 # Validate indices before slicing
                                 text_len = len(result.inputText.arabicText)
                                 start = gap.sourceLocation.start
                                 end = gap.sourceLocation.end
                                 if 0 <= start <= end <= text_len: # Ensure start <= end and within bounds
                                     source_snippet = result.inputText.arabicText[start:end]
                                     st.markdown(f"   - **Source Snippet:** `{source_snippet}`")
                                 else:
                                     st.caption(f"   - _Source location indices invalid ({start}-{end}) for text length {text_len}._")
                             except Exception as e:
                                 logging.warning(f"Error displaying source snippet for gap {i}: {e}")
                                 st.caption("   - _Error displaying source snippet._")
                         else:
                              st.caption("   - _Source location not available._")

                         st.markdown(f"   - **Challenge:** {gap.description}")
                         st.markdown(f"   - **Strategy:** {gap.translationStrategy}")
                         snippet_parts = []
                         # Use original result data for snippets
                         if gap.sourceLocation and result.inputText.arabicText:
                              snippet_parts.append(f"_Source: ...{result.inputText.arabicText[gap.sourceLocation.start:gap.sourceLocation.end]}..._")
                         if gap.targetLocation and result.refinedTranslation:
                              snippet_parts.append(f"_Target: ...{result.refinedTranslation.text[gap.targetLocation.start:gap.targetLocation.end]}..._")
                         if snippet_parts:
                              st.caption(" | ".join(snippet_parts))
                         st.markdown("---") # Separator

                     if result.culturalGapAnalysis and result.culturalGapAnalysis.generatedTime:
                          st.caption(f"Generated in {result.culturalGapAnalysis.generatedTime} ms")
             elif result.culturalGapAnalysis: # Analysis exists but maybe no displayable gaps (e.g., no targetLocation)
                 with st.container(border=True):
                     st.markdown("##### Cultural Gap Analysis (Details)")
                     st.info("No cultural gaps identified that could be highlighted in the text.")
                     if result.culturalGapAnalysis.generatedTime:
                          st.caption(f"Generated in {result.culturalGapAnalysis.generatedTime} ms")
             # else: show processing or N/A message if analysis itself is missing

        elif highlight_details_type == "Linguistic Nuances":
             # Display details using the processed 'displayable_nuances'
             if result.refinedTranslation and result.refinedTranslation.linguisticNuances:
                 with st.container(border=True):
                     st.markdown("##### Linguistic Nuances (Details)")
                     st.markdown("**Identified Nuances (with color key):**")
                     for i, nuance in enumerate(result.refinedTranslation.linguisticNuances):
                         nuance_color = generate_distinct_color(i, base_hue=200)
                         st.markdown(f'<span style="display:inline-block; width: 12px; height: 12px; background-color:{nuance_color}; border-radius: 50%; margin-right: 8px;"></span>'
                                     f'**{i+1}. {nuance.text} ({nuance.category.capitalize()})**', unsafe_allow_html=True)
                         st.markdown(f"   > {nuance.explanation}")
                         st.markdown("---") # Separator
             elif result.refinedTranslation and result.refinedTranslation.linguisticNuances is not None:
                 # Handle case where nuance list exists but might be empty or have no targetLocations
                 with st.container(border=True):
                      st.markdown("##### Linguistic Nuances (Details)")
                      st.info("No linguistic nuances identified that could be highlighted in the text.")
             # else: show processing or N/A message if nuance analysis is missing


        # --- Row 2 (Now Row 3): Context & Evaluation --- #
        # This section remains the same, displaying Context and Evaluation side-by-side
        if result.contextAnalysis or result.evaluation:
            col1_ctx, col2_eval = st.columns([1, 2])
            with col1_ctx:
                with st.container(border=True):
                    st.markdown("##### Context Analysis")
                    if result.contextAnalysis:
                         ctx = result.contextAnalysis
                         st.markdown(f"**Genre:** {ctx.genre}")
                         st.markdown(f"**Time Period:** {ctx.timePeriod}")
                         st.markdown(f"**Tone:** {ctx.tone}")
                         with st.expander("Key Terminology"):
                             for term in ctx.keyTerminology:
                                 st.markdown(f"- {term}")
                         with st.expander("Historical Context"):
                             st.markdown(ctx.historicalContext)
                         if ctx.generatedTime:
                              st.caption(f"Generated in {ctx.generatedTime} ms")
                    else:
                        st.markdown("_Processing..._")

            with col2_eval:
                with st.container(border=True):
                    st.markdown("##### Comparative Evaluation")
                    if result.evaluation:
                         eval_data = result.evaluation
                         st.metric("Preferred Translation", eval_data.preferredTranslation.capitalize(), f"{eval_data.preferenceConfidence}% Confidence")
                         st.markdown("**Scores (1-10):**")
                         display_evaluation_scores("Initial", eval_data.initialTranslation)
                         display_evaluation_scores("Refined", eval_data.refinedTranslation)
                         with st.expander("Detailed Assessments"):
                             st.markdown(f"**Accuracy:** {eval_data.accuracyAssessment}")
                             st.markdown(f"**Fluency:** {eval_data.fluencyAssessment}")
                             st.markdown(f"**Cultural Fidelity:** {eval_data.culturalFidelityAssessment}")
                         if eval_data.generatedTime:
                              st.caption(f"Generated in {eval_data.generatedTime} ms")
                    else:
                        st.markdown("_Processing..._")

        # --- Footer --- #
        # Add footer or final message
        if result.currentStep == 'completed':
            st.success("Translation and analysis complete.")

        # --- Download Button ---
        if result.currentStep == 'completed':
            # Convert result dataclass to dict for JSON download
            try:
                # A simple way if dataclasses module is available and result is simple
                import dataclasses
                result_dict = dataclasses.asdict(result)
            except:
                # Fallback or more robust serialization if needed
                result_dict = {"error": "Could not serialize result to dict"} # Placeholder

            st.download_button(
                label="Download Full Results (JSON)",
                data=json.dumps(result_dict, indent=4, ensure_ascii=False),
                file_name=f"translation_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )