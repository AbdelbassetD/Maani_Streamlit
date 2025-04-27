import streamlit as st
import asyncio
import colorsys # Import the colorsys module
import json # For feedback logging and download
import os # For feedback logging path
from datetime import datetime # For feedback timestamp
from typing import List, Tuple, Optional, Dict
import pandas as pd # For evaluation bar chart

# Import genai here for configuration
import google.generativeai as genai

# Updated imports for modular structure
from core.llm_client import LLMClient
from core.translation_orchestrator import TranslationOrchestrator
from shared_types import TranslationResult, InputText, TranslationStep, LinguisticNuance, CulturalGap, TextLocation

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
    if not locations or not isinstance(text, str): # Add check for text type
        return text if isinstance(text, str) else ""

    # Sort locations by start index to handle potential overlaps correctly
    locations.sort(key=lambda x: x[0].start)

    highlighted_parts = []
    last_end = 0
    for loc, color, tooltip in locations:
        # Ensure loc is valid
        if not isinstance(loc, TextLocation) or not hasattr(loc, 'start') or not hasattr(loc, 'end'):
            print(f"Skipping invalid location object: {loc}")
            continue

        start, end = loc.start, loc.end
        # Ensure valid indices relative to the current text length
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start <= end <= len(text)):
             print(f"Skipping invalid indices: start={start}, end={end}, text_len={len(text)}")
             continue # Skip invalid locations

        # Avoid overlapping highlights - if current start is before last end, adjust start
        start = max(start, last_end)
        if start >= end: # Skip if adjustment makes it invalid
             continue

        # Add text before the highlight
        highlighted_parts.append(text[last_end:start])

        # Add the highlighted segment with tooltip
        highlighted_segment = text[start:end]
        # Basic HTML structure for tooltip (Streamlit uses Markdown)
        # Escaping potential markdown/HTML characters in the tooltip
        safe_tooltip = tooltip.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\"", "&quot;").replace("'", "&#39;").replace("\n", " ") # Replace newlines too
        # Using a simple span with title for hover effect
        highlighted_parts.append(
            f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px;" title="{safe_tooltip}">{highlighted_segment}</span>'
        )
        last_end = end

    # Add any remaining text
    highlighted_parts.append(text[last_end:])

    return "".join(highlighted_parts)

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
        It leverages Google's Generative AI models to provide:
        - Initial & Refined Translations
        - Contextual Analysis (Genre, Tone, etc.)
        - Comparative Evaluation
        - Cultural Gap & Linguistic Nuance Identification
    """)

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
                # Clear display areas immediately
                results_area.empty()
                progress_area.empty()
                st.rerun() # Trigger rerun to start translation
            else:
                st.warning("Please enter text to translate.")

else:
    # Display a clear error if services couldn't initialize
    st.error("Translation services could not be initialized. Please ensure the GOOGLE_API_KEY secret is correctly set in the Streamlit Cloud app settings and refresh the page.")

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

# Placeholders for dynamic updates
progress_area = st.empty()
results_area = st.container()

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
# This block runs *after* the rerun triggered by the button click
if st.session_state.get('is_translating', False):
    # Ensure orchestrator is available (it should be if we got here)
    if not translation_orchestrator:
        st.error("Translation services unavailable. Cannot proceed.")
        st.session_state.is_translating = False # Reset state
        st.rerun()
    else:
        input_text = st.session_state.get('text_to_process', '')
        if input_text:
            input_data = InputText(arabicText=input_text)
            # Display spinner while running
            with st.spinner("Translating and analyzing..."):
                try:
                    print("Attempting translation_orchestrator.translate_with_progress...") # Add logging
                    # Call the async function directly - Streamlit handles the await
                    # Use the cached orchestrator instance
                    st.session_state.translation_result = translation_orchestrator.translate_with_progress(input_data, handle_progress)
                    st.session_state.current_step = "completed" # Explicitly set completion
                    st.session_state.progress_message = "Translation complete!"
                    print("Translation completed successfully.")
                except Exception as e:
                    print(f"Error during translation execution: {e}") # Log error
                    st.error(f"Translation Error: {e}")
                    st.session_state.current_step = "error"
                    st.session_state.translation_result = None
                    st.session_state.progress_message = "Translation failed."
                finally:
                    # Always mark as not translating anymore and trigger UI update
                    st.session_state.is_translating = False
                    st.session_state.text_to_process = None # Clear processed text
                    st.rerun()
        else:
            # Should not happen if button logic is correct, but handle defensively
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
if st.session_state.translation_result:
    result = st.session_state.translation_result

    with results_area:
        st.divider()
        st.subheader("Translation & Analysis Results")

        # --- Row 1: Initial vs Refined Translations (No Details Here) --- #
        with st.container(border=True):
            st.markdown("##### Initial & Refined Translations")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Initial Translation**")
                if result.initialTranslation:
                    st.markdown(f"> {result.initialTranslation.text}")
                    display_confidence("Initial", result.initialTranslation.confidence)
                else:
                    st.markdown("_Processing..._")

            with col2:
                st.markdown("**Refined Translation (Context-Aware)**")

                # Add radio button for selecting highlight type
                highlight_type = st.radio(
                    "Show Highlights For:",
                    options=["None", "Cultural Gaps", "Linguistic Nuances"],
                    key="highlight_display_type", # Bind to session state
                    horizontal=True,
                    label_visibility="collapsed" # Hide the label "Show Highlights For:"
                )

                # --- Prepare data structures for consistent display ---
                displayable_gaps = []
                displayable_nuances = []
                highlights_to_show = []

                if result.refinedTranslation:
                    # Process Nuances for consistent display
                    if highlight_type == "Linguistic Nuances":
                        raw_nuances = result.refinedTranslation.linguisticNuances or []
                        for i, nuance in enumerate(raw_nuances):
                            color = generate_distinct_color(i, base_hue=200)
                            tooltip = f"NUANCE ({nuance.category.upper()}): {nuance.explanation}"
                            displayable_nuances.append({
                                "index": i,
                                "color": color,
                                "tooltip": tooltip,
                                "data": nuance
                            })
                            # Add to highlights list only if target location exists
                            if nuance.targetLocation:
                                highlights_to_show.append((nuance.targetLocation, color, tooltip))

                    # Process Cultural Gaps for consistent display
                    elif highlight_type == "Cultural Gaps":
                        raw_gaps = result.culturalGapAnalysis.gaps if result.culturalGapAnalysis else []
                        for i, gap in enumerate(raw_gaps):
                            color = generate_distinct_color(i, base_hue=0) # Different base hue
                            tooltip = f"CULTURAL GAP ({gap.category.upper()}): {gap.description} | Strategy: {gap.translationStrategy}"
                            displayable_gaps.append({
                                "index": i,
                                "color": color,
                                "tooltip": tooltip,
                                "data": gap
                            })
                            # Add to highlights list only if target location exists
                            if gap.targetLocation:
                                highlights_to_show.append((gap.targetLocation, color, tooltip))

                    # Sort highlights by start index before applying (important for correct rendering)
                    highlights_to_show.sort(key=lambda x: x[0].start)

                    # Apply highlights to the text
                    highlighted_refined_text = highlight_text(result.refinedTranslation.text, highlights_to_show)

                    # Display refined text
                    st.markdown(f"> {highlighted_refined_text}", unsafe_allow_html=True)
                    display_confidence("Refined", result.refinedTranslation.confidence)
                else:
                    st.markdown("_Processing..._")

        # --- NEW Row: Display Details Based on Selection --- #
        # Use the pre-processed displayable_gaps/nuances lists for details
        highlight_details_type = st.session_state.highlight_display_type

        if highlight_details_type == "Cultural Gaps":
             # Display details using the processed 'displayable_gaps'
             if displayable_gaps: # Check if the list was populated
                 with st.container(border=True):
                     st.markdown("##### Cultural Gap Analysis (Details)")
                     # Display overall strategy etc. from the analysis object if needed
                     if result.culturalGapAnalysis:
                          st.markdown(f"**Overall Strategy:** {result.culturalGapAnalysis.overallStrategy}")
                          st.metric("Overall Effectiveness", f"{result.culturalGapAnalysis.effectivenessRating}/10")

                     st.markdown("**Identified Gaps (with color key):**")
                     for item in displayable_gaps:
                         gap = item['data'] # Get the original gap object
                         gap_color = item['color'] # Use the pre-calculated color
                         st.markdown(f'<span style="display:inline-block; width: 12px; height: 12px; background-color:{gap_color}; border-radius: 50%; margin-right: 8px;"></span>'
                                     f'**{item["index"]+1}. {gap.name} ({gap.category.capitalize()})**', unsafe_allow_html=True)
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
             if displayable_nuances:
                 with st.container(border=True):
                     st.markdown("##### Linguistic Nuances (Details)")
                     st.markdown("**Identified Nuances (with color key):**")
                     for item in displayable_nuances:
                         nuance = item['data']
                         nuance_color = item['color']
                         st.markdown(f'<span style="display:inline-block; width: 12px; height: 12px; background-color:{nuance_color}; border-radius: 50%; margin-right: 8px;"></span>'
                                     f'**{item["index"]+1}. {nuance.text} ({nuance.category.capitalize()})**', unsafe_allow_html=True)
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