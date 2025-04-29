# Ma'ani: Classical Arabic-to-English Translation & Analysis System

## 1. Core Purpose

Ma'ani is a web-based system designed to provide high-fidelity translations of Classical Arabic text into English. Beyond simple translation, its primary goal is to bridge potential comprehension gaps by offering deep linguistic and cultural analysis, making the nuances and context of the source text more accessible to the English reader. It leverages Large Language Models (LLMs) for both translation and the complex analytical tasks.

## 2. Key Features & Logic

The system integrates several features, managed through a multi-stage process orchestrated behind the scenes:

### 2.1. Multi-Stage Translation

-   **Initial Translation:** Generates a direct, competent translation of the source Arabic text.
    -   _Logic:_ An initial prompt is sent to the LLM with a straightforward instruction: "Translate the following Classical Arabic text to English". The system expects a plain text English translation as output.
-   **Refined Translation:** Produces an improved translation that aims to incorporate better contextual understanding, potentially addressing issues or complexities identified during intermediate analysis steps (though the current implementation primarily focuses refinement based on the initial pass and general instructions).
    -   _Logic:_ A subsequent prompt provides the LLM with the original source text _and_ the initial translation. The instructions ask the LLM to refine the initial translation, focusing on improving fluency, accuracy, preserving the classical tone/style where appropriate, and ensuring contextual coherence. The expected output is the refined plain text English translation.

### 2.2. Context Analysis

-   **Function:** Identifies key contextual elements of the source text.
    -   _Genre:_ (e.g., Religious scripture, Historical narrative, Philosophical treatise)
    -   _Time Period:_ Estimated historical era of the text.
    -   _Tone:_ (e.g., Formal, Didactic, Polemical)
    -   _Key Terminology:_ Lists significant terms specific to the text's domain.
    -   _Historical Context:_ Provides a brief overview of the relevant historical background.
-   _Logic:_ The prompt instructs the LLM to analyze the source Arabic text and act as a historical/literary analyst. It explicitly requests the extraction of the specific fields listed above (Genre, Time Period, Tone, Key Terminology, Historical Context). The prompt likely specifies a desired output format (e.g., JSON or structured key-value pairs) to ensure reliable parsing. The system then parses this structured response into the `ContextAnalysis` Python dataclass object for later display.

### 2.3. Comparative Evaluation

-   **Function:** Assesses the quality of both the Initial and Refined translations side-by-side.
    -   _Metrics:_ Scores (1-10) for Accuracy, Fluency, Nuance Preservation, and Cultural Fidelity.
    *   _Preference:_ Indicates which translation (Initial or Refined) is deemed superior, with a confidence score.
    *   _Qualitative Assessment:_ Provides brief textual justifications for the scores across different categories.
-   _Logic:_ The prompt engineers the LLM to act as an expert translation evaluator. It provides the source text, the initial translation, and the refined translation. The instructions ask the LLM to compare the two translations against the source based on the predefined criteria (Accuracy, Fluency, Nuance, Cultural Fidelity), assign a score from 1 to 10 for each criterion for _both_ translations, provide brief textual assessments justifying these scores, and finally, state which translation it prefers overall with an estimated confidence percentage. A structured output format (like JSON) is requested to capture these multiple pieces of information reliably. This output is then parsed into the `Evaluation` dataclass object.

### 2.4. Cultural Gap Analysis

-   **Function:** Identifies concepts, idioms, or phrases in the source text that may lack direct cultural equivalents or could be easily misunderstood by the target audience.
    -   _Output per Gap:_ Name, Category (e.g., Historical event, Religious concept, Social custom), Description of the gap, suggested Translation Strategy, and `TextLocation` (start/end character indices) in both the source Arabic and the refined English translation.
-   _Logic:_ The prompt instructs the LLM to perform a cross-cultural analysis comparing the source Arabic text and the _refined_ English translation. It asks the LLM to identify specific segments where cultural differences might lead to misunderstandings. Crucially, the prompt demands a structured output (e.g., a JSON list of gap objects). For each identified gap, the LLM must provide the specified fields: Name, Category, Description, Translation Strategy, and the precise start/end character indices (`TextLocation`) marking the relevant segment in _both_ the original Arabic source text and the refined English translation. The system relies on parsing this structured JSON output into a list of `CulturalGap` objects. The accurate `TextLocation` data is essential for the interactive highlighting feature.

### 2.5. Linguistic Nuance Analysis

-   **Function:** Highlights subtle linguistic features within the _refined_ English translation that contribute significantly to the meaning or style but might require further explanation.
    -   _Output per Nuance:_ The specific text segment (`text`), its Category (e.g., Rhetorical device, Idiomatic expression, Grammatical structure), an Explanation, and its `TextLocation` in the refined translation.
-   _Logic:_ This prompt focuses the LLM on the _refined_ English translation only. It asks the LLM to act as a linguistic analyst and identify segments containing notable nuances (like rhetoric, specific word choices, complex grammar). Similar to the cultural gap analysis, the prompt requires a structured output (e.g., a JSON list of nuance objects). For each nuance, the LLM must return the identified text segment itself, a Category, a clear Explanation of the nuance, and the precise `TextLocation` (start/end character indices) within the refined English translation. This structured data is parsed into a list of `LinguisticNuance` objects, with the `TextLocation` being vital for the highlighting feature.

### 2.6. Interactive Highlighting

-   **Function:** Visually marks the segments corresponding to identified Cultural Gaps or Linguistic Nuances within the displayed refined English translation. Users can select which type of analysis to highlight.
    -   _Visuals:_ Uses distinct background colors for gaps and nuances.
    -   _Interactivity:_ Hovering over a highlighted segment reveals a tooltip containing the detailed description/explanation.
-   _Logic:_
    1.  The user selects "Cultural Gaps" or "Linguistic Nuances" via radio buttons, updating the session state.
    2.  Based on the selection, the system retrieves the list of corresponding analysis objects (Gaps or Nuances) and their `TextLocation` data (start/end character indices) for the _refined translation_.
    3.  The locations are validated (must be within text bounds, start < end).
    4.  Locations are sorted (primarily by start index, then descending by end index to handle nesting correctly).
    5.  The system iterates through the sorted locations. For each valid, non-overlapping location, it constructs an HTML `<span style="background-color: ...">...</span>` tag containing the text segment from the refined translation. The `title` attribute of the span is populated with the gap/nuance description/explanation for the tooltip.
    6.  These spans are carefully inserted into the original text string, replacing the plain text segments. Non-highlighted text portions are preserved.
    7.  The final HTML string is displayed using Streamlit's `st.markdown(..., unsafe_allow_html=True)`.

### 2.7. Editable Gap Details

-   **Function:** Allows the user to modify the textual descriptions associated with identified Cultural Gaps (Name, Category, Description, Strategy) after the initial analysis is complete.
-   _Logic:_
    1.  Upon successful completion of the `cultural_gap_analysis` step, the list of identified `CulturalGap` objects is copied into a list of dictionaries stored in `st.session_state['editable_gaps']`. This decouples the editable data from the original, immutable result.
    2.  A toggle button (`st.toggle("Enable Editing", key="edit_mode_enabled")`) controls the UI display.
    3.  If `edit_mode_enabled` is `False`, the gap details are displayed as read-only text.
    4.  If `edit_mode_enabled` is `True`, the UI dynamically renders `st.text_input` and `st.text_area` widgets for each field of each gap in the `editable_gaps` list. The `value` of these widgets is bound to the corresponding dictionary keys in the session state list. Any changes made by the user directly update the dictionaries within the session state.
    5.  The original `TextLocation` data associated with the gaps is preserved but not made editable.

### 2.8. User Interface & Workflow Management

-   **Function:** Provides the primary user interaction layer.
    -   Input: Text area for Arabic input, example text selector, clear button.
    -   Control: "Translate & Analyze" button.
    -   Display: Organizes results into sections (Translations, Context, Evaluation, Gap/Nuance Details), manages highlight selection (radio buttons), provides edit toggle.
    -   Feedback: Positive/Negative buttons.
    -   Download: Button to get results as JSON.
-   _Logic:_ Built using the Streamlit framework.
    -   `st.session_state` is critical for managing the application's state: storing the input text, the current stage of processing (`current_step`), the full `TranslationResult` object once complete, the user's highlight preference (`highlight_display_type`), the edit mode status (`edit_mode_enabled`), the editable gap data (`editable_gaps`), and flags to trigger processing (`is_translating`).
    -   The UI dynamically updates based on changes in session state (e.g., showing results only when `translation_result` is populated, switching between read-only and edit modes for gaps).
    -   Button clicks modify session state variables, often triggering a `st.rerun()` to refresh the UI and execute conditional logic based on the new state.

### 2.9. Progress Indication

-   **Function:** Informs the user about the current stage of the backend processing.
-   _Logic:_ The `TranslationOrchestrator` calls a progress handler function (`handle_progress`) at each step. This handler updates `st.session_state.current_step` and `st.session_state.progress_message`. The UI reads these state variables to display a `st.progress` bar and associated text message.

### 2.10. Feedback Mechanism

-   **Function:** Collects simple user feedback on translation quality.
-   _Logic:_ Feedback buttons (Thumbs Up/Down) trigger a `log_feedback` function. This function writes a JSON object containing the timestamp, input text, translations, and feedback type ("positive" or "negative") as a new line to a local file (`feedback_log.jsonl`).

### 2.11. Results Download

-   **Function:** Allows the user to save the complete analysis results.
-   _Logic:_ The final `TranslationResult` dataclass object is converted into a Python dictionary using `dataclasses.asdict`. This dictionary is then serialized into a JSON formatted string. A `st.download_button` is configured with this JSON string as its data payload.

## 3. Technical Aspects

-   **Language:** Python
-   **Core Library:** Streamlit (for UI and application structure)
-   **LLM Integration:** `google-generativeai` library interacting with Google's Generative AI models (via a custom `LLMClient` abstraction).
-   **Orchestration:** A custom `TranslationOrchestrator` class manages the sequence of LLM calls for the different analysis stages and aggregates the results.
-   **Data Modeling:** Python `dataclasses` are used extensively (`shared_types.py`) to define structured objects for inputs (`InputText`) and results (`TranslationResult`, `InitialTranslation`, `RefinedTranslation`, `ContextAnalysis`, `Evaluation`, `CulturalGap`, `LinguisticNuance`, `TextLocation`, etc.), ensuring type safety and clear data contracts between components.
-   **Deployment Target:** Streamlit Cloud (utilizing `st.secrets` for API key management).
-   **Modularity:** Code is organized into logical components (e.g., `core` for backend logic, `utils` for helpers, `shared_types` for data structures, `app.py` for the UI).
