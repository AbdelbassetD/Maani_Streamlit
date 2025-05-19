# Ma'ani: Classical Arabic Translation & Analysis (Python/Streamlit Version)

This application provides a multi-step translation and analysis of Classical Arabic text using Google's Generative AI.

## Features

-   **Initial Translation:** Generates a baseline English translation.
-   **Context Analysis:** Determines genre, time period, tone, key terms, and historical context.
-   **Refined Translation:** Improves the initial translation based on the analyzed context.
-   **Comparative Evaluation:** Scores and compares the initial and refined translations on accuracy, fluency, nuance, and cultural fidelity.
-   **Cultural Gap Analysis:** Identifies potential cultural discrepancies (_gaps_) and the strategies used to bridge them.
-   **Linguistic Nuances:** Explains specific terms or phrases in the refined translation that have cultural or linguistic significance (with highlighting).
-   **Interactive UI:** Built with Streamlit for ease of use.

## Setup

1. **Clone the repository (or ensure you have the files):**

    ```bash
    # If applicable
    # git clone <repository-url>
    # cd <repository-directory>
    ```

2. **Create a Python virtual environment (recommended):**

    ```bash
    python -m venv venv
    # Activate it:
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Set up your API Key:**

    - Create a file named `.env` in the project root.
    - Add your Google Generative AI API key to it:
        ```.env
        GOOGLE_API_KEY=YOUR_API_KEY_HERE
        ```
    - Replace `YOUR_API_KEY_HERE` with your actual key.

## Running the App

1. Make sure your virtual environment is activated.
2. Run the Streamlit app from your terminal:
    ```bash
    streamlit run app.py
    ```
3. The application should open in your web browser.

## Project Structure

-   `app.py`: The main Streamlit application file (UI and flow control).
-   `requirements.txt`: Lists Python dependencies.
-   `shared_types.py`: Defines Python data classes for structuring translation results.
-   `.env`: Stores the API key (should not be committed to Git).
-   `README.md`: This file.
-   `config/`: Contains configuration files.
    -   `__init__.py`
    -   `settings.py`: Model IDs, generation parameters.
    -   `prompts.py`: LLM prompt templates.
-   `core/`: Contains the core application logic.
    -   `__init__.py`
    -   `llm_client.py`: Base class for interacting with the Google AI API.
    -   `translation_orchestrator.py`: Class that manages the overall translation workflow.
-   `components/`: Contains modules for each specific step in the translation process.
    -   `__init__.py`
    -   `initial_translator.py`
    -   `context_analyzer.py`
    -   `refiner.py`
    -   `evaluator.py`
    -   `cultural_analyzer.py`
    -   `nuance_analyzer.py`
    -   `confidence_calculator.py` (Non-LLM step)
-   `utils/`: Contains helper functions.
    -   `__init__.py`
    -   `helpers.py`: JSON parsing, text matching utilities.
