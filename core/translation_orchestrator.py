import time
import logging
from typing import List, Optional, Callable

from core.llm_client import LLMClient
from shared_types import (
    InputText, TranslationResult, Dataset, TranslationConfidence,
    TranslationStep, InitialTranslation, ContextAnalysis,
    LinguisticNuance, RefinedTranslation, TranslationEvaluation,
    CulturalGapAnalysis
)
# Import component functions
from components.initial_translator import generate_initial_translation
from components.context_analyzer import generate_context_analysis, DEFAULT_CONTEXT_ANALYSIS
from components.refiner import generate_refined_translation
from components.evaluator import generate_translation_evaluation, FALLBACK_EVALUATION
from components.cultural_analyzer import generate_cultural_gap_analysis, FALLBACK_CULTURAL_GAPS
from components.nuance_analyzer import generate_linguistic_nuances, FALLBACK_LINGUISTIC_NUANCES
from components.confidence_calculator import generate_reliable_confidence

# Import sample data (can be moved to config or data module)
SAMPLE_DATASETS: List[Dataset] = [
    Dataset(id='arabic-english-parallel', name='Arabic-English Parallel Corpus'),
    Dataset(id='classical-arabic-literature', name='Classical Arabic Literature'),
    Dataset(id='academic-texts', name='Academic Texts'),
    Dataset(id='religious-texts', name='Religious Texts'),
]

class TranslationOrchestrator:
    """Orchestrates the multi-step translation process using modular components."""
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        if not self.llm_client or not self.llm_client.model:
             logging.warning("TranslationOrchestrator initialized without a functional LLMClient. Operations requiring LLM will use fallbacks.")

    def get_datasets(self) -> List[Dataset]:
        """Get available datasets (simulated)."""
        return SAMPLE_DATASETS

    async def simulate_translation(self, arabic_text: str, dataset: Optional[str] = None) -> TranslationResult:
        """Simulates a full translation result, using components with fallbacks where needed."""
        logging.info(f"Simulating full translation for: {arabic_text[:30]}...")
        input_data = InputText(arabicText=arabic_text, dataset=dataset)
        result = TranslationResult(inputText=input_data, currentStep='not_started')

        # Simulate initial translation (might use LLM if available but primarily fallback)
        start_time = time.time()
        sim_initial_text = await generate_initial_translation(self.llm_client, arabic_text)
        sim_initial_conf = generate_reliable_confidence(arabic_text, sim_initial_text, False)
        result.initialTranslation = InitialTranslation(
            text=sim_initial_text,
            generatedTime=int((time.time() - start_time) * 1000),
            confidence=sim_initial_conf
        )

        # Use real context analysis if possible, else default
        start_time = time.time()
        result.contextAnalysis = await generate_context_analysis(self.llm_client, arabic_text)
        result.contextAnalysis.generatedTime = int((time.time() - start_time) * 1000)

        # Use real refined translation if possible, else simulate
        start_time = time.time()
        sim_refined_text = await generate_refined_translation(
            self.llm_client, arabic_text, result.initialTranslation.text, result.contextAnalysis
        )
        sim_refined_conf = generate_reliable_confidence(arabic_text, sim_refined_text, True)
        sim_nuances = await generate_linguistic_nuances(self.llm_client, arabic_text, sim_refined_text, result.contextAnalysis)
        result.refinedTranslation = RefinedTranslation(
            text=sim_refined_text,
            generatedTime=int((time.time() - start_time) * 1000),
            confidence=sim_refined_conf,
            linguisticNuances=sim_nuances
        )

        # Use real evaluation if possible, else fallback
        start_time = time.time()
        result.evaluation = await generate_translation_evaluation(
            self.llm_client, arabic_text, result.initialTranslation.text, result.refinedTranslation.text, result.contextAnalysis
        )
        result.evaluation.generatedTime = int((time.time() - start_time) * 1000)

        # Use real cultural gaps if possible, else fallback
        start_time = time.time()
        result.culturalGapAnalysis = await generate_cultural_gap_analysis(
            self.llm_client, arabic_text, result.refinedTranslation.text, result.contextAnalysis
        )
        result.culturalGapAnalysis.generatedTime = int((time.time() - start_time) * 1000)

        result.currentStep = 'completed'
        logging.info("Simulation complete.")
        return result

    def translate_with_progress(
        self,
        input_data: InputText,
        progress_callback: Optional[Callable[[TranslationStep, Optional[TranslationResult]], None]] = None
    ) -> TranslationResult:
        """Runs the full translation pipeline step-by-step."""

        def _update_progress(step: TranslationStep, current_result: TranslationResult):
            if progress_callback:
                try:
                    # Ensure callback receives a complete TranslationResult object
                    progress_callback(step, current_result)
                except Exception as e:
                    logging.error(f"Error in progress callback for step {step}: {e}")

        result = TranslationResult(inputText=input_data, currentStep='not_started')
        start_time_total = time.time()
        _update_progress(result.currentStep, result)

        try:
            logging.info(f"Starting translation orchestrator for: {input_data.arabicText[:50]}...")

            # Step 1: Initial Translation
            result.currentStep = 'initial_translation'
            _update_progress(result.currentStep, result)
            start_step_time = time.time()
            initial_text = generate_initial_translation(self.llm_client, input_data.arabicText)
            if initial_text is None: raise ValueError("Initial translation failed critically.")
            initial_confidence = generate_reliable_confidence(input_data.arabicText, initial_text, False)
            result.initialTranslation = InitialTranslation(
                text=initial_text,
                generatedTime=int((time.time() - start_step_time) * 1000),
                confidence=initial_confidence
            )
            logging.info(f"Step 1 (Initial) took {result.initialTranslation.generatedTime} ms")
            _update_progress(result.currentStep, result)

            # Step 2: Context Analysis
            result.currentStep = 'context_analysis'
            _update_progress(result.currentStep, result)
            start_step_time = time.time()
            result.contextAnalysis = generate_context_analysis(self.llm_client, input_data.arabicText)
            if result.contextAnalysis.generatedTime is None: # Assign time if component didn't
                 result.contextAnalysis.generatedTime = int((time.time() - start_step_time) * 1000)
            logging.info(f"Step 2 (Context) took {result.contextAnalysis.generatedTime} ms. Genre: {result.contextAnalysis.genre}")
            _update_progress(result.currentStep, result)

            # Step 3: Refinement
            result.currentStep = 'refinement'
            _update_progress(result.currentStep, result)
            start_step_time = time.time()
            refined_text = generate_refined_translation(
                self.llm_client, input_data.arabicText, result.initialTranslation.text, result.contextAnalysis
            )
            refined_confidence = generate_reliable_confidence(input_data.arabicText, refined_text, True)
            linguistic_nuances = generate_linguistic_nuances(
                 self.llm_client, input_data.arabicText, refined_text, result.contextAnalysis
            )
            result.refinedTranslation = RefinedTranslation(
                text=refined_text,
                generatedTime=int((time.time() - start_step_time) * 1000), # Combined time for refine+nuance
                confidence=refined_confidence,
                linguisticNuances=linguistic_nuances
            )
            logging.info(f"Step 3 (Refine+Nuance) took {result.refinedTranslation.generatedTime} ms")
            _update_progress(result.currentStep, result)

            # Step 4: Evaluation
            result.currentStep = 'evaluation'
            _update_progress(result.currentStep, result)
            start_step_time = time.time()
            result.evaluation = generate_translation_evaluation(
                self.llm_client, input_data.arabicText, result.initialTranslation.text, result.refinedTranslation.text, result.contextAnalysis
            )
            if result.evaluation.generatedTime is None:
                 result.evaluation.generatedTime = int((time.time() - start_step_time) * 1000)
            logging.info(f"Step 4 (Evaluation) took {result.evaluation.generatedTime} ms. Preferred: {result.evaluation.preferredTranslation}")
            _update_progress(result.currentStep, result)

            # Step 5: Cultural Gap Analysis
            result.currentStep = 'cultural_gap_analysis'
            _update_progress(result.currentStep, result)
            start_step_time = time.time()
            result.culturalGapAnalysis = generate_cultural_gap_analysis(
                self.llm_client, input_data.arabicText, result.refinedTranslation.text, result.contextAnalysis
            )
            if result.culturalGapAnalysis.generatedTime is None:
                 result.culturalGapAnalysis.generatedTime = int((time.time() - start_step_time) * 1000)
            logging.info(f"Step 5 (Cultural Gap) took {result.culturalGapAnalysis.generatedTime} ms. Gaps: {len(result.culturalGapAnalysis.gaps)}")
            _update_progress(result.currentStep, result)

            # --- Completion ---
            result.currentStep = 'completed'
            total_time = time.time() - start_time_total
            logging.info(f"Translation orchestration completed in {total_time:.2f}s")
            _update_progress(result.currentStep, result)
            return result

        except Exception as e:
            logging.error(f"Translation orchestration failed: {e}", exc_info=True)
            # Set step to error
            result.currentStep = 'error'
            # Try to update progress one last time if possible
            try:
                 _update_progress(result.currentStep, result)
            except Exception as callback_err:
                 logging.error(f"Error in final error progress callback: {callback_err}")
            # Re-raise the exception to be caught by the caller (app.py)
            raise e
            # --- Old logic: returning partial result ---
            # result.currentStep = 'completed' # Mark as completed even on error
            # _update_progress(result.currentStep, result) # Update UI
            # Return the partially filled result object for debugging/display
            # Or potentially run full simulation as fallback:
            # logging.warning("Falling back to simulation due to error.")
            # return await self.simulate_translation(input_data.arabicText, input_data.dataset)
            # return result # Return partial result on error