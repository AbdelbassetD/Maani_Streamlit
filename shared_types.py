from dataclasses import dataclass, field
from typing import List, Optional, Literal

TranslationStep = Literal[
    "not_started",
    "initial_translation",
    "context_analysis",
    "refinement",
    "evaluation",
    "cultural_gap_analysis",
    "completed",
]

@dataclass
class Dataset:
    id: str
    name: str

@dataclass
class InputText:
    arabicText: str
    dataset: Optional[str] = None

@dataclass
class TextLocation:
    start: int
    end: int

@dataclass
class ConfidenceSegment:
    text: str
    start: int
    end: int
    confidence: float

@dataclass
class TranslationConfidence:
    overall: float
    segments: List[ConfidenceSegment] = field(default_factory=list)

@dataclass
class InitialTranslation:
    text: str
    generatedTime: Optional[int] = None
    confidence: Optional[TranslationConfidence] = None

@dataclass
class ContextAnalysis:
    # Required fields first
    genre: str
    timePeriod: str
    tone: str
    historicalContext: str
    # Fields with defaults/Optionals last
    keyTerminology: List[str] = field(default_factory=list)
    generatedTime: Optional[int] = None

@dataclass
class LinguisticNuance:
    text: str
    explanation: str
    category: str
    targetLocation: Optional[TextLocation] = None
    sourceLocation: Optional[TextLocation] = None

@dataclass
class RefinedTranslation:
    text: str
    generatedTime: Optional[int] = None
    confidence: Optional[TranslationConfidence] = None
    linguisticNuances: List[LinguisticNuance] = field(default_factory=list)

@dataclass
class EvaluationScore:
    accuracy: int
    fluency: int
    nuance: int
    culturalFidelity: int

@dataclass
class TranslationEvaluation:
    # Required fields first
    initialTranslation: EvaluationScore
    refinedTranslation: EvaluationScore
    preferredTranslation: Literal["initial", "refined"]
    preferenceConfidence: int # Percentage 0-100
    accuracyAssessment: str
    fluencyAssessment: str
    culturalFidelityAssessment: str
    # Optional fields last
    generatedTime: Optional[int] = None

@dataclass
class CulturalGap:
    name: str
    category: str
    description: str
    translationStrategy: str
    sourceLocation: Optional[TextLocation] = None
    targetLocation: Optional[TextLocation] = None

@dataclass
class CulturalGapAnalysis:
    # Required fields first
    overallStrategy: str
    effectivenessRating: int # Score 1-10
    # Fields with defaults last
    gaps: List[CulturalGap] = field(default_factory=list)
    generatedTime: Optional[int] = None

@dataclass
class TranslationResult:
    inputText: InputText
    initialTranslation: Optional[InitialTranslation] = None
    contextAnalysis: Optional[ContextAnalysis] = None
    refinedTranslation: Optional[RefinedTranslation] = None
    evaluation: Optional[TranslationEvaluation] = None
    culturalGapAnalysis: Optional[CulturalGapAnalysis] = None
    currentStep: TranslationStep = "not_started"