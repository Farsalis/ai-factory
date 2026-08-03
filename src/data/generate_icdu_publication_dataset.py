"""Publication-ready ICDU dataset generation with advanced augmentation.

This module generates Intent-Conscious Data Unit (ICDU) datasets from source JSONL
with data augmentation, diversity evaluation, and visualization.

Features:
    - Parallel processing for efficient dataset generation
    - Advanced perturbation and paraphrasing techniques
    - Human-in-the-loop (HITL) simulation
    - Diversity metrics and visualization
    - Configurable augmentation strategies per persona archetype
    - Scenario-perturbation method for combinatorial augmentation

The module processes source entries and converts them to ICDU format with:
    - Persona archetype inference
    - Governing principle assignment
    - Capability layer classification
    - Context-aware response generation
    - Multiple augmentation strategies

Usage:
    python generate_icdu_publication_dataset.py \\
        --input-file source_data.jsonl \\
        --output-dir output/ \\
        --augmentation-factor 20 \\
        --validation-split 0.15 \\
        --num-processes 8
"""

import argparse
import json
import logging
import random
import re
import uuid
from collections import Counter
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

# Default values
DEFAULT_AUGMENTATION_FACTOR = 20
DEFAULT_VALIDATION_SPLIT = 0.15
DEFAULT_PERTURBATION_DEPTH = 2
DEFAULT_HITL_RATIO = 0.1
DEFAULT_DIVERSITY_THRESHOLD = 0.7
DEFAULT_RANDOM_STATE = 42
DEFAULT_CHAPTER_WEIGHTS = {"Chapter 1": 1.0, "Chapter 2": 1.0, "Chapter 3": 1.0}

# Default persona for unmatched queries
DEFAULT_PERSONA = "General User > Problem Solver"

# Capability layers
CAPABILITY_LAYERS = ["Foundational", "Transformational", "Aspirational"]

# Chapter principles mapping
CHAPTER_PRINCIPLES = {
    "Chapter 1": ["People", "Clarity", "Transparency"],
    "Chapter 2": ["People", "Process", "Tools"],
    "Chapter 3": ["Foundational", "Transformational", "Aspirational"],
}

# Output file names
TRAIN_FILE_NAME = "icdu_training_data_augmented.jsonl"
VAL_FILE_NAME = "icdu_validation_data_augmented.jsonl"
EVAL_METRICS_FILE = "evaluation_metrics.json"

# ============================================================================
# Persona Pattern Definitions
# ============================================================================

PERSONA_PATTERNS: dict[str, str] = {
    "Fitness Seeker > Struggling Starter": (
        r"\b(get fit|workout|diet|exercise|gym|health|fitness|run|swim|weight)\b"
    ),
    "Relationship Manager > Financial Planner": (
        r"\b(money|financial|budget|spouse|partner|debt|save|invest|bill)\b"
    ),
    "Creative Professional > Blocked Writer": (
        r"\b(writer|blocked|creative|novel|script|inspiration|art|draw|paint|music)\b"
    ),
    "Career Changer > Aspiring Professional": (
        r"\b(learn to code|career|promotion|job|switch|skill|profession|"
        r"resume|interview|network)\b"
    ),
    "Business Professional > Team Leader": (
        r"\b(client|team|manager|boss|meeting|leadership|colleague|project|delegate|morale)\b"
    ),
    "Social Host > First-Time Planner": (
        r"\b(dinner party|hosting|event|guests|social|entertain|plan|invite|menu)\b"
    ),
    "Content Creator > AI Adopter": (
        r"\b(ai for content|ai writing|blog|video|generate|tool|content|"
        r"edit|post|seo)\b"
    ),
    "Parent > Academic Supporter": (
        r"\b(child|parent|teenager|homework|school|kid|education|tutor|grade|study)\b"
    ),
    "Team Member > Conflict Navigator": (
        r"\b(feedback|work|colleague|conflict|dispute|navigate|resolve|mediate|argue)\b"
    ),
    "Style Seeker > Budget-Conscious Shopper": (
        r"\b(wardrobe|style|fashion|outfit|shop|budget|clothes|accessory|minimalist)\b"
    ),
    "Media Creator > Tech-Novice Creator": (
        r"\b(podcast|newsletter|media|audience|create|launch|episode|subscribe|market)\b"
    ),
    "Elderly User > Tech Adopter": (
        r"\b(tech|app|device|elderly|senior|learn|adopt|gadget|smartphone|tablet)\b"
    ),
    "Student > Overachiever": (
        r"\b(student|exam|study|overachiever|school|test|grades|college|homework|procrastinate)\b"
    ),
    "Executive > Strategic Planner": (
        r"\b(strategy|executive|plan|vision|business|decision|lead|goal|prioritize|forecast)\b"
    ),
    "Entrepreneur > Startup Founder": (
        r"\b(startup|business|founder|idea|pitch|fund|launch|scale|investor|pivot)\b"
    ),
}

# ============================================================================
# Intent and Context Mappings
# ============================================================================

USER_INTENTS: dict[str, str] = {
    "get fit": "To establish a sustainable fitness routine",
    "money": "To resolve financial conflicts with a partner",
    "writer blocked": "To overcome creative writing block",
    "learn to code": "To acquire coding skills for career advancement",
    "procrastination": "To overcome procrastination and improve productivity",
    "anxiety": "To find strategies for managing anxiety",
    "team morale": "To improve team morale and collaboration",
    "dinner party": "To plan and host a successful dinner party",
    "ai for content": "To leverage AI for content creation efficiency",
    "child": "To support a child's academic organization and success",
    "feedback at work": "To navigate workplace feedback and conflicts",
    "wardrobe": "To update personal style on a limited budget",
    "podcast": "To launch and grow a podcast or newsletter",
    "tech": "To adopt new technology effectively",
    "study": "To optimize study habits for academic success",
    "startup": "To launch and scale a startup venture",
    "strategy": "To develop strategic plans for business growth",
    "executive": "To enhance executive decision-making skills",
    "entrepreneur": "To innovate and fund entrepreneurial ideas",
}

BASE_CONTEXTS: dict[str, str] = {
    "get fit": (
        "User has repeatedly tried and failed to maintain fitness "
        "routines, feeling discouraged."
    ),
    "money": (
        "User and partner have conflicting financial habits, "
        "causing frequent arguments."
    ),
    "writer blocked": (
        "User is a writer struggling with creative block and lack of inspiration."
    ),
    "learn to code": (
        "User is seeking to learn a new skill to enhance their career prospects."
    ),
    "procrastination": (
        "User struggles with delaying important tasks and wants to be more productive."
    ),
    "anxiety": (
        "User is experiencing feelings of anxiety and is looking for coping mechanisms."
    ),
    "team morale": (
        "User is a leader or member of a team facing challenges "
        "with performance or morale."
    ),
    "dinner party": (
        "User is hosting a dinner party for the first time and "
        "is nervous about planning."
    ),
    "ai for content": "User is a content creator exploring AI tools for efficiency.",
    "child": "Parent helping child with academic organization.",
    "feedback at work": "Team member navigating workplace feedback or conflict.",
    "wardrobe": "User looking to update style on a budget.",
    "podcast": "Media creator starting a podcast or newsletter with tech challenges.",
    "tech": "Elderly user trying to learn a new technology or device.",
    "study": "Student preparing for exams and needing better study strategies.",
    "startup": "Entrepreneur starting a new venture with funding challenges.",
    "strategy": "Executive facing decision-making under uncertainty.",
    "executive": "Senior leader balancing strategic planning with team dynamics.",
    "entrepreneur": "Founder ideation phase with market validation needs.",
}

CONTEXT_PERTURBATIONS: list[str] = [
    " The user is feeling particularly overwhelmed this week.",
    " The user has tried other solutions without success and is feeling skeptical.",
    " The user is on a tight budget and needs low-cost solutions.",
    " The user has very limited free time and needs a time-efficient approach.",
    " The user is dealing with family involvement, adding emotional layers.",
    " The user seeks long-term transformation beyond quick fixes.",
    " The user faces high-stakes decisions with ethical implications.",
    " The user involves multiple stakeholders with conflicting interests.",
    " The user is recovering from a setback and needs gentle restarts.",
    " The user has cultural or regional constraints influencing the approach.",
]

# ============================================================================
# Response Attributes and CoT Steps
# ============================================================================

RESPONSE_ATTRIBUTES: dict[str, list[str]] = {
    "Aspirational": [
        "Affirming",
        "Proactive",
        "Knowledgeable",
        "Forward-looking",
        "Empathetic",
        "Inspirational",
        "Visionary",
        "Motivational",
    ],
    "Transformational": [
        "Empathetic",
        "Insightful",
        "Principle-driven",
        "Clarifying",
        "Reflective",
        "Adaptive",
        "Innovative",
    ],
    "Foundational": [
        "Encouraging",
        "Actionable",
        "Clear",
        "Practical",
        "Concise",
        "Supportive",
        "Structured",
        "Reliable",
    ],
}

RESPONSE_COT_STEPS: dict[str, list[str]] = {
    "Aspirational": [
        "Affirm the user's vision.",
        "Connect to long-term principles.",
        "Propose proactive next steps.",
        "Encourage sustained growth.",
        "Visualize future outcomes.",
        "Align with personal values.",
    ],
    "Transformational": [
        "Acknowledge the dilemma.",
        "Reframe around core values.",
        "Guide toward a clarifying choice.",
        "Suggest experimental adjustments.",
        "Evaluate potential shifts.",
        "Integrate feedback loops.",
    ],
    "Foundational": [
        "Acknowledge the user's problem.",
        "Reframe it using a core principle.",
        "Provide a concrete, foundational first step.",
        "Offer encouragement for consistency.",
        "Break down into manageable parts.",
        "Set immediate goals.",
    ],
}

# ============================================================================
# Paraphrasing Synonyms
# ============================================================================

PARAPHRASE_SYNONYMS: dict[str, list[str]] = {
    "start": ["begin", "initiate", "commence", "launch"],
    "small": ["tiny", "minimal", "basic", "modest"],
    "habit": ["routine", "practice", "pattern", "custom"],
    "process": ["method", "approach", "system", "procedure"],
    "clarity": ["focus", "precision", "understanding", "lucidity"],
    "tool": ["resource", "aid", "instrument", "device"],
    "reframe": ["rethink", "reposition", "recast", "redefine"],
    "foundational": ["basic", "core", "essential", "fundamental"],
    "transformational": ["evolving", "shifting", "revolutionary", "metamorphic"],
    "aspirational": ["visionary", "ambitious", "inspirational", "idealistic"],
    "people": ["individuals", "team", "partners", "stakeholders"],
    "transparency": ["openness", "honesty", "candor", "clarity"],
}

# ============================================================================
# Persona-Specific Augmentations
# ============================================================================

PERSONA_AUGMENTATIONS: dict[str, list[tuple[str, str]]] = {
    "Fitness Seeker": [
        (
            "Training for a specific event like marathon.",
            "Marathon prep; advanced tweaks?",
        ),
        ("Recovering from injury.", "Post-injury; gentle start?"),
        ("Group fitness with friends.", "Involve buddies for motivation?"),
        ("Dietary integration.", "Combine with meal planning?"),
    ],
    "Relationship Manager": [
        ("Planning major life event like wedding.", "Wedding budget stress; tips?"),
        ("Disagreement on long-term goals.", "Different retirement visions; align?"),
        ("Involves children or family.", "Kids' expenses adding up?"),
        ("Debt management focus.", "How to pay off debt together?"),
    ],
    "Creative Professional": [
        ("Commercial project with deadline.", "Client deadline; pressure on?"),
        ("Seeking new voice or style.", "Style stale; fresh angle?"),
        ("Imposter syndrome.", "Feel like fraud; overcome?"),
        ("Collaboration with others.", "Co-write with partner?"),
    ],
    "Career Changer": [
        ("Mid-career switch.", "40s pivot; risks?"),
        ("Skill gap analysis.", "What skills missing?"),
        ("Networking events.", "Best way to network?"),
        ("Resume overhaul.", "Update CV for new field?"),
    ],
    "Business Professional": [
        ("Team reorganization stress.", "Company restructure; adapt?"),
        ("Client negotiation.", "Tough client; strategies?"),
        ("Performance reviews.", "Prep for annual review?"),
        ("Delegation challenges.", "Hard to delegate; tips?"),
    ],
    "Social Host": [
        ("Small space hosting.", "Tiny apartment; not cramped?"),
        ("Dietary restrictions.", "Guests vegan/gluten-free?"),
        ("Theme party ideas.", "Themed event; suggestions?"),
        ("Budget-friendly decor.", "Cheap but nice setup?"),
    ],
    "Content Creator": [
        ("Repurpose content formats.", "Blog to video script?"),
        ("Ethical AI use.", "AI ethics; avoid plagiarism?"),
        ("SEO optimization.", "AI for SEO keywords?"),
        ("Audience engagement.", "Boost interaction with AI?"),
    ],
    "Parent": [
        ("Multiple children balancing.", "Three kids; manage all?"),
        ("ADHD or special needs.", "Child with ADHD; strategies?"),
        ("Homework battles.", "Kid hates homework; motivate?"),
        ("Extracurricular integration.", "Balance school and sports?"),
    ],
    "Team Member": [
        ("Remote work conflicts.", "Virtual team disputes?"),
        ("Feedback delivery.", "Give constructive criticism?"),
        ("Mediation role.", "Mediate between colleagues?"),
        ("Burnout prevention.", "Team burnout; address?"),
    ],
    "Style Seeker": [
        ("Capsule wardrobe build.", "Minimalist essentials?"),
        ("Event-specific outfit.", "Conference attire; comfy?"),
        ("Sustainable shopping.", "Eco-friendly brands?"),
        ("Color palette advice.", "Best colors for me?"),
    ],
    "Media Creator": [
        ("Guest interviewing.", "Find/convince guests?"),
        ("Marketing growth.", "Grow audience fast?"),
        ("Editing tools.", "Free editing software?"),
        ("Monetization.", "Earn from content?"),
        ("Cross-promotion.", "Partner with others?"),
    ],
    "Elderly User": [
        ("Fear of errors.", "Scared to mess up device?"),
        ("Accessibility needs.", "Large fonts/simple terms?"),
        ("Sensory issues.", "Trouble seeing/hearing; tips?"),
        ("Family help.", "Grandkids teach me?"),
    ],
    "Student": [
        ("Multi-subject balance.", "Math/history exams; prioritize?"),
        ("Distraction management.", "Easily distracted; focus?"),
        ("Group study.", "Study with friends; effective?"),
        ("Exam anxiety.", "Test nerves; calm strategies?"),
    ],
    "Executive": [
        ("Strategic forecasting.", "Predict market trends?"),
        ("Decision under uncertainty.", "Risky choices; evaluate?"),
        ("Leadership development.", "Build executive presence?"),
        ("Goal alignment.", "Team goals sync?"),
    ],
    "Entrepreneur": [
        ("Idea validation.", "Test business idea?"),
        ("Pitch preparation.", "Investor pitch; perfect?"),
        ("Funding sources.", "Bootstrap or VC?"),
        ("Scaling challenges.", "Grow without breaking?"),
    ],
}

# ============================================================================
# Inference Functions
# ============================================================================


def infer_persona_archetype(user_query: str) -> str:
    """Infer persona archetype from user query using pattern matching.

    Args:
        user_query: The user's input query text.

    Returns:
        The inferred persona archetype string, or DEFAULT_PERSONA if no match.
    """
    user_query_lower = user_query.lower()
    for persona, pattern in PERSONA_PATTERNS.items():
        if re.search(pattern, user_query_lower):
            return persona
    return DEFAULT_PERSONA


def infer_governing_principle(
    user_query: str, chapter_weights: dict[str, float]
) -> str:
    """Infer governing principle with weighted random selection.

    Args:
        user_query: The user's input query text.
        chapter_weights: Dictionary mapping chapter names to selection weights.

    Returns:
        A string in the format "Chapter X > SubPrinciple".
    """
    user_query_lower = user_query.lower()

    # Pattern-based chapter inference
    if re.search(
        r"\b(relationship|team|partner|people|colleague|social|trust|bond|communicate)\b",
        user_query_lower,
    ):
        chapter = "Chapter 1"
    elif re.search(
        r"\b(process|tool|method|step|habit|routine|system|workflow|optimize)\b",
        user_query_lower,
    ):
        chapter = "Chapter 2"
    elif re.search(
        r"\b(long-term|vision|growth|aspiration|transform|future|goal|dream|evolve)\b",
        user_query_lower,
    ):
        chapter = "Chapter 3"
    else:
        # Weighted random selection
        chapters = list(CHAPTER_PRINCIPLES.keys())
        weights = [chapter_weights.get(ch, 1.0) for ch in chapters]
        chapter = random.choices(chapters, weights=weights)[0]  # noqa: S311

    sub_principle = random.choice(CHAPTER_PRINCIPLES[chapter])  # noqa: S311
    return f"{chapter} > {sub_principle}"


def infer_user_intent(user_query: str) -> str:
    """Infer user intent from query text.

    Args:
        user_query: The user's input query text.

    Returns:
        A descriptive intent string.
    """
    user_query_lower = user_query.lower()
    for key, intent in USER_INTENTS.items():
        if key in user_query_lower:
            return intent
    return "To solve a personal or professional challenge"


def infer_context_summary(user_query: str) -> str:
    """Generate context summary with random perturbations.

    Args:
        user_query: The user's input query text.

    Returns:
        A context summary string with an appended perturbation.
    """
    user_query_lower = user_query.lower()
    key = next((k for k in BASE_CONTEXTS if k in user_query_lower), None)
    default_ctx = "User is facing a general challenge requiring practical guidance."
    base = BASE_CONTEXTS.get(key, default_ctx) if key is not None else default_ctx
    perturbation = random.choice(CONTEXT_PERTURBATIONS)  # noqa: S311
    return base + perturbation


def infer_capability_layer(response: str) -> str:
    """Infer capability layer from response text.

    Args:
        response: The assistant's response text.

    Returns:
        One of: "Foundational", "Transformational", or "Aspirational".
    """
    response_lower = response.lower()

    if re.search(
        r"\b(aspirational|long-term|vision|growth|future|dream|ambition)\b",
        response_lower,
    ):
        return "Aspirational"
    if re.search(
        r"\b(transformational|reframe|identity|experiment|change|shift|evolve)\b",
        response_lower,
    ):
        return "Transformational"
    if re.search(
        r"\b(foundational|start small|simple|habit|basic|core|essential)\b",
        response_lower,
    ):
        return "Foundational"

    return random.choice(CAPABILITY_LAYERS)  # noqa: S311


def infer_response_attributes(layer: str) -> list[str]:
    """Sample response attributes for a given capability layer.

    Args:
        layer: The capability layer name.

    Returns:
        A list of 3 randomly sampled attributes for the layer.
    """
    attr_pool = RESPONSE_ATTRIBUTES.get(layer, RESPONSE_ATTRIBUTES["Foundational"])
    return random.sample(attr_pool, min(3, len(attr_pool)))


def infer_response_cot(layer: str) -> list[str]:
    """Generate chain-of-thought steps for a given capability layer.

    Args:
        layer: The capability layer name.

    Returns:
        A list of 3-4 randomly sampled CoT steps for the layer.
    """
    steps = RESPONSE_COT_STEPS.get(layer, RESPONSE_COT_STEPS["Foundational"])
    num_steps = random.randint(3, min(4, len(steps)))  # noqa: S311
    return random.sample(steps, num_steps)


# ============================================================================
# Perturbation and Paraphrasing
# ============================================================================


def paraphrase_response(response: str) -> str:
    """Paraphrase response using synonym substitution.

    Args:
        response: The original response text.

    Returns:
        A paraphrased version of the response.
    """
    words = response.split()
    paraphrased = [
        random.choice(PARAPHRASE_SYNONYMS.get(word.lower(), [word]))  # noqa: S311
        for word in words
    ]
    return " ".join(paraphrased)


def generate_dynamic_perturbation(base: str, depth: int) -> str:
    """Generate dynamic perturbations via template combinations.

    Args:
        base: The base context string to perturb.
        depth: Number of perturbation templates to combine.

    Returns:
        A perturbed context string.
    """
    inverted = (
        base.replace("limited", "unlimited")
        .replace("tight", "generous")
        .replace("nervous", "excited")
    )
    templates = [
        f"Inverted constraint: {inverted}",
        f"Multi-stakeholder: {base} involving team or family "
        "consensus and diverse opinions.",
        f"Ethical twist: {base} with moral considerations and fairness principles.",
        f"High-stakes: {base} under pressure from deadlines or high expectations.",
        f"Cultural variant: {base} adapted to regional or cultural nuances.",
        f"Outcome-focused: {base} aiming for measurable success metrics.",
    ]
    selected = random.sample(templates, min(depth, len(templates)))
    return " ".join(selected)


def get_augmentations(
    persona: str, context: str, prompt: str, perturbation_depth: int
) -> list[tuple[str, str]]:
    """Generate augmentation pairs using scenario-perturbation method.

    Args:
        persona: The persona archetype string.
        context: The context summary string.
        prompt: The application prompt string.
        perturbation_depth: Number of perturbation templates to combine.

    Returns:
        A list of (context, prompt) tuples for augmentation.
    """
    base_augs: list[tuple[str, str]] = [
        (generate_dynamic_perturbation(context, perturbation_depth), prompt),
        (
            f"{context} The user has unlimited resources and time.",
            f"{prompt} Resources no issue; best premium plan?",
        ),
        (
            f"{context} Involves ethical dilemmas with stakeholders.",
            f"{prompt} How to handle moral conflicts?",
        ),
        (
            f"{context} High-stakes scenario with tight deadlines.",
            f"{prompt} Urgent; quick but effective?",
        ),
        (
            f"{context} Cultural adaptation needed for global context.",
            f"{prompt} Adjust for my region?",
        ),
        (
            f"{context} Focus on measurable outcomes and KPIs.",
            f"{prompt} How to track progress?",
        ),
    ]

    # Add persona-specific augmentations
    for persona_key, augs in PERSONA_AUGMENTATIONS.items():
        if persona_key in persona:
            base_augs.extend(
                [
                    (f"{context} {aug_context}", f"{prompt} {aug_prompt}")
                    for aug_context, aug_prompt in augs
                ]
            )
            break  # Only match first persona

    random.shuffle(base_augs)
    return base_augs


# ============================================================================
# HITL Simulation
# ============================================================================


def simulate_hitl(
    entries: list[dict[str, Any]], hitl_ratio: float
) -> list[dict[str, Any]]:
    """Simulate human-in-the-loop review process.

    Flags a ratio of entries for manual review and applies simulated edits.

    Args:
        entries: List of ICDU entry dictionaries.
        hitl_ratio: Fraction of entries to flag for review (0.0 to 1.0).

    Returns:
        The entries list with simulated HITL edits applied.
    """
    if not entries or hitl_ratio <= 0:
        return entries

    num_flagged = max(1, int(len(entries) * hitl_ratio))
    flagged = random.sample(entries, min(num_flagged, len(entries)))

    for entry in flagged:
        entry_id = entry.get("icdu_id", "unknown")
        prompt_preview = entry.get("application_prompt", "")[:50]
        response_preview = entry.get("ideal_response_final", "")[:50]
        logger.warning(
            f"HITL Flag: Review ID {entry_id}: "
            f"Prompt '{prompt_preview}...' "
            f"Response '{response_preview}...'"
        )
        # Simulate edit: Apply extra paraphrase
        entry["ideal_response_final"] = paraphrase_response(
            entry.get("ideal_response_final", "")
        )

    return entries


# ============================================================================
# Diversity and Evaluation
# ============================================================================


def compute_diversity(
    entries: list[dict[str, Any]],
    field: str,
    threshold: float = DEFAULT_DIVERSITY_THRESHOLD,
) -> dict[str, Any]:
    """Compute text diversity metrics for a field across entries.

    Uses unique ratio and average Jaccard similarity. Lower Jaccard = more diverse.

    Args:
        entries: List of entry dictionaries.
        field: Field name to compute diversity for.
        threshold: Jaccard similarity threshold for diversity flag.

    Returns:
        Dictionary with 'unique_ratio', 'avg_jaccard', and 'diverse' keys.
    """
    values = [str(entry.get(field, "")) for entry in entries]
    if not values:
        return {"unique_ratio": 0.0, "avg_jaccard": 0.0, "diverse": True}

    unique_ratio = len(set(values)) / len(values)

    # Optimize: sample pairs for large datasets
    num_samples = min(1000, len(values) * (len(values) - 1) // 2)
    jaccards = []

    if len(values) > 1:
        # Sample random pairs for efficiency
        for _ in range(num_samples):
            i, j = random.sample(range(len(values)), 2)
            set_i = set(values[i].split())
            set_j = set(values[j].split())
            intersection = len(set_i & set_j)
            union = len(set_i | set_j)
            if union > 0:
                jaccards.append(intersection / union)

    avg_jaccard = sum(jaccards) / len(jaccards) if jaccards else 0.0

    return {
        "unique_ratio": unique_ratio,
        "avg_jaccard": avg_jaccard,
        "diverse": avg_jaccard < threshold,
    }


def visualize_stats(stats: dict[str, dict[str, int]], output_dir: Path) -> None:
    """Generate bar plots for distribution statistics.

    Args:
        stats: Dictionary mapping stat names to count dictionaries.
        output_dir: Directory to save visualization files.
    """
    try:
        for key, counts in stats.items():
            if not counts:
                continue

            _fig, ax = plt.subplots(figsize=(10, 6))
            labels = list(counts.keys())
            values = list(counts.values())

            ax.bar(labels, values)
            ax.set_title(f"{key.capitalize()} Distribution")
            ax.set_xlabel(key.capitalize())
            ax.set_ylabel("Count")
            ax.set_xticklabels(labels, rotation=45, ha="right")
            plt.tight_layout()

            output_path = output_dir / f"{key}_distribution.png"
            plt.savefig(output_path)
            plt.close()

            logger.info(f"Saved visualization: {output_path}")

    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")


def eval_dataset(
    entries: list[dict[str, Any]], output_dir: Path, diversity_threshold: float
) -> None:
    """Evaluate dataset and save diversity metrics.

    Args:
        entries: List of ICDU entry dictionaries.
        output_dir: Directory to save evaluation results.
        diversity_threshold: Jaccard threshold for diversity evaluation.
    """
    diversity = {
        "prompts": compute_diversity(
            entries, "application_prompt", diversity_threshold
        ),
        "responses": compute_diversity(
            entries, "ideal_response_final", diversity_threshold
        ),
        "contexts": compute_diversity(entries, "context_summary", diversity_threshold),
    }

    output_path = output_dir / EVAL_METRICS_FILE
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(diversity, f, indent=4)
        logger.info(f"Evaluation metrics saved to {output_path}")
        logger.debug(f"Diversity metrics: {diversity}")
    except Exception as e:
        logger.error(f"Error saving evaluation metrics: {e}")


# ============================================================================
# Main Conversion Logic
# ============================================================================


def convert_to_icdu(
    entry: dict[str, Any],
    augmentation: tuple[str, str] | None = None,
    chapter_weights: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    """Convert source entry to ICDU format with optional augmentation.

    Args:
        entry: Source entry dictionary with 'messages' field.
        augmentation: Optional (context, prompt) tuple for augmentation.
        chapter_weights: Optional chapter selection weights.

    Returns:
        ICDU entry dictionary, or None if conversion fails.
    """
    messages = entry.get("messages", [])
    if not messages:
        logger.warning("Entry missing 'messages' field")
        return None

    original_prompt = next(
        (m["content"] for m in messages if m.get("role") == "user"), ""
    )
    original_response = next(
        (m["content"] for m in messages if m.get("role") == "assistant"), ""
    )

    if not original_prompt or not original_response:
        logger.warning("Entry missing user or assistant message")
        return None

    if augmentation:
        context, prompt = augmentation
    else:
        context = infer_context_summary(original_prompt)
        prompt = original_prompt

    persona = infer_persona_archetype(original_prompt)
    weights = chapter_weights or DEFAULT_CHAPTER_WEIGHTS
    principle = infer_governing_principle(original_prompt, weights)
    layer = infer_capability_layer(original_response)
    perturbed_response = paraphrase_response(original_response)

    return {
        "icdu_id": str(uuid.uuid4()),
        "persona_archetype": persona,
        "governing_principle": principle,
        "capability_layer": layer,
        "user_intent": infer_user_intent(original_prompt),
        "context_summary": context,
        "application_prompt": prompt,
        "ideal_response_final": perturbed_response,
        "ideal_response_attributes": infer_response_attributes(layer),
        "ideal_response_cot": infer_response_cot(layer),
    }


def ensure_uniqueness(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate entries based on key field hash.

    Args:
        entries: List of ICDU entry dictionaries.

    Returns:
        Deduplicated list of entries.
    """
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []

    for entry in entries:
        key_dict = {
            "persona": entry.get("persona_archetype", ""),
            "prompt": entry.get("application_prompt", ""),
            "response": entry.get("ideal_response_final", ""),
        }
        key = json.dumps(key_dict, sort_keys=True)

        if key not in seen:
            seen.add(key)
            unique.append(entry)

    return unique


# ============================================================================
# Parallel Processing
# ============================================================================


def process_entry(args: tuple[dict[str, Any], dict[str, Any]]) -> list[dict[str, Any]]:
    """Worker function for parallel entry processing.

    Args:
        args: Tuple of (entry dictionary, processing arguments dictionary).

    Returns:
        List of generated ICDU entries (base + augmentations).
    """
    entry, args_dict = args
    chapter_weights = args_dict.get("chapter_weights", DEFAULT_CHAPTER_WEIGHTS)

    base_icdu = convert_to_icdu(entry, chapter_weights=chapter_weights)
    if not base_icdu:
        return []

    local_entries = [base_icdu]
    persona = base_icdu["persona_archetype"]
    context = base_icdu["context_summary"]
    prompt = base_icdu["application_prompt"]
    perturbation_depth = args_dict.get("perturbation_depth", DEFAULT_PERTURBATION_DEPTH)

    augs = get_augmentations(persona, context, prompt, perturbation_depth)
    augmentation_factor = args_dict.get(
        "augmentation_factor", DEFAULT_AUGMENTATION_FACTOR
    )

    if augs and augmentation_factor > 1:
        num_augs = augmentation_factor - 1
        selected = random.choices(augs, k=min(num_augs, len(augs)))  # noqa: S311

        for aug in selected:
            aug_icdu = convert_to_icdu(entry, aug, chapter_weights=chapter_weights)
            if aug_icdu:
                local_entries.append(aug_icdu)

    return local_entries


# ============================================================================
# Main Function
# ============================================================================


def generate_icdu_dataset(
    input_file: Path,
    output_dir: Path,
    augmentation_factor: int = DEFAULT_AUGMENTATION_FACTOR,
    validation_split: float = DEFAULT_VALIDATION_SPLIT,
    chapter_weights: dict[str, float] | None = None,
    perturbation_depth: int = DEFAULT_PERTURBATION_DEPTH,
    hitl_ratio: float = DEFAULT_HITL_RATIO,
    diversity_threshold: float = DEFAULT_DIVERSITY_THRESHOLD,
    num_processes: int = cpu_count(),
) -> None:
    """Generate publication-ready ICDU dataset with advanced features.

    Main orchestration function that:
        - Loads source entries from JSONL
        - Processes entries in parallel with augmentation
        - Applies deduplication and HITL simulation
        - Evaluates diversity and generates visualizations
        - Splits into train/validation sets

    Args:
        input_file: Path to source JSONL file.
        output_dir: Directory for output files and visualizations.
        augmentation_factor: Number of augmentations per base entry.
        validation_split: Fraction of data for validation (0.0 to 1.0).
        chapter_weights: Optional chapter selection weights dictionary.
        perturbation_depth: Number of perturbation templates to combine.
        hitl_ratio: Fraction of entries for HITL simulation (0.0 to 1.0).
        diversity_threshold: Jaccard threshold for diversity evaluation.
        num_processes: Number of parallel worker processes.
    """
    # Validate inputs
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if not (0.0 <= validation_split <= 1.0):
        logger.error(f"Invalid validation_split: {validation_split}")
        raise ValueError("validation_split must be between 0.0 and 1.0")

    if not (0.0 <= hitl_ratio <= 1.0):
        logger.error(f"Invalid hitl_ratio: {hitl_ratio}")
        raise ValueError("hitl_ratio must be between 0.0 and 1.0")

    # Setup
    chapter_weights = chapter_weights or DEFAULT_CHAPTER_WEIGHTS
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load source entries
    logger.info(f"Loading source entries from {input_file}")
    try:
        with open(input_file, encoding="utf-8") as f:
            source_entries = [json.loads(line) for line in f if line.strip()]
        logger.info(f"Loaded {len(source_entries)} source entries")
    except Exception as e:
        logger.error(f"Error loading input file: {e}")
        raise

    if not source_entries:
        logger.warning("No source entries found")
        return

    # Prepare processing arguments
    args_dict = {
        "augmentation_factor": augmentation_factor,
        "perturbation_depth": perturbation_depth,
        "chapter_weights": chapter_weights,
    }

    # Parallel processing
    logger.info(f"Processing with {num_processes} parallel workers...")
    try:
        with Pool(num_processes) as pool:
            results = list(
                tqdm(
                    pool.imap(
                        process_entry, [(entry, args_dict) for entry in source_entries]
                    ),
                    total=len(source_entries),
                    desc="Processing entries",
                )
            )
    except Exception as e:
        logger.error(f"Error during parallel processing: {e}")
        raise

    # Flatten results
    all_icdu_entries = [item for sublist in results for item in sublist]
    logger.info(f"Generated {len(all_icdu_entries)} total entries")

    # Deduplication
    all_icdu_entries = ensure_uniqueness(all_icdu_entries)
    logger.info(f"After deduplication: {len(all_icdu_entries)} unique entries")

    # HITL simulation
    all_icdu_entries = simulate_hitl(all_icdu_entries, hitl_ratio)

    # Compute statistics
    stats = {
        "personas": dict(
            Counter(e.get("persona_archetype", "") for e in all_icdu_entries)
        ),
        "principles": dict(
            Counter(e.get("governing_principle", "") for e in all_icdu_entries)
        ),
        "layers": dict(
            Counter(e.get("capability_layer", "") for e in all_icdu_entries)
        ),
    }
    logger.info(f"Dataset statistics: {stats}")

    # Evaluation and visualization
    eval_dataset(all_icdu_entries, output_dir, diversity_threshold)
    visualize_stats(stats, output_dir)

    # Train/validation split
    if len(all_icdu_entries) < 2:
        logger.warning("Insufficient entries for train/validation split")
        train, val = all_icdu_entries, []
    else:
        train, val = train_test_split(
            all_icdu_entries,
            test_size=validation_split,
            random_state=DEFAULT_RANDOM_STATE,
        )

    # Save output files
    train_file = output_dir / TRAIN_FILE_NAME
    val_file = output_dir / VAL_FILE_NAME

    try:
        with open(train_file, "w", encoding="utf-8") as f:
            for item in train:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(train)} training entries to {train_file}")

        if val:
            with open(val_file, "w", encoding="utf-8") as f:
                for item in val:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            logger.info(f"Saved {len(val)} validation entries to {val_file}")
    except Exception as e:
        logger.error(f"Error saving output files: {e}")
        raise

    logger.info("Dataset generation complete")


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> None:
    """Command-line interface entry point."""
    parser = argparse.ArgumentParser(
        description="Publication-ready ICDU generator with advanced features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("breaking_better_training_data_v6.jsonl"),
        help="Input JSONL source file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory for outputs and visualizations",
    )
    parser.add_argument(
        "--augmentation-factor",
        type=int,
        default=DEFAULT_AUGMENTATION_FACTOR,
        help="Number of augmentations per base entry",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=DEFAULT_VALIDATION_SPLIT,
        help="Fraction of data for validation (0.0 to 1.0)",
    )
    parser.add_argument(
        "--chapter-weights",
        type=json.loads,
        default=json.dumps(DEFAULT_CHAPTER_WEIGHTS),
        help="JSON dict of chapter selection weights",
    )
    parser.add_argument(
        "--perturbation-depth",
        type=int,
        default=DEFAULT_PERTURBATION_DEPTH,
        help="Number of perturbation templates to combine",
    )
    parser.add_argument(
        "--hitl-ratio",
        type=float,
        default=DEFAULT_HITL_RATIO,
        help="Fraction of entries for HITL simulation (0.0 to 1.0)",
    )
    parser.add_argument(
        "--diversity-threshold",
        type=float,
        default=DEFAULT_DIVERSITY_THRESHOLD,
        help="Jaccard threshold for diversity alert",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=cpu_count(),
        help="Number of parallel processes",
    )

    args = parser.parse_args()

    try:
        generate_icdu_dataset(
            input_file=args.input_file,
            output_dir=args.output_dir,
            augmentation_factor=args.augmentation_factor,
            validation_split=args.validation_split,
            chapter_weights=args.chapter_weights,
            perturbation_depth=args.perturbation_depth,
            hitl_ratio=args.hitl_ratio,
            diversity_threshold=args.diversity_threshold,
            num_processes=args.num_processes,
        )
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        raise


if __name__ == "__main__":
    main()
