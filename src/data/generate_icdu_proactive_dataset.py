"""ICDU dataset generation with proactive questioning capabilities.

This module generates Intent-Conscious Data Unit (ICDU) datasets with proactive
follow-up questions appended to responses. The proactive approach trains models
to engage users and anticipate their next needs.

Features:
    - Conversion from source format to ICDU with proactive questions
    - Update existing ICDU entries to include proactive questions
    - Parallel processing for efficient dataset generation
    - Data augmentation with persona-aware variations
    - Optional evaluation reports and visualizations
    - Dataset size sampling for controlled output

The module processes source entries and converts them to ICDU format with:
    - Persona archetype inference
    - Governing principle assignment
    - Capability layer classification
    - Proactive follow-up question generation
    - Context-aware response generation

Usage:
    python generate_icdu_proactive_dataset.py \\
        --input-file source_data.jsonl \\
        --output-dir output/ \\
        --augmentation-factor 10 \\
        --validation-split 0.1 \\
        --dataset-size 5000 \\
        --run-reports \\
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
DEFAULT_AUGMENTATION_FACTOR = 10
DEFAULT_VALIDATION_SPLIT = 0.1
DEFAULT_DIVERSITY_THRESHOLD = 0.7
DEFAULT_RANDOM_STATE = 42
DEFAULT_MAX_DIVERSITY_SAMPLES = 10000

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
TRAIN_FILE_NAME = "icdu_training_data_proactive.jsonl"
VAL_FILE_NAME = "icdu_validation_data_proactive.jsonl"
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
    "Home Chef > Meal Planner": (
        r"\b(cook|recipe|meal prep|kitchen|groceries|dinner|lunch|breakfast)\b"
    ),
    "DIY Homeowner > Project Beginner": (
        r"\b(home improvement|diy|renovate|fix|paint|garden|repair)\b"
    ),
    "Mindfulness Seeker > Habit Builder": (
        r"\b(meditation|mindfulness|anxiety|stress|focus|well-being|calm|habit)\b"
    ),
    "Pet Owner > New Trainer": (
        r"\b(dog|cat|pet|puppy|kitten|training|behavior|animal)\b"
    ),
    "Recent Graduate > Job Seeker": (
        r"\b(graduate|job search|entry-level|first job|resume|linkedin)\b"
    ),
    "Retiree > Life Planner": (
        r"\b(retire|retirement|pension|social security|downsize|travel)\b"
    ),
    "Event Planner > Detail Coordinator": (
        r"\b(wedding|conference|event plan|vendor|budget|timeline)\b"
    ),
    "Hobbyist > Skill Developer": (
        r"\b(hobby|learn guitar|photography|woodworking|craft|practice)\b"
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
    "cook": "To streamline weekly meal planning and preparation",
    "diy": (
        "To successfully complete a home improvement project "
        "without getting overwhelmed"
    ),
    "meditation": "To build a consistent mindfulness or meditation practice",
    "pet": "To effectively train a new pet and build a good relationship",
    "graduate": "To create a standout resume and land a first job",
    "retire": "To create a fulfilling and financially stable retirement plan",
    "event plan": "To organize a successful event while managing all the details",
    "hobby": "To develop a new skill or hobby in a structured way",
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
    "cook": (
        "User enjoys cooking but finds weekly meal planning "
        "stressful and time-consuming."
    ),
    "diy": (
        "User is a new homeowner feeling overwhelmed by a list "
        "of potential DIY projects."
    ),
    "meditation": (
        "User has heard about the benefits of meditation but "
        "struggles to practice it consistently."
    ),
    "pet": (
        "User has recently adopted a new pet and is facing "
        "challenges with basic training."
    ),
    "graduate": (
        "User is a recent graduate feeling overwhelmed by the job application process."
    ),
    "retire": (
        "User is approaching retirement and is unsure how to "
        "structure their time and finances."
    ),
    "event plan": (
        "User is planning a major event and is struggling to "
        "keep track of all the moving parts."
    ),
    "hobby": "User wants to learn a new skill but doesn't know where to start.",
}

CONTEXT_PERTURBATIONS: list[str] = [
    " The user is feeling particularly overwhelmed this week.",
    " The user has tried other solutions without success and is feeling skeptical.",
    " The user is on a tight budget and needs low-cost solutions.",
    " The user has very limited free time and needs a time-efficient approach.",
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
    ],
    "Transformational": [
        "Empathetic",
        "Insightful",
        "Principle-driven",
        "Clarifying",
        "Reflective",
        "Adaptive",
    ],
    "Foundational": [
        "Encouraging",
        "Actionable",
        "Clear",
        "Practical",
        "Concise",
        "Supportive",
        "Structured",
    ],
}

RESPONSE_COT_STEPS: dict[str, list[str]] = {
    "Aspirational": [
        "Affirm the user's long-term vision.",
        "Connect the immediate goal to a larger principle.",
        "Propose proactive next steps that build toward the vision.",
        "Formulate a proactive follow-up question to explore future possibilities.",
    ],
    "Transformational": [
        "Acknowledge the user's current dilemma or perspective.",
        "Reframe the problem around a core value or principle.",
        "Guide the user toward a clarifying choice or experiment.",
        "Formulate a follow-up question to encourage deeper reflection.",
    ],
    "Foundational": [
        "Acknowledge the user's immediate problem.",
        "Provide a concrete, actionable first step based on a core principle.",
        "Offer encouragement to build consistency.",
        "Formulate a clarifying follow-up question to ensure the "
        "first step is manageable.",
    ],
}

# ============================================================================
# Paraphrasing Synonyms
# ============================================================================

PARAPHRASE_SYNONYMS: dict[str, list[str]] = {
    "start": ["begin", "initiate", "commence"],
    "small": ["tiny", "minimal", "modest"],
    "habit": ["routine", "practice", "pattern"],
    "process": ["method", "approach", "system"],
    "clarity": ["focus", "precision", "understanding"],
    "tool": ["resource", "aid", "instrument"],
    "reframe": ["rethink", "reposition", "recast"],
    "foundational": ["basic", "core", "essential"],
    "people": ["individuals", "team", "partners"],
    "transparency": ["openness", "honesty", "candor"],
}

# ============================================================================
# Proactive Question Bank
# ============================================================================

PROACTIVE_QUESTIONS: dict[str, list[str]] = {
    "Fitness Seeker": [
        "Would you like to explore how to track your progress effectively?",
        "Are you interested in how this fitness plan could be "
        "adapted for travel or busy weeks?",
    ],
    "Relationship Manager": [
        "Now that we have a plan, would it be helpful to think "
        "about how to best introduce this topic to your partner?",
        "Would you like to discuss some common pitfalls to avoid "
        "when managing shared finances?",
    ],
    "Creative Professional": [
        "Would you like some exercises to warm up your creative "
        "muscles before a writing session?",
        "Are you interested in exploring how other writers in "
        "your genre have overcome similar blocks?",
    ],
    "Career Changer": [
        "Would you be interested in identifying some small, "
        "low-risk projects to start building your portfolio?",
        "Now that you have a starting point, would you like to "
        "map out a 3-month learning plan?",
    ],
    "Home Chef": [
        "Would you be interested in some tips for grocery "
        "shopping efficiency based on this plan?",
        "How does the idea of batch cooking a few key components "
        "on the weekend sound to you?",
    ],
    "DIY Homeowner": [
        "Before you start, would you like a checklist of the "
        "essential tools and materials for this project?",
        "Would it be helpful to break this project down into "
        "smaller, weekend-sized tasks?",
    ],
    "Mindfulness Seeker": [
        "Would you like to explore different types of meditation "
        "to see which one resonates most with you?",
        "To help build this habit, would you be interested in "
        "some tips for creating a dedicated mindfulness space?",
    ],
    "Pet Owner": [
        "Would you be interested in a simple schedule for daily training sessions?",
        "Socialization is key for a well-behaved pet. Would you "
        "like some ideas for how to do this safely?",
    ],
    "Recent Graduate": [
        "Would you like to work on tailoring your resume for a "
        "specific job description you're interested in?",
        "To prepare for interviews, would you like to practice "
        "answering some common questions?",
    ],
    "Retiree": [
        "Would you like to explore some ideas for hobbies or "
        "volunteer work that align with your interests?",
        "To help with financial planning, would you be "
        "interested in creating a simple retirement budget?",
    ],
    "Event Planner": [
        "Would you like to create a master timeline for the "
        "event to ensure everything stays on track?",
        "Managing a budget is crucial. Would you be interested "
        "in a template for tracking expenses?",
    ],
    "Hobbyist": [
        "Would you be interested in a simple practice schedule "
        "to help you improve consistently?",
        "To stay motivated, would you like to set some small, "
        "achievable goals for your hobby?",
    ],
    "default": [
        "Does this initial step feel manageable for you?",
        "What's one potential obstacle you foresee, and how can we plan for it?",
        "Is there another aspect of this you'd like to explore next?",
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
    num_steps = random.randint(3, len(steps))  # noqa: S311
    return random.sample(steps, num_steps)


def generate_follow_up_question(persona: str) -> str:
    """Generate a proactive, persona-aware follow-up question.

    This function generates follow-up questions that train the model to engage
    users and anticipate their next needs based on their persona archetype.

    Args:
        persona: The persona archetype string (e.g., "Fitness Seeker > ...").

    Returns:
        A proactive follow-up question string.
    """
    # Extract base persona name (before ">")
    persona_key = persona.split(">")[0].strip()
    question_bank_key = next(
        (k for k in PROACTIVE_QUESTIONS if k in persona_key), "default"
    )
    return random.choice(PROACTIVE_QUESTIONS[question_bank_key])  # noqa: S311


# ============================================================================
# Data Augmentation and Utility Functions
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
        random.choice(  # noqa: S311
            PARAPHRASE_SYNONYMS.get(word.lower().strip(".,?!"), [word])
        )
        for word in words
    ]
    return " ".join(paraphrased)


def get_augmentations(persona: str, context: str, prompt: str) -> list[tuple[str, str]]:
    """Generate augmented context and prompt pairs for data diversity.

    Args:
        persona: The persona archetype string.
        context: The context summary string.
        prompt: The application prompt string.

    Returns:
        A list of (context, prompt) tuples for augmentation.
    """
    base_augs: list[tuple[str, str]] = [
        (
            f"{context} The user has unlimited resources.",
            f"{prompt} What if budget wasn't an issue?",
        ),
        (
            f"{context} The user is facing a tight deadline.",
            f"{prompt} I need a solution that is fast and effective.",
        ),
        (
            f"{context} This involves a team with conflicting opinions.",
            f"{prompt} How can I get my team to agree on this?",
        ),
    ]

    # Add persona-specific augmentations
    if "Fitness Seeker" in persona:
        base_augs.append(
            (
                f"{context} The user is recovering from an injury.",
                f"{prompt} How can I adapt this for a recent injury?",
            )
        )

    random.shuffle(base_augs)
    return base_augs


def ensure_uniqueness(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate entries based on key field hash.

    Args:
        entries: List of ICDU entry dictionaries.

    Returns:
        Deduplicated list of entries.
    """
    seen: set[str] = set()
    unique_entries: list[dict[str, Any]] = []

    for entry in entries:
        key_dict = {
            "p": entry.get("persona_archetype", ""),
            "q": entry.get("application_prompt", ""),
            "r": entry.get("ideal_response_final", ""),
        }
        key = json.dumps(key_dict, sort_keys=True)

        if key not in seen:
            seen.add(key)
            unique_entries.append(entry)

    return unique_entries


# ============================================================================
# Reporting and Visualization Functions
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
        Dictionary with 'unique_ratio', 'avg_jaccard_similarity', and 'is_diverse' keys.
    """
    values = [str(entry.get(field, "")) for entry in entries]
    if not values:
        return {"unique_ratio": 0.0, "avg_jaccard_similarity": 0.0, "is_diverse": True}

    unique_ratio = len(set(values)) / len(values)

    # Sample pairs to avoid O(n²) complexity on large datasets
    max_pairs = len(values) * (len(values) - 1) // 2
    num_samples = min(max_pairs, DEFAULT_MAX_DIVERSITY_SAMPLES)
    jaccards = []

    if len(values) > 1 and num_samples > 0:
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
        "avg_jaccard_similarity": avg_jaccard,
        "is_diverse": avg_jaccard < threshold,
    }


def visualize_stats(stats: dict[str, dict[str, int]], output_dir: Path) -> None:
    """Generate and save bar plots for dataset distributions.

    Args:
        stats: Dictionary mapping stat names to count dictionaries.
        output_dir: Directory to save visualization files.
    """
    try:
        for key, counts in stats.items():
            if not counts:
                continue

            _fig, ax = plt.subplots(figsize=(12, 8))
            labels = list(counts.keys())
            values = list(counts.values())

            ax.bar(labels, values)
            ax.set_title(
                f"Distribution of {key.replace('_', ' ').title()}", fontsize=16
            )
            ax.set_xlabel(key.replace("_", " ").title(), fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            ax.tick_params(axis="x", rotation=45, labelsize=10, labelright=True)
            plt.tight_layout()

            plot_path = output_dir / f"{key}_distribution.png"
            plt.savefig(plot_path)
            plt.close()

            logger.info(f"Saved {key} distribution plot to {plot_path}")

    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")


def run_dataset_evaluation(
    entries: list[dict[str, Any]],
    output_dir: Path,
    diversity_threshold: float = DEFAULT_DIVERSITY_THRESHOLD,
) -> None:
    """Run all evaluation and reporting tasks.

    Calculates diversity metrics, generates visualizations, and saves reports.

    Args:
        entries: List of ICDU entry dictionaries.
        output_dir: Directory to save evaluation results.
        diversity_threshold: Jaccard threshold for diversity evaluation.
    """
    logger.info("--- Starting Dataset Evaluation ---")

    # Calculate and log diversity metrics
    diversity_metrics = {
        "prompts_diversity": compute_diversity(
            entries, "application_prompt", diversity_threshold
        ),
        "responses_diversity": compute_diversity(
            entries, "ideal_response_final", diversity_threshold
        ),
        "contexts_diversity": compute_diversity(
            entries, "context_summary", diversity_threshold
        ),
    }

    metrics_path = output_dir / EVAL_METRICS_FILE
    try:
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(diversity_metrics, f, indent=4)
        logger.info(f"Diversity metrics saved to {metrics_path}")
    except Exception as e:
        logger.error(f"Error saving diversity metrics: {e}")
        return

    # Log diversity metrics summary
    for key, metrics in diversity_metrics.items():
        logger.info(
            f"{key.title()}: "
            f"Unique Ratio={metrics['unique_ratio']:.2f}, "
            f"Avg Jaccard Similarity={metrics['avg_jaccard_similarity']:.2f}"
        )

    # Calculate and visualize distributions
    stats = {
        "persona_archetypes": dict(
            Counter(e.get("persona_archetype", "") for e in entries)
        ),
        "governing_principles": dict(
            Counter(e.get("governing_principle", "") for e in entries)
        ),
        "capability_layers": dict(
            Counter(e.get("capability_layer", "") for e in entries)
        ),
    }
    visualize_stats(stats, output_dir)

    logger.info("--- Dataset Evaluation Complete ---")


# ============================================================================
# Main Conversion and Processing Logic
# ============================================================================


def convert_to_icdu(
    entry: dict[str, Any],
    augmentation: tuple[str, str] | None = None,
    chapter_weights: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    """Convert source entry to ICDU format with proactive question.

    Args:
        entry: Source entry dictionary with 'messages' field.
        augmentation: Optional (context, prompt) tuple for augmentation.
        chapter_weights: Optional chapter selection weights.

    Returns:
        ICDU entry dictionary with proactive question, or None if conversion fails.
    """
    messages = entry.get("messages", [])
    if not messages:
        logger.warning("Entry missing 'messages' field")
        return None

    original_prompt = next(
        (m.get("content", "") for m in messages if m.get("role") == "user"), ""
    )
    original_response = next(
        (m.get("content", "") for m in messages if m.get("role") == "assistant"), ""
    )

    if not original_prompt or not original_response:
        logger.warning("Entry missing user or assistant message")
        return None

    if augmentation:
        context, prompt = augmentation
    else:
        context = infer_context_summary(original_prompt)
        prompt = original_prompt

    persona = infer_persona_archetype(prompt)
    weights = chapter_weights or {}
    layer = infer_capability_layer(original_response)
    perturbed_response = paraphrase_response(original_response)
    follow_up_question = generate_follow_up_question(persona)
    final_response_with_question = f"{perturbed_response} {follow_up_question}"

    return {
        "icdu_id": str(uuid.uuid4()),
        "persona_archetype": persona,
        "governing_principle": infer_governing_principle(prompt, weights),
        "capability_layer": layer,
        "user_intent": infer_user_intent(prompt),
        "context_summary": context,
        "application_prompt": prompt,
        "ideal_response_final": final_response_with_question,
        "ideal_response_attributes": infer_response_attributes(layer),
        "ideal_response_cot": infer_response_cot(layer),
    }


def update_icdu_entry(entry: dict[str, Any]) -> dict[str, Any] | None:
    """Update existing ICDU entry to include proactive question.

    If the entry already ends with a question, it is returned unchanged.

    Args:
        entry: Existing ICDU entry dictionary.

    Returns:
        Updated entry dictionary, or None if update fails.
    """
    original_prompt = entry.get("application_prompt", "")
    original_response = entry.get("ideal_response_final", "")

    if not original_prompt or not original_response:
        logger.warning("Entry missing required fields")
        return None

    # Skip if already has a question
    if original_response.strip().endswith("?"):
        return entry

    persona = entry.get("persona_archetype", infer_persona_archetype(original_prompt))
    layer = entry.get("capability_layer", infer_capability_layer(original_response))

    follow_up_question = generate_follow_up_question(persona)
    final_response_with_question = f"{original_response.strip()} {follow_up_question}"

    updated_cot = infer_response_cot(layer)

    updated_entry = entry.copy()
    updated_entry["ideal_response_final"] = final_response_with_question
    updated_entry["ideal_response_cot"] = updated_cot

    return updated_entry


def process_entry(args: tuple[dict[str, Any], dict[str, Any]]) -> list[dict[str, Any]]:
    """Worker function for parallel entry processing.

    Handles both source format conversion and existing ICDU entry updates.

    Args:
        args: Tuple of (entry dictionary, processing configuration dictionary).

    Returns:
        List of generated or updated ICDU entries.
    """
    entry, config = args

    if "messages" in entry:
        # Path for converting source data and augmenting it
        chapter_weights = config.get("chapter_weights", {})
        base_icdu = convert_to_icdu(entry, chapter_weights=chapter_weights)
        if not base_icdu:
            return []

        processed_entries = [base_icdu]
        augmentation_factor = config.get("augmentation_factor", 1)

        if augmentation_factor > 1:
            augmentations = get_augmentations(
                base_icdu["persona_archetype"],
                base_icdu["context_summary"],
                base_icdu["application_prompt"],
            )
            num_to_generate = augmentation_factor - 1
            selected_augs = random.choices(  # noqa: S311
                augmentations, k=min(num_to_generate, len(augmentations))
            )

            for aug in selected_augs:
                augmented_icdu = convert_to_icdu(
                    entry, aug, chapter_weights=chapter_weights
                )
                if augmented_icdu:
                    processed_entries.append(augmented_icdu)

        return processed_entries
    else:
        # Path for updating a single existing ICDU entry
        updated_entry = update_icdu_entry(entry)
        return [updated_entry] if updated_entry else []


# ============================================================================
# Main Function
# ============================================================================


def generate_icdu_dataset(
    input_file: Path,
    output_dir: Path,
    augmentation_factor: int = DEFAULT_AUGMENTATION_FACTOR,
    validation_split: float = DEFAULT_VALIDATION_SPLIT,
    num_processes: int = cpu_count(),
    dataset_size: int | None = None,
    run_reports: bool = False,
    chapter_weights: dict[str, float] | None = None,
) -> None:
    """Generate ICDU dataset with proactive questioning capabilities.

    Main orchestration function that:
        - Loads source entries from JSONL
        - Processes entries in parallel (converts or updates)
        - Applies deduplication
        - Optionally samples to desired dataset size
        - Runs evaluation reports if requested
        - Splits into train/validation sets

    Args:
        input_file: Path to source JSONL file (source format or ICDU format).
        output_dir: Directory for output files and reports.
        augmentation_factor: Augmentations per base entry (source format only).
        validation_split: Fraction of data for validation (0.0 to 1.0).
        num_processes: Number of parallel worker processes.
        dataset_size: Optional total number of samples (with replacement).
        run_reports: If True, generate evaluation reports and visualizations.
        chapter_weights: Optional chapter selection weights dictionary.
    """
    # Validate inputs
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if not (0.0 <= validation_split <= 1.0):
        logger.error(f"Invalid validation_split: {validation_split}")
        raise ValueError("validation_split must be between 0.0 and 1.0")

    # Setup
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

    # Prepare processing configuration
    processing_config = {
        "augmentation_factor": augmentation_factor,
        "chapter_weights": chapter_weights or {},
    }

    # Parallel processing
    logger.info(f"Starting parallel processing with {num_processes} cores...")
    try:
        with Pool(num_processes) as pool:
            results = list(
                tqdm(
                    pool.imap(
                        process_entry,
                        [(entry, processing_config) for entry in source_entries],
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
    unique_entries = ensure_uniqueness(all_icdu_entries)
    logger.info(
        f"After deduplication: {len(unique_entries)} unique entries "
        f"(from {len(all_icdu_entries)} total)"
    )

    # Sample to desired dataset size
    if dataset_size and dataset_size > 0:
        if not unique_entries:
            logger.error("No unique entries were generated to sample from")
            raise ValueError("No unique entries available for sampling")

        logger.info(
            f"Sampling {dataset_size} entries (with replacement) "
            f"from {len(unique_entries)} unique entries"
        )
        final_entries = random.choices(unique_entries, k=dataset_size)  # noqa: S311
    else:
        final_entries = unique_entries

    if not final_entries:
        logger.error("No entries were generated or sampled")
        raise ValueError("No entries available for output")

    # Run optional reporting
    if run_reports:
        run_dataset_evaluation(final_entries, output_dir)

    # Train/validation split
    if len(final_entries) < 2:
        logger.warning("Insufficient entries for train/validation split")
        train_data, val_data = final_entries, []
    else:
        train_data, val_data = train_test_split(
            final_entries, test_size=validation_split, random_state=DEFAULT_RANDOM_STATE
        )

    # Save output files
    train_file = output_dir / TRAIN_FILE_NAME
    val_file = output_dir / VAL_FILE_NAME

    try:
        with open(train_file, "w", encoding="utf-8") as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(train_data)} training entries to {train_file}")

        if val_data:
            with open(val_file, "w", encoding="utf-8") as f:
                for item in val_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            logger.info(f"Saved {len(val_data)} validation entries to {val_file}")
    except Exception as e:
        logger.error(f"Error saving output files: {e}")
        raise

    logger.info(
        f"Successfully generated datasets. "
        f"Total: {len(final_entries)}, "
        f"Training: {len(train_data)}, "
        f"Validation: {len(val_data)}"
    )


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> None:
    """Command-line interface entry point."""
    parser = argparse.ArgumentParser(
        description="Generate or update an Intent-Conscious Data Unit (ICDU) "
        "dataset with proactive questioning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="Path to the source JSONL file (either in source format or ICDU format)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("generated_data"),
        help="Directory to save the output files",
    )
    parser.add_argument(
        "--augmentation-factor",
        type=int,
        default=DEFAULT_AUGMENTATION_FACTOR,
        help="Number of augmentations to create per source entry "
        "(only applies when converting from source format)",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=DEFAULT_VALIDATION_SPLIT,
        help="Fraction of the data to use for the validation set (0.0 to 1.0)",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=cpu_count(),
        help="Number of CPU cores to use for parallel processing",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=None,
        help="Desired number of total samples in the final dataset. "
        "If not provided, all generated entries will be used. "
        "Samples with replacement if needed.",
    )
    parser.add_argument(
        "--run-reports",
        action="store_true",
        help="If set, generate and save evaluation reports and plots",
    )

    args = parser.parse_args()

    try:
        generate_icdu_dataset(
            input_file=args.input_file,
            output_dir=args.output_dir,
            augmentation_factor=args.augmentation_factor,
            validation_split=args.validation_split,
            num_processes=args.num_processes,
            dataset_size=args.dataset_size,
            run_reports=args.run_reports,
        )
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        raise


if __name__ == "__main__":
    main()
