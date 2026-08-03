import json
import logging
import random
import uuid
from pathlib import Path
from typing import cast

from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
PERTURBATION_PROBABILITY = 0.5
DEFAULT_PERSONA = "General User > Problem Solver"
DEFAULT_INTENT = "To solve a personal or professional challenge"
DEFAULT_CONTEXT = "User is facing a general challenge requiring practical guidance."
DEFAULT_CAPABILITY_LAYER = "Foundational"
DEFAULT_GOVERNING_PRINCIPLE = "Chapter 1 > Clarity"

# Inference mappings for better performance and maintainability
QUERY_PATTERNS = {
    "get fit": {
        "persona": "Fitness Seeker > Struggling Starter",
        "intent": "To establish a sustainable fitness routine",
        "context": (
            "User has repeatedly tried and failed to maintain fitness "
            "routines, feeling discouraged."
        ),
    },
    "fighting about money": {
        "persona": "Relationship Manager > Financial Planner",
        "intent": "To resolve financial conflicts with partner",
        "context": (
            "User and partner have conflicting financial habits, "
            "causing frequent arguments."
        ),
    },
    "writer and I'm completely blocked": {
        "persona": "Creative Professional > Blocked Writer",
        "intent": "To overcome creative writing block",
        "context": (
            "User is a writer struggling with creative block and lack of inspiration."
        ),
    },
    "learn to code": {
        "persona": "Career Changer > Aspiring Coder",
        "intent": "To acquire coding skills for career advancement",
        "context": (
            "User is a marketer seeking to learn coding to enhance career prospects."
        ),
    },
    "client meetings": {
        "persona": "Business Professional > Meeting Organizer",
        "intent": "To streamline client meeting processes",
        "context": (
            "User manages client meetings that often run overtime, seeking efficiency."
        ),
    },
    "hosting a dinner party": {
        "persona": "Social Host > First-Time Planner",
        "intent": "To successfully host a first-time dinner party",
        "context": (
            "User is planning their first dinner party and feels "
            "stressed about logistics."
        ),
    },
    "use AI for content creation": {
        "persona": "Content Creator > AI Adopter",
        "intent": "To create original content using AI tools",
        "context": (
            "User wants to use AI for content creation but is "
            "concerned about originality."
        ),
    },
    "child's schoolwork": {
        "persona": "Parent > Academic Supporter",
        "intent": "To improve child's academic organization",
        "context": (
            "User's child struggles with schoolwork due to poor organizational skills."
        ),
    },
    "difficult feedback at work": {
        "persona": "Team Leader > Conflict Avoider",
        "intent": "To deliver constructive workplace feedback",
        "context": (
            "User avoids giving difficult feedback at work due to fear of conflict."
        ),
    },
    "revamp my wardrobe": {
        "persona": "Style Seeker > Budget-Conscious Shopper",
        "intent": "To update wardrobe on a budget",
        "context": (
            "User seeks to update their wardrobe while adhering to a limited budget."
        ),
    },
    "starting a podcast": {
        "persona": "Media Creator > Tech-Novice Podcaster",
        "intent": "To launch a podcast despite technical concerns",
        "context": (
            "User is interested in starting a podcast but is "
            "intimidated by technical aspects."
        ),
    },
}

# Validation-specific perturbation query mappings
PERTURBATION_QUERY_ADDITIONS = {
    "part-time worker": " I'm a part-time worker with a busy schedule.",
    "seeking motivation": " I need motivation to stay consistent.",
    "planning a major purchase": " We're planning a major purchase.",
    "lack of confidence in writing": " I lack confidence in my writing.",
    "personal development goals": " It's for personal development.",
    "require better preparation": " Meetings need better preparation.",
    "small intimate gathering": " It's for a small intimate gathering.",
    "focused on audience engagement": " I want to focus on audience engagement.",
    "difficulty prioritizing tasks": " My child struggles with prioritizing tasks.",
    "concern for clear communication": " I want to ensure clear communication.",
    "focus on versatile pieces": " I'm looking for versatile pieces.",
    "content creation concerns": " I'm concerned about content creation.",
}


def infer_persona_archetype(user_query: str, response: str) -> str:
    """Infers the persona archetype based on query and response content."""
    query_lower = user_query.lower()
    for pattern, data in QUERY_PATTERNS.items():
        if pattern in query_lower:
            return data["persona"]
    return DEFAULT_PERSONA


def infer_user_intent(user_query: str, response: str) -> str:
    """Infers the user's intent based on query and response."""
    query_lower = user_query.lower()
    for pattern, data in QUERY_PATTERNS.items():
        if pattern in query_lower:
            return data["intent"]
    return DEFAULT_INTENT


def infer_context_summary(user_query: str, response: str) -> str:
    """Infers the context summary based on query and response."""
    query_lower = user_query.lower()
    for pattern, data in QUERY_PATTERNS.items():
        if pattern in query_lower:
            return data["context"]
    return DEFAULT_CONTEXT


def infer_capability_layer(response: str) -> str:
    """Infers the capability layer based on response content."""
    response_lower = response.lower()
    # Direct matches
    if "foundational" in response_lower:
        return "Foundational"
    if "transformational" in response_lower:
        return "Transformational"
    if "aspirational" in response_lower:
        return "Aspirational"
    # Default based on response complexity
    if "start small" in response_lower or "simple" in response_lower:
        return "Foundational"
    if "reframe" in response_lower or "clarify" in response_lower:
        return "Transformational"
    if "growth" in response_lower or "long-term" in response_lower:
        return "Aspirational"
    return DEFAULT_CAPABILITY_LAYER


def infer_governing_principle(response: str) -> str:
    """Infers the governing principle based on response content."""
    response_lower = response.lower()
    if "people" in response_lower:
        return "Chapter 1 > People"
    if "clarity" in response_lower:
        return "Chapter 1 > Clarity"
    if "transparency" in response_lower:
        return "Chapter 1 > Transparency"
    if "process" in response_lower:
        return "Chapter 2 > Process"
    if "tools" in response_lower:
        return "Chapter 2 > Tools"
    return DEFAULT_GOVERNING_PRINCIPLE


def infer_response_attributes(response: str) -> list[str]:
    """Infers response attributes based on tone and content."""
    attributes = ["Encouraging", "Actionable"]
    response_lower = response.lower()
    if "empath" in response_lower:
        attributes.append("Empathetic")
    if "factual" in response_lower or "specific" in response_lower:
        attributes.append("Factual")
    if "proactive" in response_lower:
        attributes.append("Proactive")
    if "insight" in response_lower:
        attributes.append("Insightful")
    return attributes


def infer_chain_of_thought(response: str, user_query: str) -> list[str]:
    """Infers chain-of-thought steps based on response structure.

    Note: user_query parameter is kept for API consistency but is not currently used.
    """
    cot = []
    response_lower = response.lower()
    if "acknowledge" in response_lower or "validate" in response_lower:
        cot.append("Acknowledge the user's challenge or emotions.")
    if "reframe" in response_lower:
        cot.append("Reframe the problem to align with the 3x3 framework.")
    if "start small" in response_lower or "simple" in response_lower:
        cot.append("Suggest a small, actionable first step.")
    if "clarify" in response_lower:
        cot.append(
            "Provide clarity by asking a guiding question or breaking down the problem."
        )
    if "proactive" in response_lower or "anticipate" in response_lower:
        cot.append("Anticipate unstated needs and offer proactive solutions.")
    if not cot:
        cot.append(
            "Provide a clear, actionable response tailored to the user's intent."
        )
    return cot


def perturb_context(context: str, persona: str) -> str:
    """Applies validation-specific perturbations to generate varied contexts."""
    perturbations = [
        lambda c: (
            c.replace("busy professional", "part-time worker")
            if "Professional" in persona
            else c
        ),
        lambda c: (
            c.replace("feeling discouraged", "seeking motivation")
            if "Fitness Seeker" in persona
            else c
        ),
        lambda c: (
            c.replace("conflicting financial habits", "planning a major purchase")
            if "Relationship Manager" in persona
            else c
        ),
        lambda c: (
            c.replace("creative block", "lack of confidence in writing")
            if "Creative Professional" in persona
            else c
        ),
        lambda c: (
            c.replace("career prospects", "personal development goals")
            if "Career Changer" in persona
            else c
        ),
        lambda c: (
            c.replace("run overtime", "require better preparation")
            if "Business Professional" in persona
            else c
        ),
        lambda c: (
            c.replace("first dinner party", "small intimate gathering")
            if "Social Host" in persona
            else c
        ),
        lambda c: (
            c.replace("concerned about originality", "focused on audience engagement")
            if "Content Creator" in persona
            else c
        ),
        lambda c: (
            c.replace("poor organizational skills", "difficulty prioritizing tasks")
            if "Parent" in persona
            else c
        ),
        lambda c: (
            c.replace("fear of conflict", "concern for clear communication")
            if "Team Leader" in persona
            else c
        ),
        lambda c: (
            c.replace("limited budget", "focus on versatile pieces")
            if "Style Seeker" in persona
            else c
        ),
        lambda c: (
            c.replace("technical aspects", "content creation concerns")
            if "Media Creator" in persona
            else c
        ),
    ]
    return cast(str, random.choice(perturbations)(context))  # noqa: S311


def convert_to_icdu_validation(
    input_file: Path,
    output_file: Path,
    validation_split: float = 0.2,
    perturb: bool = True,
) -> None:
    """Converts a Breaking Better dataset to ICDU format for validation."""
    logger.info(
        f"Converting {input_file} to ICDU validation format, saving to {output_file}"
    )

    # Read all entries
    entries = []
    with open(input_file, encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                entries.append(entry)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON line: {e}")
                continue

    # Split into validation set
    _, validation_entries = train_test_split(
        entries, test_size=validation_split, random_state=42
    )
    logger.info(
        "Selected %s entries for validation (%s%% of total)",
        len(validation_entries),
        validation_split * 100,
    )

    icdu_entries = []
    for entry in validation_entries:
        if "messages" not in entry:
            logger.warning("Skipping entry: no 'messages' field")
            continue

        messages = entry["messages"]
        user_query = next((m["content"] for m in messages if m["role"] == "user"), "")
        response = next(
            (m["content"] for m in messages if m["role"] == "assistant"), ""
        )

        if not user_query or not response:
            logger.warning("Skipping entry: missing user query or response")
            continue

        # Infer ICDU fields
        persona = infer_persona_archetype(user_query, response)
        intent = infer_user_intent(user_query, response)
        context = infer_context_summary(user_query, response)
        layer = infer_capability_layer(response)
        principle = infer_governing_principle(response)
        attributes = infer_response_attributes(response)
        cot = infer_chain_of_thought(response, user_query)

        # Apply perturbation if enabled
        if perturb and random.random() < PERTURBATION_PROBABILITY:  # noqa: S311
            context = perturb_context(context, persona)
            # Adjust prompt to reflect perturbed context using dictionary lookup
            for key, addition in PERTURBATION_QUERY_ADDITIONS.items():
                if key in context:
                    user_query += addition
                    break

        icdu_entry = {
            "icdu_id": str(uuid.uuid4()),
            "persona_archetype": persona,
            "governing_principle": principle,
            "capability_layer": layer,
            "user_intent": intent,
            "context_summary": context,
            "application_prompt": user_query,
            "ideal_response_final": response,
            "ideal_response_attributes": attributes,
            "ideal_response_cot": cot,
        }
        icdu_entries.append(icdu_entry)

    with open(output_file, "w", encoding="utf-8") as f:
        for entry in icdu_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(
        f"Generated {len(icdu_entries)} ICDU validation entries in {output_file}"
    )


if __name__ == "__main__":
    input_file = Path("./breaking_better_training_data_v6.jsonl")
    output_file = Path("./icdu_validation_data.jsonl")
    convert_to_icdu_validation(
        input_file, output_file, validation_split=0.2, perturb=True
    )
