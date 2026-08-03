import json
import logging
import random
import uuid
from pathlib import Path
from typing import cast

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

# Perturbation query mappings
PERTURBATION_QUERY_ADDITIONS = {
    "student with limited time": " I'm a student with limited time.",
    "seeking quick results": " I want quick results.",
    "newly married couple": " We're newly married.",
    "tight deadline pressure": " I'm under a tight deadline.",
    "side project goals": " It's for a side project.",
    "lack focus": " Meetings often lack focus.",
    "large family gathering": " It's for a large family gathering.",
    "focused on efficiency": " I want to be more efficient.",
    "time management issues": " My child struggles with time management.",
    "desire for team harmony": " I want to maintain team harmony.",
    "sustainable fashion focus": " I'm interested in sustainable fashion.",
    "audience engagement": " I want to focus on audience engagement.",
}

# Inference mappings for better performance and maintainability
QUERY_PATTERNS = {
    "I want to get fit": {
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
    "I'm a writer and I'm completely blocked": {
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
    """Infers chain-of-thought steps based on response structure."""
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
    """Applies the Scenario-Perturbation Method to generate varied contexts."""
    perturbations = [
        lambda c: (
            c.replace("busy professional", "student with limited time")
            if "Professional" in persona
            else c
        ),
        lambda c: (
            c.replace("feeling discouraged", "seeking quick results")
            if "Fitness Seeker" in persona
            else c
        ),
        lambda c: (
            c.replace("conflicting financial habits", "newly married couple")
            if "Relationship Manager" in persona
            else c
        ),
        lambda c: (
            c.replace("creative block", "tight deadline pressure")
            if "Creative Professional" in persona
            else c
        ),
        lambda c: (
            c.replace("career prospects", "side project goals")
            if "Career Changer" in persona
            else c
        ),
        lambda c: (
            c.replace("run overtime", "lack focus")
            if "Business Professional" in persona
            else c
        ),
        lambda c: (
            c.replace("first dinner party", "large family gathering")
            if "Social Host" in persona
            else c
        ),
        lambda c: (
            c.replace("concerned about originality", "focused on efficiency")
            if "Content Creator" in persona
            else c
        ),
        lambda c: (
            c.replace("poor organizational skills", "time management issues")
            if "Parent" in persona
            else c
        ),
        lambda c: (
            c.replace("fear of conflict", "desire for team harmony")
            if "Team Leader" in persona
            else c
        ),
        lambda c: (
            c.replace("limited budget", "sustainable fashion focus")
            if "Style Seeker" in persona
            else c
        ),
        lambda c: (
            c.replace("technical aspects", "audience engagement")
            if "Media Creator" in persona
            else c
        ),
    ]
    return cast(str, random.choice(perturbations)(context))  # noqa: S311


def convert_to_icdu(input_file: Path, output_file: Path, perturb: bool = True) -> None:
    """Converts a Breaking Better dataset to ICDU format."""
    logger.info(f"Converting {input_file} to ICDU format, saving to {output_file}")
    icdu_entries = []

    with open(input_file, encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if "messages" not in entry:
                    logger.warning("Skipping entry: no 'messages' field")
                    continue

                messages = entry["messages"]
                user_query = next(
                    (m["content"] for m in messages if m["role"] == "user"), ""
                )
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
                    # Adjust prompt to reflect perturbed context
                    if "student with limited time" in context:
                        user_query += " I'm a student with limited time."
                    elif "seeking quick results" in context:
                        user_query += " I want quick results."
                    elif "newly married couple" in context:
                        user_query += " We're newly married."
                    elif "tight deadline pressure" in context:
                        user_query += " I'm under a tight deadline."
                    elif "side project goals" in context:
                        user_query += " It's for a side project."
                    elif "lack focus" in context:
                        user_query += " Meetings often lack focus."
                    elif "large family gathering" in context:
                        user_query += " It's for a large family gathering."
                    elif "focused on efficiency" in context:
                        user_query += " I want to be more efficient."
                    elif "time management issues" in context:
                        user_query += " My child struggles with time management."
                    elif "desire for team harmony" in context:
                        user_query += " I want to maintain team harmony."
                    elif "sustainable fashion focus" in context:
                        user_query += " I'm interested in sustainable fashion."
                    elif "audience engagement" in context:
                        user_query += " I want to focus on audience engagement."

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
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON line: {e}")
                continue

    with open(output_file, "w", encoding="utf-8") as f:
        for entry in icdu_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"Generated {len(icdu_entries)} ICDU entries in {output_file}")


if __name__ == "__main__":
    input_file = Path("./breaking_better_training_data_v6.jsonl")
    output_file = Path("./icdu_training_data.jsonl")
    convert_to_icdu(input_file, output_file, perturb=True)
