"""Comprehensive ICDU dataset generation pipeline.

This script provides a data generation pipeline that combines functionalities from
several data processing scripts. It supports various modes of operation including
data augmentation, ICDU conversion, and the generation of proactive, persona-aware
responses.

The script processes JSONL input files and generates ICDU-formatted datasets with:
- Persona archetype inference
- Governing principle assignment
- Capability layer classification
- Context-aware response generation
- Data augmentation and deduplication

Usage:
    python master_generate_icdu.py \\
        --input-file bb_training_data_v7.jsonl \\
        --output-dir data/generated_data \\
        --augmentation-factor 20 \\
        --validation-split 0.1 \\
        --num-processes 8 \\
        --run-reports
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
from typing import Any, cast

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]
from tqdm import tqdm
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# Constants
# Fallback when not in config: HF/LLMStudio default
DEFAULT_MODEL_NAME = "qwen/qwen3-vl-8b"
DEFAULT_MAX_TOKEN_LENGTH = 32000
DEFAULT_AUGMENTATION_FACTOR = 10
DEFAULT_VALIDATION_SPLIT = 0.1
DEFAULT_NUM_VARIANTS = 5
DEFAULT_DIVERSITY_THRESHOLD = 0.7
DEFAULT_RANDOM_STATE = 42

# Predefined tools for augmentation
DEFAULT_TOOLS = [
    "search_web",
    "calc_tool",
    "news_tool",
    "python_repl",
    "read_file",
    "write_file",
    "calendar_tool",
    "task_tracker_tool",
    "job_search_tool",
    "get_current_weather",
    "animal_medical_database",
]

# Default persona for unmatched queries
DEFAULT_PERSONA = "General User > Problem Solver"

# Tokenizer for validation (lazy-loaded)
_TOKENIZER: AutoTokenizer | None = None


def get_tokenizer() -> AutoTokenizer | None:
    """Get or initialize the tokenizer for validation.

    Returns:
        AutoTokenizer instance or None if initialization fails.
    """
    global _TOKENIZER
    if _TOKENIZER is None:
        try:
            _TOKENIZER = cast(
                AutoTokenizer,
                AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME),
            )
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
    return _TOKENIZER


# Persona archetype patterns for inference
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
        r"\b(client|team|manager|boss|meeting|leadership|colleague|"
        r"project|delegate|morale)\b"
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
        r"\b(student|exam|study|overachiever|school|test|grades|college|"
        r"homework|procrastinate)\b"
    ),
    "Executive > Strategic Planner": (
        r"\b(strategy|executive|plan|vision|business|decision|lead|goal|"
        r"prioritize|forecast)\b"
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
    "Traveler > Budget Backpacker": (
        r"\b(travel|trip|vacation|backpack|budget|destination|itinerary|"
        r"hostel|explore|adventure)\b"
    ),
    "Gamer > Competitive Player": (
        r"\b(game|esports|console|pc|strategy|multiplayer|rank|compete|"
        r"tournament|stream)\b"
    ),
    "Investor > Beginner Stock Trader": (
        r"\b(stock|market|invest|trade|portfolio|dividend|broker|risk|analyze|buy)\b"
    ),
    "Language Learner > Polyglot Aspirant": (
        r"\b(language|learn spanish|vocabulary|grammar|fluent|practice|"
        r"app|converse|translate)\b"
    ),
    "Environmentalist > Eco-Warrior": (
        r"\b(environment|sustainability|recycle|climate|green|activism|"
        r"reduce|waste|eco|planet)\b"
    ),
    "Musician > Garage Band Member": (
        r"\b(music|band|instrument|compose|gig|rehearse|song|record|perform|jam)\b"
    ),
    "Book Lover > Avid Reader": (
        r"\b(book|read|novel|author|library|genre|review|series|bestseller|club)\b"
    ),
    "Tech Enthusiast > Gadget Reviewer": (
        r"\b(tech|gadget|review|smartphone|laptop|specs|upgrade|unbox|compare|feature)\b"
    ),
    "Foodie > Restaurant Explorer": (
        r"\b(food|restaurant|cuisine|dine|review|taste|menu|chef|gourmet|reservation)\b"
    ),
    "Artist > Digital Illustrator": (
        r"\b(art|draw|illustrate|digital|tablet|software|sketch|design|create|portfolio)\b"
    ),
    "Freelancer > Gig Economy Worker": (
        r"\b(freelance|gig|client|project|invoice|remote|skill|platform|bid|deadline)\b"
    ),
    "Volunteer > Community Organizer": (
        r"\b(volunteer|community|organize|event|cause|help|donate|participate|"
        r"impact|group)\b"
    ),
    "Cyclist > Urban Commuter": (
        r"\b(bike|cycle|commute|trail|gear|helmet|route|maintenance|pedal|tour)\b"
    ),
    "Photographer > Amateur Shooter": (
        r"\b(photography|camera|shoot|lens|edit|composition|light|capture|"
        r"portfolio|snap)\b"
    ),
    "Gardener > Urban Planter": (
        r"\b(garden|plant|soil|grow|vegetable|flower|herb|compost|harvest|landscape)\b"
    ),
    "Podcaster > Episode Host": (
        r"\b(podcast|episode|host|record|edit|guest|topic|audience|publish|subscribe)\b"
    ),
    "Runner > Marathon Trainee": (
        r"\b(run|marathon|train|pace|mile|shoe|endurance|race|hydrate|track)\b"
    ),
    "Writer > Freelance Blogger": (
        r"\b(write|blog|article|post|topic|research|edit|publish|audience|seo)\b"
    ),
    "Teacher > Online Educator": (
        r"\b(teach|online|lesson|student|class|curriculum|assign|grade|engage|platform)\b"
    ),
    "Parent > New Mom/Dad": (
        r"\b(baby|parent|newborn|diaper|sleep|feed|milestone|advice|routine|care)\b"
    ),
    "Sustainability Advocate > Zero-Waste Practitioner": (
        r"\b(sustainable|zero waste|reuse|compost|plastic-free|eco-friendly|"
        r"lifestyle|reduce|shop|habit)\b"
    ),
    "Fitness Coach > Online Trainer": (
        r"\b(coach|train|session|client|program|motivate|progress|exercise|"
        r"nutrition|virtual)\b"
    ),
    "Software Developer > Open-Source Contributor": (
        r"\b(code|develop|open source|repo|pull request|bug|feature|commit|"
        r"collaborate|fork)\b"
    ),
}


def infer_persona_archetype(user_query: str) -> str:
    """Infer a persona archetype from the user's query using regex patterns.

    Args:
        user_query: The user's query text.

    Returns:
        Persona archetype string (e.g., "Fitness Seeker > Struggling Starter").
    """
    user_query_lower = user_query.lower()
    for persona, pattern in PERSONA_PATTERNS.items():
        if re.search(pattern, user_query_lower):
            return persona
    return DEFAULT_PERSONA


# Governing principle patterns
GOVERNING_PRINCIPLES: dict[str, list[str]] = {
    "Chapter 1": ["People", "Clarity", "Transparency"],
    "Chapter 2": ["People", "Process", "Tools"],
    "Chapter 3": ["Foundational", "Transformational", "Aspirational"],
}

CHAPTER_PATTERNS: dict[str, str] = {
    "Chapter 1": (
        r"\b(relationship|team|partner|people|colleague|social|trust|bond|communicate)\b"
    ),
    "Chapter 2": (
        r"\b(process|tool|method|step|habit|routine|system|workflow|optimize)\b"
    ),
    "Chapter 3": (
        r"\b(long-term|vision|growth|aspiration|transform|future|goal|dream|evolve)\b"
    ),
}


def infer_governing_principle(
    user_query: str, chapter_weights: dict[str, float] | None = None
) -> str:
    """Infer a governing principle based on the query.

    Args:
        user_query: The user's query text.
        chapter_weights: Optional weights for chapter selection (default: equal).

    Returns:
        Governing principle string (e.g., "Chapter 1 > People").
    """
    if chapter_weights is None:
        chapter_weights = {}

    user_query_lower = user_query.lower()

    # Try to match chapter patterns
    for chapter, pattern in CHAPTER_PATTERNS.items():
        if re.search(pattern, user_query_lower):
            sub_principle = random.choice(GOVERNING_PRINCIPLES[chapter])  # noqa: S311
            return f"{chapter} > {sub_principle}"

    # Fallback to weighted random selection
    chapters = list(GOVERNING_PRINCIPLES.keys())
    weights = [chapter_weights.get(ch, 1.0) for ch in chapters]
    chapter = random.choices(chapters, weights=weights)[0]  # noqa: S311
    sub_principle = random.choice(GOVERNING_PRINCIPLES[chapter])  # noqa: S311
    return f"{chapter} > {sub_principle}"


# User intent mappings
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
    "travel": "To plan a budget-friendly trip or vacation.",
    "game": "To improve gaming skills and compete at a higher level.",
    "stock": "To learn the basics of stock trading and start investing.",
    "language": "To effectively learn a new language.",
    "environment": "To adopt a more sustainable and eco-friendly lifestyle.",
    "music": "To improve musical skills and collaborate with a band.",
    "book": "To find new books and engage with a reading community.",
    "gadget": "To stay updated on the latest technology and gadgets.",
    "food": "To discover new restaurants and cuisines.",
    "art": "To develop skills in digital art and illustration.",
    "freelance": "To successfully navigate the freelance and gig economy.",
    "volunteer": "To organize community events and volunteer effectively.",
    "bike": "To use a bicycle for commuting and recreation safely.",
    "photography": "To improve photography skills.",
    "garden": "To start and maintain a garden in an urban environment.",
    "episode": "To create and manage a successful podcast.",
    "run": "To train for a marathon or long-distance race.",
    "blog": "To start and grow a successful blog.",
    "teach": "To teach effectively in an online environment.",
    "baby": "To navigate the challenges of being a new parent.",
    "sustainable": "To practice a zero-waste lifestyle.",
    "coach": "To effectively coach clients in an online setting.",
    "open source": "To contribute to open-source software projects.",
}

DEFAULT_USER_INTENT = "To solve a personal or professional challenge"


def infer_user_intent(user_query: str) -> str:
    """Infer the user's intent from their query.

    Args:
        user_query: The user's query text.

    Returns:
        User intent string describing the user's goal.
    """
    user_query_lower = user_query.lower()
    for key, intent in USER_INTENTS.items():
        if key in user_query_lower:
            return intent
    return DEFAULT_USER_INTENT


def infer_context_summary(user_query: str) -> str:
    """Generates a context summary for the user's query."""
    user_query = user_query.lower()
    base_contexts = {
        "get fit": (
            "User is motivated to improve their health and is "
            "looking for an effective fitness routine."
        ),
        "money": (
            "User and their partner are working to align their "
            "financial habits and goals."
        ),
        "writer blocked": (
            "User is a writer looking for new techniques to overcome a creative block."
        ),
        "learn to code": (
            "User is exploring coding as a way to advance their "
            "career and learn a new skill."
        ),
        "procrastination": (
            "User is seeking strategies to improve productivity "
            "and overcome procrastination."
        ),
        "cook": (
            "User enjoys cooking and is looking for ways to make "
            "weekly meal planning more efficient and enjoyable."
        ),
        "diy": (
            "User is a new homeowner, excited to start some DIY "
            "projects and looking for guidance."
        ),
        "meditation": (
            "User is interested in the benefits of meditation "
            "and wants to build a consistent practice."
        ),
        "pet": (
            "User has recently adopted a new pet and is eager to "
            "learn about positive reinforcement training."
        ),
        "graduate": (
            "User is a recent graduate, actively applying for jobs "
            "and looking to improve their resume."
        ),
        "retire": (
            "User is planning for retirement and exploring how "
            "to best manage their finances and time."
        ),
        "event plan": (
            "User is organizing a major event and wants to ensure it runs smoothly."
        ),
        "hobby": (
            "User is excited to pick up a new skill and is "
            "looking for the best way to get started."
        ),
        "travel": (
            "User is planning an upcoming trip and is looking "
            "for creative ways to travel on a budget."
        ),
        "game": (
            "User is a passionate gamer who wants to improve "
            "their skills and become more competitive."
        ),
        "stock": (
            "User is new to investing and is looking for a "
            "clear, simple way to get started in the stock market."
        ),
        "language": (
            "User is enthusiastic about learning a new language "
            "and is exploring different learning methods."
        ),
        "environment": (
            "User is passionate about sustainability and is "
            "looking for practical ways to reduce their environmental impact."
        ),
        "music": (
            "User is in a new band, and they are collaboratively "
            "working on writing their first song."
        ),
        "book": (
            "User is an avid reader who enjoys discovering new authors and genres."
        ),
        "gadget": (
            "User is a tech enthusiast who likes to stay "
            "informed about the latest gadgets."
        ),
        "food": (
            "User is a foodie who loves exploring new restaurants "
            "and sharing their experiences."
        ),
        "art": (
            "User is an aspiring digital artist who is practicing "
            "their skills and building a portfolio."
        ),
        "freelance": (
            "User is a freelancer building their business and "
            "looking for ways to attract new clients."
        ),
        "volunteer": (
            "User is looking for meaningful ways to get involved and "
            "make a positive impact in their community."
        ),
        "bike": (
            "User is interested in starting to commute by bike and is "
            "looking for tips on safety and route planning."
        ),
        "photography": (
            "User has a new camera and is excited to learn the "
            "fundamentals of photography."
        ),
        "garden": (
            "User lives in an apartment and is looking for creative ways "
            "to start a small balcony garden."
        ),
        "podcast": (
            "User is passionate about a topic and is starting a podcast "
            "to share their knowledge."
        ),
        "run": (
            "User has signed up for their first marathon and is "
            "committed to training for it."
        ),
        "blog": (
            "User wants to start a blog to share their expertise and is "
            "looking for content ideas."
        ),
        "teach": (
            "User is a teacher who is adapting their lessons for an "
            "online format and wants to keep students engaged."
        ),
        "baby": (
            "User is a new parent, navigating the joys and challenges "
            "of caring for a newborn."
        ),
        "sustainable": (
            "User is committed to a zero-waste lifestyle and is "
            "looking for new tips and ideas."
        ),
        "coach": (
            "User is a fitness coach who is transitioning their "
            "services online to reach more clients."
        ),
        "open source": (
            "User is a software developer who wants to give back to the "
            "community by contributing to open-source projects."
        ),
    }
    key = next((k for k in base_contexts if k in user_query), None)
    default_ctx = "User is facing a general challenge requiring practical guidance."
    base = base_contexts.get(key, default_ctx) if key is not None else default_ctx
    perturbations = [
        " The user is feeling particularly motivated this week.",
        " The user has tried other solutions in the past and is "
        "looking for a fresh approach.",
    ]
    return base + random.choice(perturbations)  # noqa: S311


def infer_capability_layer(response: str) -> str:
    """Infers the capability layer from the response content."""
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
    return random.choice(  # noqa: S311
        ["Foundational", "Transformational", "Aspirational"]
    )


def infer_response_attributes(layer: str) -> list[str]:
    """Assigns attributes to the response based on its capability layer."""
    attr_pool = {
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
    return random.sample(attr_pool.get(layer, attr_pool["Foundational"]), 3)


def infer_response_cot(layer: str) -> list[str]:
    """Generates a chain-of-thought for the AI's reasoning process."""
    cot_pool = {
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
    steps = cot_pool.get(layer, cot_pool["Foundational"])
    return random.sample(steps, random.randint(3, len(steps)))  # noqa: S311


def generate_follow_up_question(persona: str) -> str:
    """Generates a proactive, persona-aware follow-up question."""
    question_bank = {
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
        "Traveler": [
            "Would you like to explore some specific budget-friendly "
            "destinations or travel hacks?"
        ],
        "Gamer": [
            "Are you interested in drills for a specific game or "
            "general strategies for improving reaction time?"
        ],
        "Investor": [
            "Would it be helpful to create a sample diversified "
            "portfolio based on your risk tolerance?"
        ],
        "Language Learner": [
            "Would you like to create a weekly practice schedule to stay consistent?"
        ],
        "Environmentalist": [
            "Would you be interested in a room-by-room guide to reducing waste at home?"
        ],
        "Musician": [
            "Would you like to explore some common song structures "
            "or chord progressions to get started?"
        ],
        "Book Lover": [
            "Are you interested in recommendations for a specific genre or author?"
        ],
        "Tech Enthusiast": [
            "Is there a specific category of gadget, like smartphones "
            "or headphones, you'd like to dive into?"
        ],
        "Foodie": [
            "Are you looking for recommendations in a particular "
            "neighborhood or for a specific type of cuisine?"
        ],
        "Artist": [
            "Would you like to focus on a specific skill, like "
            "character design or color theory?"
        ],
        "Freelancer": [
            "Would you be interested in tips for creating a "
            "compelling portfolio or writing effective client proposals?"
        ],
        "Volunteer": [
            "Are you passionate about a particular cause, like "
            "animal welfare or education, that we could focus on?"
        ],
        "Cyclist": [
            "Would you like to map out a safe commuting route or "
            "discuss essential safety gear?"
        ],
        "Photographer": [
            "Would you like to learn about the exposure triangle—"
            "aperture, shutter speed, and ISO?"
        ],
        "Gardener": [
            "Would you be interested in a list of easy-to-grow plants for beginners?"
        ],
        "Podcaster": [
            "Would you like to brainstorm some potential episode "
            "topics or formats for your show?"
        ],
        "Runner": [
            "Would it be helpful to break down the training plan "
            "into weekly mileage goals?"
        ],
        "Writer": [
            "Would you like to brainstorm some blog post ideas based on your interests?"
        ],
        "Teacher": [
            "Are you interested in some icebreaker activities or "
            "tools for virtual classrooms?"
        ],
        "Parent": [
            "Would you like to discuss creating a flexible sleep or feeding schedule?"
        ],
        "Sustainability Advocate": [
            "Would you like to start with a specific area, like the "
            "kitchen or bathroom, to implement zero-waste practices?"
        ],
        "Fitness Coach": [
            "Would you be interested in exploring some tools for "
            "conducting virtual training sessions?"
        ],
        "Software Developer": [
            "Are you interested in finding beginner-friendly issues "
            "on platforms like GitHub?"
        ],
        "default": [
            "Does this initial step feel manageable for you?",
            "What's one potential obstacle you foresee, and how can we plan for it?",
        ],
    }
    key = next((k for k in question_bank if k in persona), "default")
    return random.choice(question_bank[key])  # noqa: S311


# --- Data Augmentation and Utility Functions ---
def load_jsonl(file_path: str) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dictionaries.

    Args:
        file_path: Path to the JSONL file.

    Returns:
        List of parsed JSON objects.

    Raises:
        ValueError: If the file does not exist.
    """
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        raise ValueError(f"Input file not found: {file_path}")

    data: list[dict[str, Any]] = []
    with open(file_path_obj, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Skipping invalid JSON line {line_num} in {file_path}: {e}"
                )
    logger.info(f"Loaded {len(data)} entries from {file_path}")
    return data


def save_jsonl(data: list[dict[str, Any]], file_path: str) -> None:
    """Save a list of dictionaries to a JSONL file.

    Args:
        data: List of JSON-serializable dictionaries.
        file_path: Path to output JSONL file.
    """
    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(data)} entries to {file_path}")


def paraphrase_response(response: str) -> str:
    """Performs simple, rule-based paraphrasing."""
    synonyms = {
        "start": ["begin", "initiate", "commence"],
        "small": ["tiny", "minimal", "modest"],
        "habit": ["routine", "practice", "pattern"],
        "process": ["method", "approach", "system"],
    }
    words = response.split()
    paraphrased_words = [
        random.choice(synonyms.get(word.lower().strip(".,?!"), [word]))  # noqa: S311
        for word in words
    ]
    return " ".join(paraphrased_words)


def get_augmentations(persona: str, context: str, prompt: str) -> list[tuple[str, str]]:
    """Generates augmented context and prompt pairs."""
    base_augs = [
        (
            f"{context} The user has unlimited resources.",
            f"{prompt} What if budget wasn't an issue?",
        ),
        (
            f"{context} The user is facing a tight deadline.",
            f"{prompt} I need a solution that is fast and effective.",
        ),
    ]
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
    """Deduplicates a list of ICDU entries."""
    seen: set[str] = set()
    unique_entries = []
    for entry in entries:
        key = json.dumps(
            {
                "p": entry.get("persona_archetype", ""),
                "q": entry.get("application_prompt", ""),
                "r": entry.get("ideal_response_final", ""),
            },
            sort_keys=True,
        )
        if key not in seen:
            seen.add(key)
            unique_entries.append(entry)
    return unique_entries


# --- Reporting and Visualization Functions ---
def compute_diversity(
    entries: list[dict[str, Any]], field: str, threshold: float = 0.7
) -> dict[str, Any]:
    """Computes text diversity metrics."""
    values = [entry.get(field, "") for entry in entries]
    if not values:
        return {"unique_ratio": 0, "avg_jaccard_similarity": 0, "is_diverse": False}

    # Filter out empty values for diversity calculation
    non_empty_values = [v for v in values if v]
    if not non_empty_values:
        # All values are empty - no diversity
        return {"unique_ratio": 0, "avg_jaccard_similarity": 0, "is_diverse": False}

    unique_ratio = (
        len(set(non_empty_values)) / len(non_empty_values) if non_empty_values else 0
    )
    jaccards = []
    num_samples = min(len(non_empty_values) * (len(non_empty_values) - 1) // 2, 10000)
    if num_samples > 0 and len(non_empty_values) >= 2:
        for _ in range(num_samples):
            i, j = random.sample(range(len(non_empty_values)), 2)
            set_i, set_j = (
                set(non_empty_values[i].split()),
                set(non_empty_values[j].split()),
            )
            intersection = len(set_i & set_j)
            union = len(set_i | set_j)
            jaccards.append(intersection / union if union else 0)
    avg_jaccard = sum(jaccards) / len(jaccards) if jaccards else 0
    return {
        "unique_ratio": unique_ratio,
        "avg_jaccard_similarity": avg_jaccard,
        "is_diverse": avg_jaccard < threshold,
    }


def visualize_stats(stats: dict[str, Any], output_dir: Path) -> None:
    """Generates and saves bar plots for dataset distributions."""
    for key, counts in stats.items():
        if not counts or counts is None:
            continue
        if not isinstance(counts, dict):
            continue
        if not counts.keys():
            continue
        try:
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
            logger.warning(f"Failed to generate plot for {key}: {e}")
            continue


def run_dataset_evaluation(
    entries: list[dict[str, Any]],
    output_dir: Path,
    diversity_threshold: float = 0.7,
) -> None:
    """Runs all evaluation and reporting tasks."""
    logger.info("--- Starting Dataset Evaluation ---")
    diversity_metrics = {
        "prompts_diversity": compute_diversity(
            entries, "application_prompt", diversity_threshold
        ),
        "responses_diversity": compute_diversity(
            entries, "ideal_response_final", diversity_threshold
        ),
    }
    metrics_path = output_dir / "evaluation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(diversity_metrics, f, indent=4)
    logger.info(f"Diversity metrics saved to {metrics_path}")

    stats = {
        "persona_archetypes": Counter(
            e["persona_archetype"] for e in entries if "persona_archetype" in e
        ),
        "governing_principles": Counter(
            e["governing_principle"] for e in entries if "governing_principle" in e
        ),
        "capability_layers": Counter(
            e["capability_layer"] for e in entries if "capability_layer" in e
        ),
    }
    visualize_stats(stats, output_dir)
    logger.info("--- Dataset Evaluation Complete ---")


# --- Main Conversion and Processing Logic ---
def convert_to_icdu(
    entry: dict[str, Any],
    augmentation: tuple[str, str] | None = None,
) -> dict[str, Any] | None:
    """Converts a source entry into the ICDU format."""
    messages = entry.get("messages", [])
    original_prompt = next((m["content"] for m in messages if m["role"] == "user"), "")
    original_response = next(
        (m["content"] for m in messages if m["role"] == "assistant"), ""
    )
    if not original_prompt or not original_response:
        return None

    context, prompt = (
        augmentation
        if augmentation
        else (infer_context_summary(original_prompt), original_prompt)
    )
    persona = infer_persona_archetype(prompt)
    layer = infer_capability_layer(original_response)
    perturbed_response = paraphrase_response(original_response)
    follow_up_question = generate_follow_up_question(persona)
    final_response_with_question = f"{perturbed_response} {follow_up_question}"

    return {
        "icdu_id": str(uuid.uuid4()),
        "persona_archetype": persona,
        "governing_principle": infer_governing_principle(prompt, {}),
        "capability_layer": layer,
        "user_intent": infer_user_intent(prompt),
        "context_summary": context,
        "application_prompt": prompt,
        "ideal_response_final": final_response_with_question,
        "ideal_response_attributes": infer_response_attributes(layer),
        "ideal_response_cot": infer_response_cot(layer),
    }


def update_icdu_entry(entry: dict[str, Any]) -> dict[str, Any] | None:
    """Updates an existing ICDU-formatted entry."""
    original_prompt = entry.get("application_prompt", "")
    original_response = entry.get("ideal_response_final", "")
    if not original_prompt or not original_response:
        return None

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
    """Worker function for parallel processing."""
    entry, config = args
    if "messages" in entry:
        base_icdu = convert_to_icdu(entry)
        if not base_icdu:
            return []
        processed_entries = [base_icdu]
        if config.get("augmentation_factor", 1) > 1:
            augmentations = get_augmentations(
                base_icdu["persona_archetype"],
                base_icdu["context_summary"],
                base_icdu["application_prompt"],
            )
            num_to_generate = config["augmentation_factor"] - 1
            selected_augs = random.choices(  # noqa: S311
                augmentations, k=num_to_generate
            )
            for aug in selected_augs:
                augmented_icdu = convert_to_icdu(entry, aug)
                if augmented_icdu:
                    processed_entries.append(augmented_icdu)
        return processed_entries
    else:
        updated_entry = update_icdu_entry(entry)
        return [updated_entry] if updated_entry else []


# --- Main Function ---
def main() -> None:
    """Orchestrate the dataset generation process.

    Main entry point that handles argument parsing, data loading, processing,
    and output generation with optional reporting and visualization.
    """
    parser = argparse.ArgumentParser(
        description="Generate or update an ICDU dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="Path to the source JSONL file",
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
        help="Number of augmentations per source entry",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=DEFAULT_VALIDATION_SPLIT,
        help="Fraction of data for the validation set",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=cpu_count(),
        help="Number of CPU cores for parallel processing",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=None,
        help="Desired number of samples in the final dataset",
    )
    parser.add_argument(
        "--run-reports",
        action="store_true",
        help="Generate and save evaluation reports and plots",
    )
    parser.add_argument(
        "--num_variants",
        type=int,
        default=DEFAULT_NUM_VARIANTS,
        help="Variants per original example for tool-calling fine-tuning",
    )
    parser.add_argument(
        "--tools",
        default=",".join(DEFAULT_TOOLS),
        help="Comma-separated tools for augmentation",
    )
    args = parser.parse_args()

    if not args.input_file.exists():
        logger.error(f"Input file not found: {args.input_file}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        source_entries = load_jsonl(str(args.input_file))
    except ValueError as e:
        logger.error(f"Failed to load input file: {e}")
        return

    if not source_entries:
        logger.error("No entries found in input file")
        return

    processing_config = {"augmentation_factor": args.augmentation_factor}

    logger.info(
        f"Starting parallel processing with {args.num_processes} cores "
        f"on {len(source_entries)} source entries..."
    )

    with Pool(args.num_processes) as pool:
        task_args = [(e, processing_config) for e in source_entries]
        results = list(
            tqdm(
                pool.imap(process_entry, task_args),
                total=len(source_entries),
                desc="Processing entries",
            )
        )

    all_icdu_entries = [item for sublist in results for item in sublist if item]
    unique_entries = ensure_uniqueness(all_icdu_entries)

    logger.info(
        f"Generated {len(all_icdu_entries)} total entries, "
        f"{len(unique_entries)} unique entries"
    )

    if args.dataset_size and args.dataset_size > 0:
        if not unique_entries:
            logger.error("No unique entries were generated to sample from")
            return
        final_entries = random.sample(
            unique_entries, min(args.dataset_size, len(unique_entries))
        )
        logger.info(
            f"Sampled {len(final_entries)} entries from {len(unique_entries)} unique"
        )
    else:
        final_entries = unique_entries

    if not final_entries:
        logger.error("No entries were generated or sampled")
        return

    if args.run_reports:
        run_dataset_evaluation(final_entries, args.output_dir)

    train_data, val_data = train_test_split(
        final_entries,
        test_size=args.validation_split,
        random_state=DEFAULT_RANDOM_STATE,
    )

    train_file = args.output_dir / "icdu_training_data.jsonl"
    val_file = args.output_dir / "icdu_validation_data.jsonl"

    save_jsonl(train_data, str(train_file))
    save_jsonl(val_data, str(val_file))

    logger.info(
        f"Successfully generated datasets. "
        f"Total: {len(final_entries)}, "
        f"Training: {len(train_data)}, "
        f"Validation: {len(val_data)}"
    )
    logger.info(f"Files saved to {train_file} and {val_file}")


if __name__ == "__main__":
    main()
