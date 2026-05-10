# prompts.py
"""
Structured prompts for AIDE-Emotion.

Three semantic groups, each with 3 templates, total P=9 prompts per class.
The grouping is for analysis/visualization — the model still sees them as
9 independent prompts. Group identity is exposed via PROMPT_GROUP_IDS for
later attention analysis (which group does the model attend to per class?).

Group A — Facial / expression cues (3)
Group B — Body / posture / behavioral cues (3)
Group C — Holistic / scene / global cues (3)
"""

# {LABEL} placeholder — filled per emotion class at encoding time
PROMPT_TEMPLATES = [
    # --- Group A: facial ---
    "The driver's facial expression shows {LABEL}.",
    "The driver's eyes and mouth indicate {LABEL}.",
    "Looking at the face, the driver appears {LABEL}.",
    # --- Group B: body / behavior ---
    "The driver's posture and body language convey {LABEL}.",
    "From the shoulders and hands on the wheel, the driver seems {LABEL}.",
    "The driver's overall behavior in the cabin suggests {LABEL}.",
    # --- Group C: holistic / scene ---
    "In this in-cabin video, the driver's emotional state is {LABEL}.",
    "This driving clip shows a driver who is {LABEL}.",
    "Current driver affect: {LABEL}.",
]

# Group ids for each prompt position (for analysis only, NOT used in attention)
PROMPT_GROUP_IDS = [0, 0, 0, 1, 1, 1, 2, 2, 2]
PROMPT_GROUP_NAMES = ["facial", "body", "holistic"]

NUM_PROMPTS = len(PROMPT_TEMPLATES)  # 9
NUM_GROUPS = len(PROMPT_GROUP_NAMES)  # 3


def build_prompts_for_classes(class_names):
    """
    Args:
        class_names: list of str, e.g. ['anxiety','peace','weariness','happiness','anger']
    Returns:
        list of list of str, shape [C, P]
    """
    return [
        [tpl.format(LABEL=name) for tpl in PROMPT_TEMPLATES]
        for name in class_names
    ]