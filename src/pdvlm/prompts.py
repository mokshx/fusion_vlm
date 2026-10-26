"""Prompt variants. Keep these strings frozen once the main grid starts: any edit
invalidates comparisons with earlier runs, so add a new key instead."""

PROMPTS = {
    # (A) plain question, verbatim from the study design.
    "A": "Is any person fallen or lying on the floor? Answer yes or no.",
    # (B) explicit definition that names the hard-negative postures.
    "B": (
        "You are a workplace safety monitor looking at a camera image. "
        "A 'person down' is a person whose body is lying on the floor or ground, "
        "with the torso roughly horizontal and touching or very close to the floor, "
        "for example after a fall or collapse. "
        "The following are NOT a person down: standing, walking, bending over, "
        "kneeling, crouching or squatting, sitting on the floor or on a chair, "
        "lying on a bed, sofa or other furniture, and lying under equipment "
        "while visibly doing maintenance work. "
        "Is there a person down in this image? Answer with one word: yes or no."
    ),
    # (B2) short definition by a single physical criterion. Added after Step 0
    # showed that listing look-alike postures in B primes "yes"; B2 names none,
    # and its criterion matches the YOLO-pose baseline rule (torso on the floor).
    "B2": (
        "A person is 'down' only if their torso is lying on the floor or ground. "
        "Is any person down? Answer yes or no."
    ),
}

MULTI_FRAME_PREFIX = (
    "The following {n} images are consecutive frames from one camera, in time order. "
    "Answer about the last frame, using earlier frames as context.\n"
)


def build_prompt(key: str, n_frames: int = 1) -> str:
    text = PROMPTS[key]
    if n_frames > 1:
        text = MULTI_FRAME_PREFIX.format(n=n_frames) + text
    return text
