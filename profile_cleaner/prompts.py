"""Prompt templates for profile cleaning."""

import json


PROMPT_B = """You are a strict post-processing rewriter for image understanding profiles.

You will receive the original JSON-like image profile.

Your task is to rewrite the profile while preserving the original JSON structure and field names.

========================
Core objective
========================

Remove redundancy between IAA and IQA and strictly separate their contents.

IAA must contain only aesthetic, compositional, expressive, and viewer-response content.
IQA must contain only image quality, distortion, and technical fidelity content.

========================
Allowed content for IAA
========================

IAA may discuss:
- composition
- framing
- layout
- balance
- symmetry
- visual rhythm
- leading lines
- subject saliency
- focal point
- visual hierarchy
- color harmony
- tonal relationship
- mood
- atmosphere
- theme communication
- storytelling cues
- originality
- creativity
- artistic expression
- viewer response
- emotional tone
- overall gestalt

IAA must not discuss:
- blur
- blurriness
- low resolution
- pixelation
- noise
- grain
- compression artifacts
- sharpness
- focus
- detail loss
- texture loss
- image fidelity
- distortion
- exposure defects
- technical image quality

========================
Allowed content for IQA
========================

IQA may discuss:
- blur
- sharpness
- focus
- edge clarity
- low resolution
- pixelation
- noise
- grain
- compression artifacts
- exposure problems
- color distortion
- detail loss
- texture loss
- image fidelity
- distortion severity
- distortion location
- recognizability
- usability for downstream analysis

IQA must not discuss:
- composition
- framing
- balance
- visual rhythm
- focal point
- visual hierarchy
- creativity
- originality
- theme
- storytelling
- mood
- emotion
- viewer engagement
- artistic merit
- aesthetic impression
- gestalt

========================
Rewriting rules
========================

1. Remove all IQA-type content from IAA.
2. Remove all IAA-type content from IQA.
3. Split mixed sentences and place each part in the correct section.
4. If a quality-related sentence appears in IAA and a similar sentence already exists in IQA, delete the IAA version.
5. If an aesthetic-related sentence appears in IQA and a similar sentence already exists in IAA, delete the IQA version.
6. If a field becomes empty, fill it with a concise valid sentence based only on information supported by the original profile.
7. Preserve all original field names.
8. Preserve the original JSON hierarchy.
9. Do not add new top-level fields.
10. Do not include explanations outside the JSON.
11. Keep the style concise and bullet-based if the original profile uses bullets.
12. Do not overstate certainty.
13. Avoid absolute phrases such as:
    - no artistic intent
    - fails to evoke emotion
    - negligible aesthetic merit
    - completely unusable
    unless they are explicitly justified by the original profile.

========================
Output
========================

Return only the cleaned JSON profile.

Original profile:
{{PROFILE_JSON}}"""


PROMPT_C = """You are a strict validator and repair agent for cleaned image profiles.

You will receive a cleaned JSON-like image profile.

Your task is to verify whether profile.iaa and profile.iqa are strictly separated.

If violations exist, repair them while preserving the original structure and field names.

========================
Hard constraints
========================

IAA must not contain any IQA terms or concepts.

Forbidden in IAA:
blur, blurry, blurriness, low resolution, resolution, pixelation, noise, grain, compression, artifact, sharpness, focus, detail loss, texture loss, fidelity, distortion, overexposure, underexposure, exposure defect, image quality, technical quality.

IQA must not contain any IAA terms or concepts.

Forbidden in IQA:
composition, framing, layout, balance, symmetry, rhythm, leading lines, focal point, visual hierarchy, artistic, aesthetic, creativity, originality, theme, storytelling, narrative, mood, atmosphere, emotion, viewer, engagement, memorability, gestalt.

========================
Repair rules
========================

1. If an IAA sentence contains forbidden IQA content, rewrite it into a valid aesthetic sentence or delete it.
2. If an IQA sentence contains forbidden IAA content, rewrite it into a valid image-quality sentence or delete it.
3. If a sentence cannot be repaired without crossing section boundaries, delete it.
4. If a field becomes empty, add one concise valid sentence that fits the field.
5. Do not add new fields.
6. Do not rename fields.
7. Do not change the JSON hierarchy.
8. Return only the repaired JSON.
9. Do not include explanations.

========================
Final self-check before output
========================

Before returning the JSON, internally verify:
- No IAA forbidden terms remain in profile.iaa.
- No IQA forbidden terms remain in profile.iqa.
- No repeated sentence appears across IAA and IQA.
- Each IAA field contains only aesthetic/compositional/expressive content.
- Each IQA field contains only quality/distortion/fidelity content.

Now validate and repair this profile:

{{CLEANED_PROFILE_JSON}}"""


JSON_REPAIR_PROMPT = """You are a JSON structure repair agent.

You will receive an invalid or incomplete cleaned image profile.

Your task:
1. Restore valid JSON format.
2. Preserve the original profile hierarchy as much as possible.
3. Ensure profile.iaa and profile.iqa both exist.
4. Do not add explanatory text.
5. Do not change content unless required to fix JSON validity.
6. Return only valid JSON.

Input:

{{BROKEN_OUTPUT}}"""


def _format_json(value):
    return json.dumps(value, ensure_ascii=False, indent=2)


def render_prompt_b(profile):
    """Render the structure-preserving rewrite prompt."""
    return PROMPT_B.replace("{{PROFILE_JSON}}", _format_json(profile))


def render_prompt_c(cleaned_profile):
    """Render the strict validation and repair prompt."""
    return PROMPT_C.replace("{{CLEANED_PROFILE_JSON}}", _format_json(cleaned_profile))


def render_json_repair_prompt(raw_output):
    """Render a prompt that asks the model to repair malformed JSON."""
    return JSON_REPAIR_PROMPT.replace("{{BROKEN_OUTPUT}}", str(raw_output or ""))
