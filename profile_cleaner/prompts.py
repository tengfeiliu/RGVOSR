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
9. Add exactly one new field at the returned profile root: suggestion.
10. Do not include explanations outside the JSON.
11. Keep the style concise and bullet-based if the original profile uses bullets.
12. Do not overstate certainty.
13. Avoid absolute phrases such as:
    - no artistic intent
    - fails to evoke emotion
    - negligible aesthetic merit
    - completely unusable
    unless they are explicitly justified by the original profile.
14. Keep profile.ista unchanged when it exists.
15. Do not add any other new fields.

========================
Token budgets, placement, and suggestion constraints
========================

Length limits below are token budgets, not character budgets.
Estimate tokens as the downstream FLUX.2/Qwen tokenizer would tokenize the returned English text.
Do not count Unicode characters or bytes.

IAA output must be very compact:
- The entire profile.iaa section must be no more than 50 tokens total.
- Put the IAA summary in profile.iaa.comprehensive when that field exists.
- Set the other string fields under profile.iaa to "".
- Summarize only the key aesthetic, composition, atmosphere, or expressive evidence.

IQA output must be informative but bounded:
- The entire profile.iqa section should be around 370 tokens.
- Keep it concise, but do not make it too short if the original profile contains enough quality evidence.
- Preserve the original IQA field names, especially distortion_location, distortion_severity, distortion_type, and overall_quality when they exist.
- Distribute IQA content across those fields within the total token budget.
- Cover distortion location, distortion type, severity, and overall quality impact.
- Do not invent unsupported quality evidence just to reach the target length.

Suggestion output:
- Add a profile-level field named suggestion.
- suggestion must focus only on IQA-related improvement actions.
- suggestion must be no more than 80 tokens total.
- Include multiple short suggestions when possible.
- Use degree modifiers in each suggestion, such as mildly, moderately, strongly, selectively, or carefully.
- Prefer concise action phrases.
- Do not include aesthetic, composition, mood, theme, or viewer-response suggestions.
- Do not include explanations.

Good suggestion examples:
- Moderately reduce blur; strongly suppress noise; carefully restore edges.
- Mildly denoise flat areas; moderately sharpen edges; reduce artifacts.
- Strongly reduce pixelation; moderately recover texture; improve fidelity.

========================
Output
========================

Return only the cleaned JSON profile.

The returned profile must contain:
- iaa
- iqa
- ista, if it existed in the original profile
- suggestion

Do not include explanations outside the JSON.

Original profile:
{{PROFILE_JSON}}"""


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


def render_json_repair_prompt(raw_output):
    """Render a prompt that asks the model to repair malformed JSON."""
    return JSON_REPAIR_PROMPT.replace("{{BROKEN_OUTPUT}}", str(raw_output or ""))
