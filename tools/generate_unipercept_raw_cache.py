import argparse
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloaders.degradation_meta import (  # noqa: E402
    DEGRADATION_KEYS,
    REASONING_KEYS,
    build_degradation_vector_from_meta,
    merge_analysis_result,
    physical_suggestions_from_vector,
    read_jsonl_paths,
    to_jsonable,
)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
TAR_EXTENSIONS = {".tar"}
HF_LOCAL_ONLY_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
UNIPERCEPT_PROMPTS = {
    "iaa": "Analyze this image from the Image Aesthetic Assessment (IAA) perspective. Return your raw assessment.",
    "iqa": "Analyze this image from the Image Quality Assessment (IQA) perspective. Return your raw assessment.",
    "ista": "Analyze this image from the Image-Text Semantic Alignment (ISTA) perspective. Return your raw assessment.",
}
IAA_PROFILE_PROMPTS = {
    "composition_design": (
        "Analyze this image from the Image Aesthetics Assessment (IAA) perspective. "
        "Focus only on Composition & Design: balance, framing, symmetry, rule of thirds, "
        "leading lines, visual rhythm, layout coherence, and spatial organization. "
        "Return concise bullet points only."
    ),
    "visual_elements_structure": (
        "Analyze this image from the Image Aesthetics Assessment (IAA) perspective. "
        "Focus only on Visual Elements & Structure: color harmony, lighting, exposure, "
        "atmosphere, subject clarity, depth, and element relationships. "
        "Return concise bullet points only."
    ),
    "technical_execution": (
        "Analyze this image from the Image Aesthetics Assessment (IAA) perspective. "
        "Focus only on Technical Execution: focus, sharpness, exposure control, noise, "
        "artifacts, detail rendering, and craft quality as they affect aesthetics. "
        "Return concise bullet points only."
    ),
    "originality_creativity": (
        "Analyze this image from the Image Aesthetics Assessment (IAA) perspective. "
        "Focus only on Originality & Creativity: novelty, intent, visual concept, "
        "genre coherence, and expressive choices. Return concise bullet points only."
    ),
    "theme_communication": (
        "Analyze this image from the Image Aesthetics Assessment (IAA) perspective. "
        "Focus only on Theme & Communication: storytelling, subject message, mood, "
        "and whether the image communicates a coherent visual idea. "
        "Return concise bullet points only."
    ),
    "emotion_viewer_response": (
        "Analyze this image from the Image Aesthetics Assessment (IAA) perspective. "
        "Focus only on Emotion & Viewer Response: emotional impact, attraction, "
        "memorability, immersion, and viewer engagement. Return concise bullet points only."
    ),
    "overall_gestalt": (
        "Analyze this image from the Image Aesthetics Assessment (IAA) perspective. "
        "Focus only on Overall Gestalt: whether the image works as a unified whole, "
        "including harmony between composition, subject, color, and technical execution. "
        "Return concise bullet points only."
    ),
    "comprehensive": (
        "Analyze this image from the Image Aesthetics Assessment (IAA) perspective. "
        "Provide a comprehensive aesthetic evaluation that summarizes composition, "
        "visual elements, technical execution, creativity, communication, emotional impact, "
        "and overall aesthetic merit. Return concise bullet points only."
    ),
}
IQA_PROFILE_PROMPTS = {
    "distortion_location": (
        "Analyze this image from the Image Quality Assessment (IQA) perspective. "
        "Focus only on Distortion Location: identify where blur, noise, compression, "
        "ringing, aliasing, exposure issues, color problems, or other quality degradations "
        "are visible. Return concise bullet points only."
    ),
    "distortion_severity": (
        "Analyze this image from the Image Quality Assessment (IQA) perspective. "
        "Focus only on Distortion Severity: explain how severe the visible degradations are "
        "and how much they reduce clarity, fidelity, and usability. "
        "Return concise bullet points only."
    ),
    "distortion_type": (
        "Analyze this image from the Image Quality Assessment (IQA) perspective. "
        "Focus only on Distortion Type: name the dominant visible degradation categories "
        "such as blur, noise, JPEG artifacts, ringing, aliasing, texture loss, exposure error, "
        "or color shift. Return concise bullet points only."
    ),
    "overall_quality": (
        "Analyze this image from the Image Quality Assessment (IQA) perspective. "
        "Provide an overall quality evaluation that summarizes visible distortion location, "
        "distortion type, distortion severity, clarity, fidelity, and perceived technical quality. "
        "Return concise bullet points only."
    ),
}
ISTA_STRUCTURAL_ANNOTATION_PROMPT = """Prompt for ISTA Structural Annotation

[PRIOR KNOWLEDGE BASE]
- Base Morphology
blotchy, braided, bubbly, bumpy, chequered, cobwebbed, cracked, crosshatched, crystalline, dotted, fibrous, flecked, freckled, frilly, grid, grooved, honeycombed, interlaced, knitted, lacelike, lined, marbled, matted, meshed, paisley, perforated, pitted, pleated, porous, scaly, smeared, spiralled, sprinkled, stratified, striped, studded, swirly, veined, woven, wrinkled, zigzagged, smooth
- Material Type
1. Natural Materials: Foliage, Grass, Skin, Stone, Wood, Water, Hair
2. Man-Made Materials: Brick, Carpet, Ceramic, Fabric, Glass, Leather, Metal, Mirror, Painted Surface, Paper, Plastic, Polished Stone, Tile, Wallpaper, Concrete, Food Surface
3. Environmental / Background Textures: Sky, Clouds, Fog / Mist
- Two-Dimensional Shape
Rectangle, Square, Circle, Ellipse / Oval, Triangle, Equilateral Triangle, Isosceles Triangle, Scalene Triangle, Right Triangle, Trapezoid / Trapezium, Parallelogram, Rhombus, Pentagon, Hexagon, Heptagon, Octagon, Nonagon, Decagon, Star, Pentagram, Hexagram, Cross, Arrow, Semicircle, Sector, Crescent, Annulus / Ring, Heart, Lemniscate, Lune / Bow Shape, Spiral, Waveform, Teardrop
- Three-Dimensional Shape Categories
Sphere, Ellipsoid, Cube, Cuboid, Cylinder, Cone, Pyramid, Tetrahedron, Octahedron, Dodecahedron, Icosahedron, Prism, Triangular Prism, Rectangular Prism, Pentagonal Prism, Hexagonal Prism, Torus, Annular Torus, Paraboloid, Hyperboloid, Elliptic Cylinder, Hyperbolic Cylinder, Truncated Cone, Truncated Pyramid, Capsule, Dome, Lens, Bipyramid, Frustum, Mobius Strip, Knot, Klein Bottle
- Style Semantics
Embossed, Engraved, Rough, Smooth, Matte, Glossy, Brushed, Honeycomb, Geometric, Fractal, Tile Mosaic, Chinese Cloud Pattern, Dragon Scale, Cyberpunk Holographic, Steampunk Mechanical

[STRUCTURE TEMPLATE]
- Scene Decomposition Principles
A. Single Scene: Please introduce the description object.
B. Composite Scene: Please introduce the different objects and describe each object separately.
- Description Content
1. Physical Structure
Base Morphology (*) -> Select 1-3 terms from the Base Morphology lexicon to describe surface texture.
Arrangement (!) -> Describe spatial layout or directionality of texture: orientation, distribution pattern, or density changes.
Dynamics (!) -> Motion/transition states when applicable.
2. Material Representation
Material Class (*) -> Select from Material Type.
Surface Properties (!) -> Reflectivity/translucency when applicable.
3. Geometric Composition
Planar Contour (!) -> 2D shape terms where applicable.
Volumetric Form (!) -> 3D form terms where applicable.
4. Semantic Perception
Functional Inference (!) -> Only when text/icons are present.
Style Type (!) -> Use style semantics terms where applicable.

[Execution Standards]
1. Terminology Enforcement: (*)-marked fields require exact term matches.
2. Format Purity: Output only structured JSON content, no markdown, explanations, or code fences.
3. Hierarchy Preservation: Apply the complete template per independent unit.
4. Complexity Adaptation: Use a single description for simple objects and multi-unit decomposition for complex scenes.
5. Lexicon Flexibility: For *-marked fields, use official lexicon terms where possible. Free-form extensions are allowed only when necessary.
6. Mixed Mode Expression: Structured descriptions may combine fixed taxonomy terms with precise natural language for edge cases.

[JSON TEMPLATE]
{
  "SceneType": "<Single Scene|Composite Scene>",
  "SceneName": "<SceneName>",
  "Components": [
    {
      "ComponentName": "<ComponentName>",
      "DescriptionContent": {
        "PhysicalStructure": {
          "BaseMorphology": ["<BaseMorphology>"],
          "Arrangement": ["<Arrangement>"],
          "Dynamics": ["N/A"]
        },
        "MaterialRepresentation": {
          "MaterialClass": ["<MaterialClass>"],
          "SurfaceProperties": ["<SurfaceProperties>"]
        },
        "GeometricComposition": {
          "PlanarContour": ["<PlanarContour>"],
          "VolumetricForm": ["<VolumetricForm>"]
        },
        "SemanticPerception": {
          "FunctionalInference": ["N/A"],
          "StyleType": ["N/A"]
        }
      }
    }
  ]
}

Please use the above foundational knowledge and template to perform a texture and structural analysis of the image. Return valid JSON only."""


def force_hf_local_only_env(env=None):
    target = os.environ if env is None else dict(env)
    for key, value in HF_LOCAL_ONLY_ENV.items():
        target[key] = value
    return target


def require_local_unipercept_model_path(model_path, backend):
    if not model_path:
        raise ValueError(
            f"--unipercept-model-path is required when --unipercept-backend={backend} "
            "so UniPercept loads local weights instead of downloading from Hugging Face."
        )
    path = Path(model_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Local UniPercept model path does not exist: {path}")
    return str(path.resolve())


def patch_unipercept_config_for_transformers_logging():
    module_name = "unipercept_reward.internvl.model.internvl_chat.configuration_internvl_chat"
    try:
        config_module = importlib.import_module(module_name)
    except ImportError:
        return False
    config_class = getattr(config_module, "InternVLChatConfig", None)
    if config_class is None:
        return False

    # Transformers may instantiate this custom config with no args while formatting logs.
    # UniPercept's empty default has an invalid blank architecture, so skip default diffing.
    setattr(config_class, "has_no_defaults_at_init", True)
    return True


def append_jsonl(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(payload), ensure_ascii=False) + "\n")


def first_column_path(line):
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    return line.split(",", 1)[0].strip()


def _expand_source(path, visited):
    path = Path(path).expanduser()
    if path in visited:
        return []
    visited.add(path)

    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return [path]
    if suffix in TAR_EXTENSIONS:
        raise NotImplementedError(
            f"Tar input is not supported by generate_unipercept_raw_cache.py yet: {path}. "
            "Use txt image lists or directories for v1 raw UniPercept cache generation."
        )
    if path.is_dir():
        return sorted(
            item
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        )
    if path.is_file():
        images = []
        base_dir = path.parent
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw_entry = first_column_path(line)
                if not raw_entry:
                    continue
                entry = Path(raw_entry).expanduser()
                if not entry.is_absolute():
                    entry = base_dir / entry
                images.extend(_expand_source(entry, visited))
        return images
    raise FileNotFoundError(f"Input path does not exist: {path}")


def list_hq_images(input_path):
    return _expand_source(input_path, visited=set())


def stable_lq_name(image_path):
    image_path = Path(image_path)
    digest = hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()[:10]
    return f"{digest}_{image_path.stem}.png"


def load_image_tensor(image_path, device):
    from torchvision import transforms

    with Image.open(image_path) as image:
        image = image.convert("RGB")
    return transforms.ToTensor()(image).unsqueeze(0).to(device)


def save_lq_tensor(tensor, output_path):
    from torchvision import transforms

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tensor = tensor.detach().clamp(0, 1).squeeze(0).cpu()
    transforms.ToPILImage()(tensor).save(output_path)


def default_empty_result():
    return {
        "reasoning": {key: "" for key in REASONING_KEYS},
        "suggestions": [],
        "score": 0,
        "degradation_vector": {key: 0.0 for key in DEGRADATION_KEYS},
    }


def parse_subprocess_output(raw_text):
    raw_text = str(raw_text or "").strip()
    if not raw_text:
        return ""
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return raw_text


def extract_conversation_answer(raw_text):
    text = str(raw_text or "").strip()
    if not text:
        return ""

    lines = [line.rstrip() for line in text.splitlines()]
    assistant_patterns = (
        re.compile(r"^\s*(?:assistant|ASSISTANT|Assistant)\s*:\s*(.*)$"),
        re.compile(r"^\s*#+\s*(?:assistant|ASSISTANT|Assistant)\s*:?\s*(.*)$"),
        re.compile(r"^\s*(?:response|RESPONSE|Response)\s*:\s*(.*)$"),
    )
    for index in range(len(lines) - 1, -1, -1):
        line = lines[index]
        for pattern in assistant_patterns:
            match = pattern.match(line)
            if match:
                first = match.group(1).strip()
                tail = [item.strip() for item in lines[index + 1 :] if item.strip()]
                parts = [first] if first else []
                parts.extend(tail)
                return "\n".join(parts).strip()
    return text


def _parse_json_object(raw_text):
    text = str(raw_text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])

    last_error = None
    for candidate in candidates:
        repaired = re.sub(r",\s*([}\]])", r"\1", candidate.strip())
        try:
            parsed = json.loads(repaired)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError as exc:
            last_error = exc
    raise ValueError(f"invalid JSON response: {last_error}")


def normalize_ista_profile_response(raw_text):
    raw_text = extract_conversation_answer(raw_text)
    try:
        structural_annotation = _parse_json_object(raw_text)
    except ValueError:
        structural_annotation = {}
    return {
        "structural_annotation": structural_annotation,
        "raw_structural_annotation": raw_text,
    }


def _list_text(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        values = value
    else:
        values = [value]
    result = []
    for item in values:
        text = str(item or "").strip()
        if text:
            result.append(text)
    return result


def _join_profile_parts(*parts):
    clean = []
    for part in parts:
        if isinstance(part, (list, tuple)):
            nested = _join_profile_parts(*part)
            if nested and nested not in clean:
                clean.append(nested)
            continue
        text = str(part or "").strip()
        if text and text not in clean:
            clean.append(text)
    return " ".join(clean).strip()


def _safe_profile_text(section, key):
    if not isinstance(section, dict):
        return ""
    value = section.get(key, "")
    if isinstance(value, (dict, list)):
        return json.dumps(to_jsonable(value), ensure_ascii=False)
    return str(value or "").strip()


def _summarize_ista_annotation(annotation):
    if not isinstance(annotation, dict) or not annotation:
        return ""

    scene_type = str(annotation.get("SceneType") or "").strip()
    scene_name = str(annotation.get("SceneName") or "").strip()
    summary = []
    if scene_name or scene_type:
        label = " ".join(part for part in (scene_type, scene_name) if part)
        summary.append(f"Scene: {label}.")

    components = annotation.get("Components")
    if not isinstance(components, list):
        components = []
    component_summaries = []
    for component in components[:5]:
        if not isinstance(component, dict):
            continue
        name = str(component.get("ComponentName") or "").strip()
        content = component.get("DescriptionContent")
        content = content if isinstance(content, dict) else {}
        physical = content.get("PhysicalStructure") if isinstance(content.get("PhysicalStructure"), dict) else {}
        material = (
            content.get("MaterialRepresentation")
            if isinstance(content.get("MaterialRepresentation"), dict)
            else {}
        )
        geometric = (
            content.get("GeometricComposition")
            if isinstance(content.get("GeometricComposition"), dict)
            else {}
        )

        terms = []
        terms.extend(_list_text(physical.get("BaseMorphology")))
        terms.extend(_list_text(physical.get("Arrangement")))
        terms.extend(_list_text(material.get("MaterialClass")))
        terms.extend(_list_text(material.get("SurfaceProperties")))
        terms.extend(_list_text(geometric.get("PlanarContour")))
        terms.extend(_list_text(geometric.get("VolumetricForm")))
        terms = [term for term in terms if term.lower() not in {"n/a", "na", "none"}]
        if name and terms:
            component_summaries.append(f"{name}: {', '.join(terms[:8])}.")
        elif name:
            component_summaries.append(f"{name}.")
    summary.extend(component_summaries)
    return " ".join(summary).strip()


def _iter_values(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_values(item)
    else:
        yield value


def _text_region_risk_from_annotation(annotation):
    if not isinstance(annotation, dict):
        return 0.0
    components = annotation.get("Components")
    if not isinstance(components, list):
        return 0.0

    text_terms = ("text", "icon", "letter", "word", "character", "logo", "sign", "street", "traffic")
    for component in components:
        if not isinstance(component, dict):
            continue
        content = component.get("DescriptionContent")
        content = content if isinstance(content, dict) else {}
        semantic = content.get("SemanticPerception")
        semantic = semantic if isinstance(semantic, dict) else {}
        functional = semantic.get("FunctionalInference")
        for value in _iter_values(functional):
            text = str(value or "").strip().lower()
            if not text or text in {"n/a", "na", "none"}:
                continue
            if any(term in text for term in text_terms):
                return 0.7
    return 0.0


def _sr_strategy_from_suggestions(suggestions):
    suggestions = list(suggestions or [])
    if not suggestions:
        return "Use the degradation vector and UniPercept profile to preserve global structure while restoring visible detail."
    return "Prioritize " + ", ".join(suggestions) + " while preserving the original scene layout and avoiding semantic changes."


def _physical_vector_from_meta(meta):
    meta = meta if isinstance(meta, dict) else {}
    vector = meta.get("degradation_vector")
    if isinstance(vector, dict):
        return vector
    try:
        return build_degradation_vector_from_meta(meta)
    except Exception:
        return {}


def build_result_from_unipercept_profile(meta, unipercept_raw):
    if not isinstance(unipercept_raw, dict):
        return default_empty_result()
    profile = unipercept_raw.get("profile")
    if not isinstance(profile, dict):
        return default_empty_result()

    iaa = profile.get("iaa") if isinstance(profile.get("iaa"), dict) else {}
    iqa = profile.get("iqa") if isinstance(profile.get("iqa"), dict) else {}
    ista = profile.get("ista") if isinstance(profile.get("ista"), dict) else {}
    annotation = ista.get("structural_annotation") if isinstance(ista, dict) else {}
    ista_summary = _summarize_ista_annotation(annotation)
    if not ista_summary:
        ista_summary = str(ista.get("raw_structural_annotation") or "").strip()

    degradation_analysis = _join_profile_parts(
        _safe_profile_text(iqa, "overall_quality"),
        _safe_profile_text(iqa, "distortion_location"),
        _safe_profile_text(iqa, "distortion_severity"),
    )
    texture_edge_analysis = _join_profile_parts(
        ista_summary,
        _safe_profile_text(iqa, "distortion_type"),
    )
    semantic_risk_analysis = _join_profile_parts(
        _safe_profile_text(iaa, "comprehensive"),
        ista_summary,
    )

    physical_vector = _physical_vector_from_meta(meta)
    suggestions = physical_suggestions_from_vector(physical_vector)
    text_region_risk = _text_region_risk_from_annotation(annotation)
    if text_region_risk > 0 and "improve text readability" not in suggestions:
        suggestions.append("improve text readability")
    suggestions = suggestions[:5]

    semantic_result = {
        "reasoning": {
            "degradation_analysis": degradation_analysis or "UniPercept IQA profile did not provide quality details.",
            "texture_edge_analysis": texture_edge_analysis or "UniPercept ISTA profile did not provide structure details.",
            "semantic_risk_analysis": semantic_risk_analysis or "UniPercept IAA/ISTA profile did not provide semantic risk details.",
            "sr_strategy": _sr_strategy_from_suggestions(suggestions),
        },
        "suggestions": suggestions,
        "score": 0,
        "degradation_vector": {
            "text_region_risk": text_region_risk,
            "hallucination_risk": 0.0,
        },
    }
    return merge_analysis_result(physical_vector, semantic_result)


class UniPerceptRawAnalyzer:
    def __init__(self, device="cuda", model_path=None, unipercept_repo=None, command=None, backend="reward"):
        self.device = device
        if backend in {"reward", "conversation", "profile"}:
            self.model_path = require_local_unipercept_model_path(model_path, backend)
        elif model_path:
            self.model_path = str(Path(model_path).expanduser())
        else:
            self.model_path = None
        self.unipercept_repo = Path(unipercept_repo).expanduser() if unipercept_repo else None
        self.backend = backend
        self.command = command
        self.inferencer = None
        if unipercept_repo:
            repo_path = str(self.unipercept_repo)
            if repo_path not in sys.path:
                sys.path.insert(0, repo_path)
        if backend == "command" and not command:
            raise ValueError("--unipercept-command is required when --unipercept-backend=command")
        if backend == "conversation":
            self._conversation_script()
        elif backend == "profile":
            self._conversation_script()
            self.inferencer = self._load_reward_inferencer()
        elif backend == "reward":
            self.inferencer = self._load_reward_inferencer()
        elif backend != "command":
            raise ValueError(f"Unsupported UniPercept backend: {backend}")

    def _load_reward_inferencer(self):
        force_hf_local_only_env()
        try:
            from unipercept_reward import UniPerceptRewardInferencer
        except ImportError as exc:
            raise RuntimeError(
                "UniPercept raw scoring requires either `pip install unipercept-reward` "
                "or --unipercept-command for a custom full-repo inference command."
            ) from exc
        patch_unipercept_config_for_transformers_logging()
        kwargs = {"device": self.device}
        if self.model_path:
            kwargs["model_path"] = self.model_path
        return UniPerceptRewardInferencer(**kwargs)

    def _analyze_with_command(self, image_path):
        result = {}
        for domain in ("iaa", "iqa", "ista"):
            command = self.command.format(
                image=str(image_path),
                domain=domain,
                model_path=self.model_path or "",
                device=self.device,
            )
            completed = subprocess.run(
                command,
                shell=True,
                check=True,
                capture_output=True,
                text=True,
                env=force_hf_local_only_env(os.environ),
            )
            result[domain] = parse_subprocess_output(completed.stdout)
        return result

    def _conversation_script(self):
        if self.unipercept_repo is None:
            raise ValueError("--unipercept-repo is required when --unipercept-backend=conversation/profile")
        script = self.unipercept_repo / "src" / "eval" / "conversation.py"
        if not script.exists():
            raise FileNotFoundError(f"UniPercept conversation script not found: {script}")
        if not self.model_path:
            raise ValueError("--unipercept-model-path is required when --unipercept-backend=conversation/profile")
        return script

    def _run_conversation_prompt(self, image_path, prompt):
        script = self._conversation_script()
        command = [
            sys.executable,
            str(script),
            "--model_path",
            str(self.model_path),
            "--image",
            str(image_path),
            "--prompt",
            prompt,
        ]
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=force_hf_local_only_env(os.environ),
        )
        return extract_conversation_answer(completed.stdout)

    def _analyze_with_conversation(self, image_path):
        result = {}
        for domain, prompt in UNIPERCEPT_PROMPTS.items():
            result[domain] = parse_subprocess_output(self._run_conversation_prompt(image_path, prompt))
        return result

    def _reward_scores(self, image_path):
        rewards = self.inferencer.reward(image_paths=[str(image_path)])
        reward = rewards[0] if rewards else {}
        reward = to_jsonable(reward or {})
        return {
            "iaa": reward.get("iaa"),
            "iqa": reward.get("iqa"),
            "ista": reward.get("ista"),
            "raw_reward": reward,
        }

    def _analyze_with_profile(self, image_path):
        result = self._reward_scores(image_path)
        profile = {
            "iaa": {},
            "iqa": {},
            "ista": {},
        }
        for key, prompt in IAA_PROFILE_PROMPTS.items():
            profile["iaa"][key] = self._run_conversation_prompt(image_path, prompt)
        for key, prompt in IQA_PROFILE_PROMPTS.items():
            profile["iqa"][key] = self._run_conversation_prompt(image_path, prompt)
        ista_raw = self._run_conversation_prompt(image_path, ISTA_STRUCTURAL_ANNOTATION_PROMPT)
        profile["ista"] = normalize_ista_profile_response(ista_raw)
        result["profile"] = profile
        return result

    def analyze(self, image_path):
        image_path = Path(image_path)
        if self.backend == "command":
            return self._analyze_with_command(image_path)
        if self.backend == "conversation":
            return self._analyze_with_conversation(image_path)
        if self.backend == "profile":
            return self._analyze_with_profile(image_path)

        return self._reward_scores(image_path)


def process_image(image_path, args, degradation, device, analyzer):
    hq = load_image_tensor(image_path, device)
    _, lq, meta = degradation.degrade_process(hq, resize_bak=args.resize_bak, return_meta=True)

    lq_path = Path(args.lq_output_dir) / stable_lq_name(image_path)
    save_lq_tensor(lq, lq_path)

    unipercept_raw = analyzer.analyze(lq_path)
    result = build_result_from_unipercept_profile(meta, unipercept_raw)
    return {
        "hq_path": str(image_path),
        "lq_path": str(lq_path),
        "raw_degradation_params": to_jsonable(meta),
        "unipercept_raw": to_jsonable(unipercept_raw),
        "result": result,
    }


def load_seen_hq_paths(output, invalid_output):
    seen = set()
    seen.update(read_jsonl_paths(output, key="hq_path"))
    seen.update(read_jsonl_paths(invalid_output, key="hq_path"))
    return seen


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic LQ images and raw UniPercept IAA/IQA/ISTA JSONL cache."
    )
    parser.add_argument("--input", default="configs/train_txt/train_dataset_txt.txt", help="HQ image, directory, txt list, or train dataset config.")
    parser.add_argument("--lq-output-dir", default="datasets/LSDIR_unipercept_lq", help="Directory for generated LQ PNG images.")
    parser.add_argument("--output", default="datasets/LSDIR_unipercept_raw_cache/valid.jsonl", help="Valid raw UniPercept cache JSONL path.")
    parser.add_argument("--invalid-output", default="datasets/LSDIR_unipercept_raw_cache/invalid.jsonl", help="Invalid/error JSONL path.")
    parser.add_argument("--opt-name", default="params_realsr.yml", help="RealESRGAN degradation YAML under dataloaders/.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resize-bak", action="store_true", default=True)
    parser.add_argument("--no-resize-bak", dest="resize_bak", action="store_false")
    parser.add_argument("--unipercept-repo", default=None, help="Optional local UniPercept repo path to add to PYTHONPATH.")
    parser.add_argument(
        "--unipercept-model-path",
        default='/data/models/UniPercept/',
        help=(
            "Local UniPercept model/checkpoint path. Required for reward, conversation, and profile "
            "backends; Hugging Face downloads are disabled."
        ),
    )
    parser.add_argument(
        "--unipercept-backend",
        choices=["reward", "conversation", "command", "profile"],
        default="reward",
        help=(
            "UniPercept inference backend. reward uses unipercept-reward; conversation calls the full repo script; "
            "profile combines reward scores with per-aspect conversation prompts; command runs a custom template."
        ),
    )
    parser.add_argument(
        "--unipercept-command",
        default=None,
        help=(
            "Optional custom command template for full UniPercept repo inference. "
            "Available placeholders: {image}, {domain}, {model_path}, {device}. "
            "The command must print JSON or raw text to stdout."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    images = list_hq_images(args.input)
    if args.limit > 0:
        images = images[: args.limit]

    seen = load_seen_hq_paths(args.output, args.invalid_output) if args.resume else set()

    import torch

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    from dataloaders.realesrgan_gpu import RealESRGAN_degradation

    degradation = RealESRGAN_degradation(args.opt_name, device=device)
    analyzer = UniPerceptRawAnalyzer(
        device=args.device,
        model_path=args.unipercept_model_path,
        unipercept_repo=args.unipercept_repo,
        command=args.unipercept_command,
        backend=args.unipercept_backend,
    )

    for image_path in images:
        image_key = str(image_path)
        if image_key in seen:
            continue
        try:
            record = process_image(image_path, args, degradation, device, analyzer)
            append_jsonl(args.output, record)
        except Exception as exc:
            append_jsonl(
                args.invalid_output,
                {
                    "hq_path": image_key,
                    "reason": str(exc),
                },
            )


if __name__ == "__main__":
    main()
