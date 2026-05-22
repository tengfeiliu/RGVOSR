import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloaders.degradation_meta import DEGRADATION_KEYS, REASONING_KEYS, read_jsonl_paths, to_jsonable  # noqa: E402


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
TAR_EXTENSIONS = {".tar"}
UNIPERCEPT_PROMPTS = {
    "iaa": "Analyze this image from the Image Aesthetic Assessment (IAA) perspective. Return your raw assessment.",
    "iqa": "Analyze this image from the Image Quality Assessment (IQA) perspective. Return your raw assessment.",
    "ista": "Analyze this image from the Image-Text Semantic Alignment (ISTA) perspective. Return your raw assessment.",
}


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


class UniPerceptRawAnalyzer:
    def __init__(self, device="cuda", model_path=None, unipercept_repo=None, command=None, backend="reward"):
        self.device = device
        self.model_path = model_path
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
        elif backend == "reward":
            self.inferencer = self._load_reward_inferencer()
        elif backend != "command":
            raise ValueError(f"Unsupported UniPercept backend: {backend}")

    def _load_reward_inferencer(self):
        try:
            from unipercept_reward import UniPerceptRewardInferencer
        except ImportError as exc:
            raise RuntimeError(
                "UniPercept raw scoring requires either `pip install unipercept-reward` "
                "or --unipercept-command for a custom full-repo inference command."
            ) from exc
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
            )
            result[domain] = parse_subprocess_output(completed.stdout)
        return result

    def _conversation_script(self):
        if self.unipercept_repo is None:
            raise ValueError("--unipercept-repo is required when --unipercept-backend=conversation")
        script = self.unipercept_repo / "src" / "eval" / "conversation.py"
        if not script.exists():
            raise FileNotFoundError(f"UniPercept conversation script not found: {script}")
        if not self.model_path:
            raise ValueError("--unipercept-model-path is required when --unipercept-backend=conversation")
        return script

    def _analyze_with_conversation(self, image_path):
        script = self._conversation_script()
        result = {}
        for domain, prompt in UNIPERCEPT_PROMPTS.items():
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
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
            result[domain] = parse_subprocess_output(completed.stdout)
        return result

    def analyze(self, image_path):
        image_path = Path(image_path)
        if self.backend == "command":
            return self._analyze_with_command(image_path)
        if self.backend == "conversation":
            return self._analyze_with_conversation(image_path)

        rewards = self.inferencer.reward(image_paths=[str(image_path)])
        reward = rewards[0] if rewards else {}
        reward = to_jsonable(reward or {})
        return {
            "iaa": reward.get("iaa"),
            "iqa": reward.get("iqa"),
            "ista": reward.get("ista"),
            "raw_reward": reward,
        }


def process_image(image_path, args, degradation, device, analyzer):
    hq = load_image_tensor(image_path, device)
    _, lq, meta = degradation.degrade_process(hq, resize_bak=args.resize_bak, return_meta=True)

    lq_path = Path(args.lq_output_dir) / stable_lq_name(image_path)
    save_lq_tensor(lq, lq_path)

    unipercept_raw = analyzer.analyze(lq_path)
    return {
        "hq_path": str(image_path),
        "lq_path": str(lq_path),
        "raw_degradation_params": to_jsonable(meta),
        "unipercept_raw": to_jsonable(unipercept_raw),
        "result": default_empty_result(),
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
    parser.add_argument("--unipercept-model-path", default=None, help="Optional local UniPercept model/checkpoint path.")
    parser.add_argument(
        "--unipercept-backend",
        choices=["reward", "conversation", "command"],
        default="reward",
        help="UniPercept inference backend. reward uses unipercept-reward; conversation calls the full repo script; command runs a custom template.",
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
