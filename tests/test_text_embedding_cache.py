import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None


def make_config(cache_dir, mode="cached", max_length=512):
    return {
        "model": {
            "flux_backend": "flux2_klein",
            "flux_model_path": "/models/FLUX.2-klein-base-4B",
            "dtype": "bf16",
            "text_encoder_dtype": "fp32",
            "max_prompt_sequence_length": max_length,
        },
        "condition": {
            "use_prompt": True,
            "use_suggestions": True,
        },
        "text_encoding": {
            "mode": mode,
            "cache_dir": str(cache_dir),
            "strict": True,
            "dtype": "bf16",
            "validate_prompt_hash": True,
        },
    }


@unittest.skipIf(torch is None, "torch is not installed")
class TextEmbeddingCacheTests(unittest.TestCase):
    def embedding_state(self):
        return {
            "prompt_embeds": torch.randn(1, 4, 8),
            "pooled_prompt_embeds": torch.randn(1, 3),
            "text_ids": torch.randn(1, 4, 4),
        }

    def test_valid_existing_embedding_is_loaded_without_online_encode(self):
        from models.text_embedding_cache import TextEmbeddingCache, resolve_prompt_embeddings

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = make_config(tmp_dir)
            cache = TextEmbeddingCache.from_config(config, dtype="bf16")
            cache.save_embedding("restore details", "/images/a.png", "/images/a.png", None, self.embedding_state())

            class FakeArtist:
                calls = 0

                def encode_prompts(self, prompts, device=None, dtype=None):
                    self.calls += 1
                    raise AssertionError("cached mode must not call the online text encoder")

            artist = FakeArtist()
            prompt_embeds, pooled_prompt_embeds, text_ids = resolve_prompt_embeddings(
                artist,
                ["restore details"],
                ["/images/a.png"],
                config,
                device="cpu",
                dtype=torch.float32,
                cache=cache,
            )
            self.assertEqual(artist.calls, 0)
            self.assertEqual(tuple(prompt_embeds.shape), (1, 4, 8))
            self.assertEqual(tuple(pooled_prompt_embeds.shape), (1, 3))
            self.assertEqual(tuple(text_ids.shape), (1, 4, 4))

    def test_prompt_or_encoder_change_invalidates_existing_cache(self):
        from models.text_embedding_cache import TextEmbeddingCache

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = make_config(tmp_dir)
            cache = TextEmbeddingCache.from_config(config, dtype="bf16")
            record = cache.save_embedding(
                "original prompt",
                "/images/a.png",
                "/images/a.png",
                None,
                self.embedding_state(),
            )
            self.assertIsNotNone(cache.find_record("original prompt", "/images/a.png"))
            self.assertIsNone(cache.find_record("changed prompt", "/images/a.png"))

            changed_config = make_config(tmp_dir, max_length=256)
            changed_cache = TextEmbeddingCache.from_config(changed_config, dtype="bf16")
            self.assertNotEqual(cache.encoder_signature, changed_cache.encoder_signature)
            self.assertIsNone(changed_cache.find_record("original prompt", "/images/a.png"))

            cache.embedding_path(record).unlink()
            self.assertIsNone(cache.find_record("original prompt", "/images/a.png"))

    def test_prompt_level_dedup_can_register_another_image(self):
        from models.text_embedding_cache import TextEmbeddingCache

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = make_config(tmp_dir)
            cache = TextEmbeddingCache.from_config(config, dtype="bf16")
            first = cache.save_embedding(
                "same prompt",
                "/images/a.png",
                "/images/a.png",
                None,
                self.embedding_state(),
            )
            reused = cache.find_record("same prompt", "/images/b.png", allow_prompt_reuse=True)
            self.assertEqual(reused["embedding_path"], first["embedding_path"])
            cache.register_existing_embedding(reused, "same prompt", "/images/b.png", "/images/b.png", None)
            self.assertIn("/images/b.png", cache.records_by_image)

    def test_auto_mode_falls_back_to_online_encoding(self):
        from models.text_embedding_cache import TextEmbeddingCache, resolve_prompt_embeddings

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = make_config(tmp_dir, mode="auto")
            cache = TextEmbeddingCache.from_config(config, dtype="bf16")

            class FakeArtist:
                calls = 0

                def encode_prompts(self, prompts, device=None, dtype=None):
                    self.calls += 1
                    return (
                        torch.ones(1, 2, 3, device=device, dtype=dtype),
                        torch.ones(1, 4, device=device, dtype=dtype),
                        torch.ones(1, 2, 4, device=device, dtype=dtype),
                    )

            artist = FakeArtist()
            result = resolve_prompt_embeddings(
                artist,
                ["missing prompt"],
                ["/images/missing.png"],
                config,
                device="cpu",
                dtype=torch.float32,
                cache=cache,
            )
            self.assertEqual(artist.calls, 1)
            self.assertEqual(tuple(result[0].shape), (1, 2, 3))


class TextEmbeddingCacheStaticTests(unittest.TestCase):
    def test_cache_cli_supports_resume_skip_existing_and_overwrite(self):
        source = Path("tools/cache_rg_flux_text_embeddings.py").read_text(encoding="utf-8")

        self.assertIn("--resume", source)
        self.assertIn("--skip-existing", source)
        self.assertIn("--overwrite", source)
        self.assertIn("cache.find_record", source)
        self.assertIn("register_existing_embedding", source)

    def test_cached_artists_skip_text_pipeline_and_guard_online_encode(self):
        flux1 = Path("models/flux_sr_artist.py").read_text(encoding="utf-8")
        flux2 = Path("models/flux2_klein_sr_artist.py").read_text(encoding="utf-8")

        for source in (flux1, flux2):
            self.assertIn("self.load_text_encoder = text_encoder_should_load(config)", source)
            self.assertIn("if self.load_text_encoder:", source)
            self.assertIn("Text encoder is disabled because text_encoding.mode='cached'", source)

    def test_train_evaluation_and_inference_use_shared_embedding_resolver(self):
        train_source = Path("train_rg_flux_sr.py").read_text(encoding="utf-8")
        inference_source = Path("inference_rg_flux_sr.py").read_text(encoding="utf-8")

        self.assertGreaterEqual(train_source.count("resolve_prompt_embeddings("), 2)
        self.assertIn("text_embedding_cache=text_embedding_cache", train_source)
        self.assertIn("resolve_prompt_embeddings(", inference_source)
        self.assertIn("--text_encoding_mode", inference_source)
        self.assertIn("--text_embedding_cache", inference_source)


if __name__ == "__main__":
    unittest.main()
