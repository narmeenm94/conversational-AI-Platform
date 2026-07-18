import unittest

from model_presets import catalog, normalize_runtime, resolve_runtime


class ModelPresetTests(unittest.TestCase):
    def test_default_is_current_streaming_stack(self):
        runtime = resolve_runtime({})
        self.assertEqual(runtime["brain"], "llama3.2-3b")
        self.assertEqual(runtime["llm_model"], "llama3.2:3b")
        self.assertEqual(runtime["tts_backend"], "pocket")

    def test_independent_brain_and_voice_overrides(self):
        runtime = resolve_runtime({
            "runtime": {
                "preset": "premium-streaming",
                "brain": "qwen3.5-4b",
                "voice": "kokoro",
            }
        })
        self.assertEqual(runtime["llm_model"], "qwen3.5:4b")
        self.assertEqual(runtime["tts_backend"], "kokoro")

    def test_unknown_engine_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_runtime({"voice": "mystery"})

    def test_catalog_explains_hardware_and_latency(self):
        data = catalog()
        for preset in data["presets"].values():
            self.assertTrue(preset["hardware"])
            self.assertTrue(preset["expected_turn"])


if __name__ == "__main__":
    unittest.main()
