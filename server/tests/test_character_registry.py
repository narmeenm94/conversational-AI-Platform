import tempfile
import unittest
from pathlib import Path

from control.character_registry import (
    CharacterRegistry,
    normalize_character,
)
from pipeline.llm_service import performance_cue_guide


class CharacterRegistryTests(unittest.TestCase):
    def test_language_is_validated_and_rendered_in_prompt(self):
        profile = normalize_character({"id": "aino", "name": "Aino", "language": "fi"})
        rendered = CharacterRegistry.render_prompt_profile(profile)
        self.assertEqual(profile["language"], "fi")
        self.assertIn("Language: Finnish (fi)", rendered)

        with self.assertRaises(ValueError):
            normalize_character({"id": "bad", "language": "xx"})

    def test_registry_persists_activation_and_isolates_profiles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = CharacterRegistry(
                root / "characters",
                state_path=root / "runtime" / "active.json",
                default_id="alex",
            )
            registry.create({"id": "aino", "name": "Aino", "language": "fi"})
            registry.activate("aino")

            reloaded = CharacterRegistry(
                root / "characters",
                state_path=root / "runtime" / "active.json",
                default_id="alex",
            )
            self.assertEqual(reloaded.active_id, "aino")
            self.assertEqual(reloaded.active["language"], "fi")
            self.assertNotEqual(
                reloaded.active["knowledge"]["collection_name"],
                reloaded.get("alex")["knowledge"]["collection_name"],
            )

    def test_multilingual_mode_forbids_turbo_stage_tags(self):
        guide = performance_cue_guide("chatterbox", "fi")
        self.assertIn("Do not emit performance tags", guide)
        self.assertNotIn("<laugh>", guide)

    def test_social_initiative_and_runtime_are_normalized(self):
        profile = normalize_character({
            "id": "friend",
            "conversation": {
                "relationship": "rival",
                "initiative": 2,
                "creativity": -4,
                "talkativeness": 4,
                "follow_up_frequency": -2,
                "min_sentences": 2,
                "max_sentences": 9,
                "greet_on_connect": True,
                "opening_lines": "First line.\nSecond line.",
            },
            "runtime": {
                "preset": "richer-character",
                "brain": "qwen3.5-4b",
                "voice": "chatterbox",
            },
            "knowledge": {"max_distance": 9},
        })
        self.assertEqual(profile["conversation"]["relationship"], "rival")
        self.assertEqual(profile["conversation"]["initiative"], 1.0)
        self.assertEqual(profile["conversation"]["creativity"], 0.0)
        self.assertEqual(profile["conversation"]["talkativeness"], 1.0)
        self.assertEqual(profile["conversation"]["follow_up_frequency"], 0.0)
        self.assertEqual(profile["conversation"]["min_sentences"], 2)
        self.assertEqual(profile["conversation"]["max_sentences"], 6)
        self.assertEqual(profile["conversation"]["opening_lines"], ["First line.", "Second line."])
        self.assertEqual(profile["runtime"]["brain"], "qwen3.5-4b")
        self.assertEqual(profile["knowledge"]["max_distance"], 2.0)
        rendered = CharacterRegistry.render_prompt_profile(profile)
        self.assertIn("Relationship: rival", rendered)
        self.assertIn("Initiative: 1.00", rendered)
        self.assertIn("Creativity: 0.00", rendered)
        self.assertIn("Talkativeness: 1.00", rendered)
        self.assertIn("Follow-up tendency: 0.00", rendered)
        self.assertIn("Do not stop after the direct answer", rendered)
        self.assertIn("do not merely praise it and interview them", rendered)
        self.assertIn("Never ask yourself a rhetorical question", rendered)
        self.assertIn("Turn length: 2 to 6", rendered)

    def test_animation_map_is_normalized_for_unity(self):
        profile = normalize_character({
            "id": "animated",
            "animations": {
                "thinking": "Think_Loop",
                "walking": "Walk_Female",
                "blend_seconds": 9,
            },
        })
        self.assertEqual(profile["animations"]["thinking"], "Think_Loop")
        self.assertEqual(profile["animations"]["walking"], "Walk_Female")
        self.assertEqual(profile["animations"]["blend_seconds"], 1.5)

    def test_talkative_shared_work_turn_gets_peer_brainstorm_guidance(self):
        profile = normalize_character({
            "id": "peer",
            "conversation": {"talkativeness": 0.9},
        })
        guidance = CharacterRegistry.render_turn_guidance(
            profile,
            "I'm working on a mixed reality project.",
        )
        self.assertIn("Output exactly four sentences", guidance)
        self.assertIn("Ask only one question total", guidance)
        self.assertEqual(
            CharacterRegistry.render_turn_guidance(profile, "How are you?"),
            "",
        )

    def test_verified_people_allowlist_is_normalized_and_rendered(self):
        profile = normalize_character({
            "id": "grounded",
            "knowledge": {
                "strict_people_grounding": True,
                "verified_people": "Tiina Vuorio\nSanteri Saarinen\n",
            },
        })
        self.assertTrue(profile["knowledge"]["strict_people_grounding"])
        self.assertEqual(
            profile["knowledge"]["verified_people"],
            ["Tiina Vuorio", "Santeri Saarinen"],
        )
        rendered = CharacterRegistry.render_prompt_profile(profile)
        self.assertIn("exclusive allowlist", rendered)
        self.assertIn("Tiina Vuorio; Santeri Saarinen", rendered)


if __name__ == "__main__":
    unittest.main()
