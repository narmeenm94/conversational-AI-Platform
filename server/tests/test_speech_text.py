import json
import tempfile
import unittest
from pathlib import Path

from pipeline.llm_service import (
    build_system_prompt,
    load_character_profile,
    performance_turn_rule,
)
from pipeline.speech_text import (
    guard_verified_people,
    is_continuation_request,
    prepare_for_chatterbox,
    prepare_for_kokoro,
    prepare_for_pocket,
    prepare_for_orpheus,
    should_use_instant_backchannel,
)


class SpeechTextTests(unittest.TestCase):
    def test_kokoro_strips_stage_directions_but_keeps_expression(self):
        result = prepare_for_kokoro("<emotional> Oh! <chuckle> That worked.")
        self.assertEqual(result.text, "Oh! That worked.")
        self.assertEqual(result.emotion, "happy")
        self.assertTrue(result.speakable)

    def test_orpheus_keeps_only_supported_tags(self):
        result = prepare_for_orpheus("<pause> Wait. <gasp> Really?")
        self.assertNotIn("pause", result.text)
        self.assertIn("<gasp>", result.text)
        self.assertEqual(result.emotion, "surprised")

    def test_chatterbox_converts_documented_tags(self):
        result = prepare_for_chatterbox("<laugh> Yes. <sigh> Fine.")
        self.assertIn("[laugh]", result.text)
        self.assertIn("[sigh]", result.text)
        self.assertEqual(result.emotion, "happy")

    def test_pocket_strips_unsupported_cues_without_speaking_markup(self):
        result = prepare_for_pocket("[chuckle] Fine. <sigh> You win.")
        self.assertEqual(result.text, "Fine. You win.")
        self.assertNotIn("[", result.text)
        self.assertNotIn("<", result.text)
        self.assertEqual(result.emotion, "happy")

    def test_pocket_never_speaks_model_generated_action_labels(self):
        result = prepare_for_pocket(
            "Because he didn't get arrays! *laughs* Haha, it's a groaner."
        )
        self.assertEqual(
            result.text,
            "Because he didn't get arrays. Haha, it's a groaner.",
        )
        self.assertNotIn("laughs", result.text.lower())
        self.assertEqual(result.emotion, "happy")

    def test_pocket_does_not_treat_plain_exclamation_as_surprise(self):
        result = prepare_for_pocket("I must have gotten it wrong then!")
        self.assertEqual(result.text, "I must have gotten it wrong then.")
        self.assertEqual(result.emotion, "neutral")

    def test_pocket_keeps_explicit_surprise_cue(self):
        result = prepare_for_pocket("[gasp] Really!")
        self.assertEqual(result.text, "Really.")
        self.assertEqual(result.emotion, "surprised")

    def test_parenthetical_directions_are_backend_aware(self):
        source = (
            "(laughs sarcastically) Exactly! (Sarcastic tone) Sure. "
            "(Pausing for dramatic effect) The result (in theory) works."
        )
        pocket = prepare_for_pocket(source)
        self.assertEqual(pocket.text, "Exactly. Sure. The result (in theory) works.")
        self.assertEqual(pocket.emotion, "happy")
        self.assertNotIn("laugh", pocket.text.lower())
        self.assertNotIn("sarcastic", pocket.text.lower())
        self.assertNotIn("paus", pocket.text.lower())

        chatterbox = prepare_for_chatterbox(source)
        self.assertIn("[laugh] Exactly!", chatterbox.text)
        self.assertNotIn("Sarcastic tone", chatterbox.text)
        self.assertNotIn("Pausing", chatterbox.text)

        orpheus = prepare_for_orpheus("(Groan) Fine. (Waiting for a reaction)")
        self.assertEqual(orpheus.text, "<groan> Fine.")

    def test_voice_turn_rule_exposes_only_backend_supported_format(self):
        pocket = performance_turn_rule("pocket")
        self.assertIn("cannot execute performance tokens", pocket)
        self.assertIn("never write direction labels", pocket)
        self.assertNotIn("[laugh]", pocket)

        chatterbox = performance_turn_rule("chatterbox")
        self.assertIn("square-bracket cues", chatterbox)
        orpheus = performance_turn_rule("orpheus-cpp")
        self.assertIn("angle-bracket cues", orpheus)

    def test_people_guard_allows_verified_and_blocks_invented_names(self):
        settings = {
            "latest_user": "Who else is on your team?",
            "verified_people": ["Tiina Vuorio", "Santeri Saarinen"],
            "profile_name": "Narm",
        }
        verified, unknown = guard_verified_people(
            "Tiina Vuorio is our Head of Unit.", **settings
        )
        self.assertEqual(verified, "Tiina Vuorio is our Head of Unit.")
        self.assertEqual(unknown, [])

        guarded, unknown = guard_verified_people(
            "There is also Juhani Lehtinen from ops.", **settings
        )
        self.assertEqual(guarded, "")
        self.assertEqual(unknown, ["Juhani", "Lehtinen"])

        _, unknown = guard_verified_people(
            "Saara and Eero work here too.", **settings
        )
        self.assertEqual(unknown, ["Saara", "Eero"])

    def test_punctuation_only_is_not_speakable(self):
        self.assertFalse(prepare_for_kokoro("... !").speakable)

    def test_markdown_is_never_pronounced(self):
        result = prepare_for_kokoro("This is **very** *important*, *laughs softly*.")
        self.assertEqual(result.text, "This is very important.")
        self.assertNotIn("*", result.text)
        self.assertEqual(result.emotion, "happy")

    def test_common_written_abbreviations_are_expanded_for_speech(self):
        result = prepare_for_kokoro("Use a local model, e.g. Kokoro; i.e., no API.")
        self.assertEqual(
            result.text,
            "Use a local model, for example Kokoro; that is, no API.",
        )

    def test_orpheus_converts_asterisk_action_to_supported_cue(self):
        result = prepare_for_orpheus("*gasps* Really?")
        self.assertEqual(result.text, "<gasp> Really?")
        self.assertEqual(result.emotion, "surprised")

    def test_emoji_becomes_a_real_chatterbox_performance_cue(self):
        result = prepare_for_chatterbox("That was incredible! 😂")
        self.assertEqual(result.text, "That was incredible! [laugh]")
        self.assertNotIn("😂", result.text)

    def test_chatterbox_supports_extended_paralinguistic_cues(self):
        result = prepare_for_chatterbox(
            "<clear-throat> Well... <gasp> wow. <sniffle>"
        )
        self.assertEqual(result.text, "[clear throat] Well... [gasp] wow. [sniff]")

    def test_chatterbox_normalizes_descriptive_model_cues(self):
        result = prepare_for_chatterbox(
            "[laughs sarcastically] Fine. *rolls eyes and sighs* Let's try again."
        )
        self.assertEqual(result.text, "[laugh] Fine. [sigh] Let's try again.")

    def test_chatterbox_converts_eye_roll_without_speaking_stage_direction(self):
        result = prepare_for_chatterbox(
            "*rolls eyes and shakes head* Sure thing. Why did the robot leave?"
        )
        self.assertEqual(
            result.text,
            "[groan] Sure thing. Why did the robot leave?",
        )

    def test_chatterbox_adds_restrained_implicit_emotion(self):
        self.assertTrue(
            prepare_for_chatterbox("That's a genuinely funny joke.").text.startswith(
                "[chuckle]"
            )
        )
        self.assertTrue(
            prepare_for_chatterbox("I'm sorry you failed again.").text.startswith(
                "[sigh]"
            )
        )

    def test_real_unicode_emoji_becomes_a_cue(self):
        result = prepare_for_chatterbox("That worked! \U0001f602")
        self.assertEqual(result.text, "That worked! [laugh]")

    def test_character_profile_validation_and_rendering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "character.json"
            path.write_text(
                json.dumps({"name": "A", "traits": ["warm", "direct"]}),
                encoding="utf-8",
            )
            rendered = load_character_profile(str(path))
            self.assertEqual(rendered, "Traits: warm; direct")

    def test_continuation_requests_never_get_canned_backchannels(self):
        for text in ("Brain what?", "Use what?", "Go on", "Finish that sentence"):
            self.assertTrue(is_continuation_request(text), text)
            self.assertFalse(should_use_instant_backchannel(text), text)

    def test_backchannels_are_reserved_for_complex_turns(self):
        self.assertFalse(should_use_instant_backchannel("I am doing my best."))
        self.assertFalse(is_continuation_request("What is XR?"))
        self.assertFalse(should_use_instant_backchannel("What is XR?"))
        self.assertTrue(
            should_use_instant_backchannel(
                "What can you tell me about extended reality technology?"
            )
        )

    def test_team_query_requires_verified_names_before_generic_commentary(self):
        prompt = build_system_prompt(
            character_name="Narm",
            character_description="an XR colleague",
            user_text="Who are your colleagues?",
        )
        self.assertIn("VERIFIED TEAM ANSWER", prompt)
        self.assertIn("Start the first substantive sentence with one actual name", prompt)


if __name__ == "__main__":
    unittest.main()
