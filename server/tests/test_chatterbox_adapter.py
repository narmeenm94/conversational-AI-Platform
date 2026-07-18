import unittest

import numpy as np

from pipeline.chatterbox_tts_service import LowLatencySpeechAggregator, smooth_audio_boundaries


class LowLatencySpeechAggregatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_releases_a_short_sentence_without_lookahead(self):
        aggregator = LowLatencySpeechAggregator()
        chunks = [item.text async for item in aggregator.aggregate("Hello, I can hear you.")]
        self.assertEqual(chunks, ["Hello, I can hear you."])

    async def test_bounds_a_long_first_phrase_at_a_safe_word(self):
        aggregator = LowLatencySpeechAggregator()
        chunks = [
            item.text
            async for item in aggregator.aggregate(
                "This answer keeps streaming, without cutting at a comma or word limit. "
            )
        ]
        tail = await aggregator.flush()
        if tail:
            chunks.append(tail.text)
        self.assertEqual(
            chunks,
            [
                "This answer keeps streaming, without cutting at a comma",
                "or word limit.",
            ],
        )

    async def test_cached_hesitation_does_not_disable_first_answer_bound(self):
        aggregator = LowLatencySpeechAggregator()
        hesitation = [
            item.text async for item in aggregator.aggregate("Hmm... uh... right...")
        ]
        answer = [
            item.text
            async for item in aggregator.aggregate(
                "Extended reality is like the ultimate modern technology upgrade for everyone."
            )
        ]
        self.assertEqual(hesitation, ["Hmm... uh... right..."])
        self.assertEqual(
            answer,
            ["Extended reality is like the ultimate modern technology"],
        )

    async def test_short_internal_comma_does_not_fragment_continuation(self):
        aggregator = LowLatencySpeechAggregator()
        chunks = [
            item.text
            async for item in aggregator.aggregate(
                "Fair point. Like, this continuation still sounds natural."
            )
        ]
        self.assertEqual(
            chunks,
            ["Fair point.", "Like, this continuation still sounds natural."],
        )

    async def test_does_not_split_inside_asterisk_stage_direction(self):
        aggregator = LowLatencySpeechAggregator()
        chunks = [
            item.text
            async for item in aggregator.aggregate(
                "*pauses thoughtfully. then grins* I know the answer."
            )
        ]
        self.assertEqual(
            chunks,
            ["*pauses thoughtfully. then grins* I know the answer."],
        )

    async def test_interruption_discards_partial_clause(self):
        aggregator = LowLatencySpeechAggregator()
        self.assertEqual(
            [item.text async for item in aggregator.aggregate("An obsolete partial")],
            [],
        )
        await aggregator.handle_interruption()
        self.assertIsNone(await aggregator.flush())

    async def test_drops_orphan_word_at_generation_limit(self):
        aggregator = LowLatencySpeechAggregator()
        chunks = [item.text async for item in aggregator.aggregate("Emmi")]
        self.assertEqual(chunks, [])
        self.assertIsNone(await aggregator.flush())

    async def test_drops_any_unfinished_generation_tail(self):
        aggregator = LowLatencySpeechAggregator()
        chunks = [
            item.text
            async for item in aggregator.aggregate(
                "She's been looking into how we can use"
            )
        ]
        self.assertEqual(chunks, [])
        self.assertIsNone(await aggregator.flush())

    def test_audio_edges_are_ramped_to_silence(self):
        audio = np.linspace(0.25, 0.75, 2400, dtype=np.float32)
        smoothed = smooth_audio_boundaries(audio, 24000, fade_ms=8)
        self.assertAlmostEqual(float(smoothed[0]), 0.0, places=6)
        self.assertAlmostEqual(float(smoothed[-1]), 0.0, places=6)
        self.assertLess(abs(float(np.mean(smoothed))), 0.01)


if __name__ == "__main__":
    unittest.main()
