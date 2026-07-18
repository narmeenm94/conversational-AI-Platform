import unittest

from pipeline.pocket_tts_service import CompleteSentenceAggregator


class CompleteSentenceAggregatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_releases_complete_sentence_without_next_sentence_lookahead(self):
        aggregator = CompleteSentenceAggregator()
        chunks = [item.text async for item in aggregator.aggregate("I'm good. ")]
        self.assertEqual(chunks, ["I'm good."])

    async def test_drops_incomplete_tail(self):
        aggregator = CompleteSentenceAggregator()
        chunks = [item.text async for item in aggregator.aggregate("Still forming")]
        self.assertEqual(chunks, [])
        self.assertIsNone(await aggregator.flush())


if __name__ == "__main__":
    unittest.main()
