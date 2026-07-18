import unittest

from pipeline.client_events import ClientEventProcessor


class ClientEventProcessorTests(unittest.TestCase):
    def test_selects_semantic_animation_state(self):
        self.assertEqual(
            ClientEventProcessor._cognitive_state("What did you say earlier?"),
            "remembering",
        )
        self.assertEqual(
            ClientEventProcessor._cognitive_state("Who works on the HXRC team?"),
            "searching",
        )
        self.assertEqual(
            ClientEventProcessor._cognitive_state("Why is the sky blue?"),
            "thinking",
        )


if __name__ == "__main__":
    unittest.main()
