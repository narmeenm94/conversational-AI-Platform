import unittest

from pipeline.knowledge_router import should_retrieve_character_knowledge


class KnowledgeRouterTests(unittest.TestCase):
    def test_generic_project_conversation_does_not_inject_workplace_docs(self):
        self.assertFalse(should_retrieve_character_knowledge(
            "I'm working on a new mixed reality project.",
            [],
        ))

    def test_explicit_character_knowledge_topic_is_retrieved(self):
        self.assertTrue(should_retrieve_character_knowledge(
            "What projects is the HXRC team working on?",
            [],
        ))

    def test_short_follow_up_keeps_established_knowledge_topic(self):
        self.assertTrue(should_retrieve_character_knowledge(
            "Who else?",
            ["Tell me about your HXRC colleagues."],
        ))


if __name__ == "__main__":
    unittest.main()
