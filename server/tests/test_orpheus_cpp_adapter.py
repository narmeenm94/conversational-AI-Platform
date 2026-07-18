import unittest

import numpy as np

from pipeline.orpheus_cpp_tts_service import OrpheusCppTTSService


class OrpheusCppAdapterTests(unittest.TestCase):
    def test_float_audio_is_clipped_and_converted_to_pcm16(self):
        pcm = OrpheusCppTTSService._pcm16(
            np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        )
        samples = np.frombuffer(pcm, dtype=np.int16)
        np.testing.assert_array_equal(
            samples, np.array([-32767, -32767, 0, 32767, 32767], dtype=np.int16)
        )

    def test_existing_int16_audio_is_not_rescaled(self):
        source = np.array([[-1200, 0, 1200]], dtype=np.int16)
        self.assertEqual(
            np.frombuffer(OrpheusCppTTSService._pcm16(source), dtype=np.int16).tolist(),
            [-1200, 0, 1200],
        )


if __name__ == "__main__":
    unittest.main()
