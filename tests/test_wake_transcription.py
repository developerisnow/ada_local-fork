import unittest
from unittest.mock import patch

from core.stt import STTListener


class WakeTranscriptionTests(unittest.TestCase):
    def setUp(self):
        self.commands = []
        self.wakes = []
        self.listener = STTListener(lambda: self.wakes.append(True), self.commands.append)
        self.mode = patch('core.stt.USE_PORCUPINE_WAKE_WORD', False)
        self.mode.start()
        self.addCleanup(self.mode.stop)

    def test_requires_whole_wake_word(self):
        self.listener._handle_transcription('Turn off the light')
        self.listener._handle_transcription('Jarvison, turn off the light')
        self.assertEqual(self.commands, [])
        self.listener._handle_transcription('JARVIS, what time is it?')
        self.assertEqual(self.commands, ['what time is it'])
        self.assertEqual(len(self.wakes), 1)

    def test_wake_arms_only_one_followup(self):
        with patch('core.stt.time.monotonic', side_effect=[100, 101, 102]):
            self.listener._handle_transcription('Jarvis.')
            self.listener._handle_transcription('Read my calendar')
            self.listener._handle_transcription('Turn off the light')
        self.assertEqual(self.commands, ['Read my calendar'])

    def test_expired_wake_does_not_execute(self):
        with patch('core.stt.time.monotonic', side_effect=[100, 106]):
            self.listener._handle_transcription('Jarvis')
            self.listener._handle_transcription('Turn off the light')
        self.assertEqual(self.commands, [])

    def test_timely_recording_survives_slow_transcription(self):
        with patch('core.stt.time.monotonic', side_effect=[100, 110]):
            self.listener._handle_transcription('Jarvis')
            self.listener._handle_transcription('Read my calendar', started_at=101)
        self.assertEqual(self.commands, ['Read my calendar'])


if __name__ == '__main__':
    unittest.main()
