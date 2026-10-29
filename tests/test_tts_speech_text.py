import unittest

from core.tts import _speech_text


class SpeechTextTests(unittest.TestCase):
    def test_calendar_bold_is_not_spoken(self):
        self.assertEqual(
            _speech_text('- **Dentist Appointment** at 10:00 AM\n- **Team Call** at 3:00 PM'),
            '- Dentist Appointment at 10:00 AM\n- Team Call at 3:00 PM',
        )

    def test_arithmetic_and_negative_values_are_preserved(self):
        text = '2 * 2 = 4; 2 ** 3 + 4 ** 2 = 24; 2**3 + 4**2 = 24\n- 5'
        self.assertEqual(_speech_text(text), text)


if __name__ == '__main__':
    unittest.main()
