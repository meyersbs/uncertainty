import unittest

from uncertainty import helpers


class HelpersTestCase(unittest.TestCase):
    def test_aggregate(self):
        data = ['C'] * 25
        expected = 'C'
        self.assertEqual(expected, helpers.aggregate(data))

        data = ['U'] * 25
        expected = 'U'
        self.assertEqual(expected, helpers.aggregate(data))

        data = ['C'] * 20 + ['E'] * 5
        expected = 'E'
        self.assertEqual(expected, helpers.aggregate(data))

        data = ['C'] * 15 + ['E'] * 5 + ['D'] * 5
        expected = 'U'
        self.assertEqual(expected, helpers.aggregate(data))

        data = ['C'] * 16 + ['E'] * 5 + ['D'] * 4
        expected = 'E'
        self.assertEqual(expected, helpers.aggregate(data))
