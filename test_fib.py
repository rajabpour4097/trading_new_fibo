import unittest
from main_metatrader import build_oriented_fib

class TestFibonacciOrientation(unittest.TestCase):
    def test_bullish_orientation(self):
        fib = build_oriented_fib('bullish', 110.0, 100.0)
        self.assertAlmostEqual(fib['0.0'], 110.0)
        self.assertAlmostEqual(fib['1.0'], 100.0)
        self.assertTrue(fib['0.9'] < fib['0.0'])  # یعنی فیب 0.9 پایین‌تر از نقطه ورود هست

    def test_bearish_orientation(self):
        fib = build_oriented_fib('bearish', 100.0, 110.0)
        self.assertAlmostEqual(fib['0.0'], 100.0)
        self.assertAlmostEqual(fib['1.0'], 110.0)
        self.assertTrue(fib['0.9'] > fib['0.0'])  # یعنی فیب 0.9 بالاتر از نقطه ورود هست

    def test_reversed_inputs_bullish(self):
        # ورودی‌های برعکس ولی خروجی باید درست باشه
        fib = build_oriented_fib('bullish', 100.0, 110.0)
        self.assertAlmostEqual(fib['0.0'], 110.0)
        self.assertAlmostEqual(fib['1.0'], 100.0)

    def test_reversed_inputs_bearish(self):
        fib = build_oriented_fib('bearish', 110.0, 100.0)
        self.assertAlmostEqual(fib['0.0'], 100.0)
        self.assertAlmostEqual(fib['1.0'], 110.0)

if __name__ == '__main__':
    unittest.main()
