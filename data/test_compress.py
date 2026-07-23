"""Smallest possible check for the correctness gates. Run: python test_compress.py"""
from compress import CHECKS, extract_final_number

assert extract_final_number("The answer is 42.") == "42"
assert extract_final_number("Total: $1,234.50") == "1234.50"
assert extract_final_number("no digits here") is None

assert CHECKS["gsm8k"]("She has 12 apples.", "The final count is 12.") is True
assert CHECKS["gsm8k"]("She has 13 apples.", "The final count is 12.") is False

assert CHECKS["json_exact"]('{"a": 1}', '{"a": 1}') is True
assert CHECKS["json_exact"]('{"a": 1}', '{"a": 2}') is False
assert CHECKS["json_exact"]("not json", '{"a": 1}') is False

print("ok")
