"""Pytest configuration for documentation tests.

This conftest.py prevents pytest from loading parent fixtures
that have heavy dependencies (pytorch, cuvis_ai_core, etc.)
which are not needed for documentation testing.
"""

# Intentionally empty - isolates docs tests from main test fixtures
