"""Unit tests for the calculator tool (safe AST-based evaluation)."""

from __future__ import annotations

import math

import pytest

from research_agent.tools.calculator import (
    CalculatorError,
    calculate,
    calculator_tool,
    safe_calculate,
)


class TestSafeCalculate:
    """Tests for the safe expression evaluator."""

    # --- Basic arithmetic ---

    def test_addition(self) -> None:
        assert safe_calculate("2 + 3") == 5

    def test_subtraction(self) -> None:
        assert safe_calculate("10 - 4") == 6

    def test_multiplication(self) -> None:
        assert safe_calculate("6 * 7") == 42

    def test_division(self) -> None:
        assert safe_calculate("15 / 4") == 3.75

    def test_floor_division(self) -> None:
        assert safe_calculate("15 // 4") == 3

    def test_modulo(self) -> None:
        assert safe_calculate("17 % 5") == 2

    def test_power(self) -> None:
        assert safe_calculate("2 ** 10") == 1024

    def test_negative_number(self) -> None:
        assert safe_calculate("-5") == -5

    def test_complex_expression(self) -> None:
        assert safe_calculate("(2 + 3) * 4 - 1") == 19

    # --- Functions ---

    def test_sqrt(self) -> None:
        assert safe_calculate("sqrt(144)") == 12.0

    def test_abs(self) -> None:
        assert safe_calculate("abs(-42)") == 42

    def test_round(self) -> None:
        assert safe_calculate("round(3.7)") == 4

    def test_log(self) -> None:
        result = safe_calculate("log(e)")
        assert abs(result - 1.0) < 1e-10

    def test_sin(self) -> None:
        result = safe_calculate("sin(0)")
        assert result == 0.0

    def test_cos(self) -> None:
        result = safe_calculate("cos(0)")
        assert result == 1.0

    # --- Constants ---

    def test_pi(self) -> None:
        result = safe_calculate("pi")
        assert abs(result - math.pi) < 1e-10

    def test_e(self) -> None:
        result = safe_calculate("e")
        assert abs(result - math.e) < 1e-10

    # --- Safety checks ---

    def test_division_by_zero(self) -> None:
        with pytest.raises(CalculatorError, match="Division by zero"):
            safe_calculate("1 / 0")

    def test_power_too_large(self) -> None:
        with pytest.raises(CalculatorError, match="exceeds maximum"):
            safe_calculate("2 ** 10000")

    def test_empty_expression(self) -> None:
        with pytest.raises(CalculatorError, match="Empty expression"):
            safe_calculate("")

    def test_expression_too_long(self) -> None:
        with pytest.raises(CalculatorError, match="too long"):
            safe_calculate("1 + " * 200 + "1")

    def test_invalid_syntax(self) -> None:
        with pytest.raises(CalculatorError, match="Invalid expression"):
            safe_calculate("2 +* 3")

    def test_unknown_function(self) -> None:
        with pytest.raises(CalculatorError, match="Unsupported function"):
            safe_calculate("exec('malicious')")

    def test_import_blocked(self) -> None:
        with pytest.raises(CalculatorError):
            safe_calculate("__import__('os').system('ls')")

    def test_unknown_variable(self) -> None:
        with pytest.raises(CalculatorError, match="Unknown variable"):
            safe_calculate("x + 1")

    def test_string_constants_blocked(self) -> None:
        with pytest.raises(CalculatorError, match="Unsupported constant"):
            safe_calculate("'hello'")


class TestCalculateTool:
    """Tests for the tool wrapper function."""

    def test_successful_calculation(self) -> None:
        result = calculate("2 + 3")
        assert "= 5" in result

    def test_error_returns_message(self) -> None:
        result = calculate("1 / 0")
        assert "Calculator error" in result

    def test_tool_metadata(self) -> None:
        assert calculator_tool.name == "calculator"
        assert "mathematical" in calculator_tool.description.lower()
