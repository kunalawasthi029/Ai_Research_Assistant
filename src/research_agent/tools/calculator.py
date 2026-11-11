"""Safe calculator tool using AST-based expression evaluation.

Never uses eval() or exec(). Evaluates mathematical expressions
using a whitelist of safe operators and functions via the ast module.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import Field

from research_agent.tools.results import ToolArguments, ToolOutput, failure

# Whitelisted binary operators
_SAFE_OPERATORS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Whitelisted functions
_SAFE_FUNCTIONS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pi": math.pi,
    "e": math.e,
}

# Maximum allowed power to prevent resource exhaustion
_MAX_POWER = 1000


class CalculatorInput(ToolArguments):
    """Input schema for the calculator tool."""

    expression: str = Field(
        min_length=1,
        max_length=500,
        description=(
            "A mathematical expression to evaluate. "
            "Supports: +, -, *, /, //, %, ** and functions: "
            "abs, round, min, max, sum, sqrt, log, log10, log2, sin, cos, tan. "
            "Constants: pi, e. Example: 'sqrt(144) + 2**3'"
        ),
    )


class CalculatorError(Exception):
    """Raised for unsafe or invalid calculator expressions."""

    pass


def _safe_eval_node(node: ast.AST) -> Any:
    """Recursively evaluate an AST node with safety checks."""
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return _bounded_number(node.value)
        raise CalculatorError(f"Unsupported constant type: {type(node.value).__name__}")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise CalculatorError(f"Unsupported operator: {ast.dump(node.op)}")

        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)

        # Power safety check
        if op_type is ast.Pow:
            if isinstance(right, (int, float)) and abs(right) > _MAX_POWER:
                raise CalculatorError(f"Power exponent {right} exceeds maximum of {_MAX_POWER}")
            if isinstance(left, int) and isinstance(right, int) and right > 0 and abs(left) > 1:
                if left.bit_length() * right > 4096:
                    raise CalculatorError("Calculation exceeds integer size budget")

        # Division by zero check
        if op_type in (ast.Div, ast.FloorDiv, ast.Mod):
            if right == 0:
                raise CalculatorError("Division by zero")

        return _bounded_number(_SAFE_OPERATORS[op_type](left, right))

    if isinstance(node, ast.UnaryOp):
        unary_type = type(node.op)
        if unary_type not in _SAFE_OPERATORS:
            raise CalculatorError(f"Unsupported unary operator: {ast.dump(node.op)}")
        operand = _safe_eval_node(node.operand)
        return _bounded_number(_SAFE_OPERATORS[unary_type](operand))

    if isinstance(node, ast.Call):
        if node.keywords:
            raise CalculatorError("Keyword arguments are unsupported")
        if not isinstance(node.func, ast.Name):
            raise CalculatorError("Only simple function calls are supported (no methods)")
        func_name = node.func.id
        if func_name not in _SAFE_FUNCTIONS:
            raise CalculatorError(
                f"Unsupported function: {func_name}. "
                f"Allowed: {', '.join(sorted(_SAFE_FUNCTIONS.keys()))}"
            )
        func = _SAFE_FUNCTIONS[func_name]
        args = [_safe_eval_node(arg) for arg in node.args]
        return _bounded_number(func(*args))

    if isinstance(node, ast.Name):
        name = node.id
        if name in _SAFE_FUNCTIONS:
            value = _SAFE_FUNCTIONS[name]
            if isinstance(value, (int, float)):
                return value  # Constants like pi, e
        raise CalculatorError(f"Unknown variable: {name}. Available constants: pi, e")

    if isinstance(node, ast.List):
        return [_safe_eval_node(el) for el in node.elts]

    if isinstance(node, ast.Tuple):
        return tuple(_safe_eval_node(el) for el in node.elts)

    raise CalculatorError(f"Unsupported expression element: {type(node).__name__}")


def _bounded_number(value: Any) -> float | int:
    """Reject non-real/nonfinite results and expensive integer intermediates."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalculatorError("Expected a real numeric result")
    if isinstance(value, int) and value.bit_length() > 4096:
        raise CalculatorError("Calculation exceeds integer size budget")
    if isinstance(value, float) and not math.isfinite(value):
        raise CalculatorError("Nonfinite numeric result")
    return int(value) if isinstance(value, int) else float(value)


def safe_calculate(expression: str) -> float | int:
    """Safely evaluate a mathematical expression.

    Parses the expression into an AST and evaluates only whitelisted
    operations. Never uses eval() or exec().

    Raises CalculatorError for unsafe or invalid expressions.
    """
    if not expression or not expression.strip():
        raise CalculatorError("Empty expression")

    # Length limit
    if len(expression) > 500:
        raise CalculatorError("Expression too long (max 500 characters)")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise CalculatorError(f"Invalid expression syntax: {e}") from e

    if sum(1 for _ in ast.walk(tree)) > 128:
        raise CalculatorError("Expression exceeds AST size budget")
    return _bounded_number(_safe_eval_node(tree))


def calculate(expression: str) -> str:
    """Calculator tool function: evaluate expression and return formatted result."""
    try:
        result = safe_calculate(expression)
        return ToolOutput(
            data={
                "expression": expression,
                "value": result,
                "display": f"Calculation: {expression} = {result}",
            }
        ).model_dump_json()
    except Exception:
        return failure("Calculator error: invalid or unsafe expression")


calculator_tool = StructuredTool.from_function(
    func=calculate,
    name="calculator",
    description=(
        "Safely evaluate mathematical expressions. "
        "Supports arithmetic (+, -, *, /, **, %), "
        "functions (sqrt, log, sin, cos, etc.), and constants (pi, e). "
        "Example: 'sqrt(144) + 2**3'"
    ),
    args_schema=CalculatorInput,
)
