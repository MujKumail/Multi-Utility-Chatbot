import ast
import operator
import re
from datetime import datetime

import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo_tools")


# -------------------
# Calculator Tool
# -------------------
_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}

_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _evaluate_expression_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)

    if isinstance(node, ast.Num):
        return float(node.n)

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_expression_node(node.left)
        right = _evaluate_expression_node(node.right)
        return float(_BINARY_OPERATORS[type(node.op)](left, right))

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        operand = _evaluate_expression_node(node.operand)
        return float(_UNARY_OPERATORS[type(node.op)](operand))

    raise ValueError("Unsupported expression. Use numbers and +, -, *, /, %, //, or **.")


@mcp.tool()
def calculate_expression(expression: str) -> dict:
    """Evaluate a math expression such as (245 * 18) / 6 or 2 ** 8."""
    try:
        parsed = ast.parse(expression, mode="eval")
        result = _evaluate_expression_node(parsed.body)
    except ZeroDivisionError:
        raise ValueError("Cannot divide by zero.")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression: {exc.msg}") from exc

    return {
        "expression": expression,
        "result": result,
    }


# -------------------
# String Tools
# -------------------
@mcp.tool()
def reverse_string(text: str) -> str:
    """Reverse a given string."""
    return text[::-1]


@mcp.tool()
def word_count(text: str) -> int:
    """Count the number of words in a given text."""
    return len(text.split())


@mcp.tool()
def to_uppercase(text: str) -> str:
    """Convert text to uppercase."""
    return text.upper()


@mcp.tool()
def to_lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


@mcp.tool()
def extract_urls(text: str) -> list[str]:
    """Extract URLs from text."""
    return re.findall(r"https?://[^\s]+", text)


@mcp.tool()
def slugify_text(text: str) -> str:
    """Convert text into a URL-friendly slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return slug or "untitled"


# -------------------
# Utility Tools
# -------------------
@mcp.tool()
def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert temperature from Celsius to Fahrenheit."""
    return float(celsius) * 9 / 5 + 32


@mcp.tool()
def fahrenheit_to_celsius(fahrenheit: float) -> float:
    """Convert temperature from Fahrenheit to Celsius."""
    return (float(fahrenheit) - 32) * 5 / 9


@mcp.tool()
def km_to_miles(km: float) -> float:
    """Convert kilometers to miles."""
    return float(km) * 0.621371


@mcp.tool()
def miles_to_km(miles: float) -> float:
    """Convert miles to kilometers."""
    return float(miles) * 1.60934


@mcp.tool()
def get_current_datetime() -> dict:
    """Return the current local date and time."""
    now = datetime.now().astimezone()
    return {
        "iso": now.isoformat(timespec="seconds"),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "timezone": str(now.tzinfo),
    }


@mcp.tool()
def days_between_dates(start_date: str, end_date: str) -> dict:
    """Return the absolute day difference between two YYYY-MM-DD dates."""
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    delta_days = abs((end - start).days)
    return {
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "days": delta_days,
    }


@mcp.tool()
def calculate_percentage(part: float, whole: float) -> dict:
    """Calculate what percentage part is of whole."""
    whole = float(whole)
    if whole == 0:
        raise ValueError("Whole cannot be zero.")
    percentage = (float(part) / whole) * 100
    return {
        "part": float(part),
        "whole": whole,
        "percentage": round(percentage, 4),
    }


@mcp.tool()
def percentage_change(old_value: float, new_value: float) -> dict:
    """Calculate percentage increase or decrease from old_value to new_value."""
    old_value = float(old_value)
    new_value = float(new_value)
    if old_value == 0:
        raise ValueError("old_value cannot be zero.")
    change = ((new_value - old_value) / old_value) * 100
    return {
        "old_value": old_value,
        "new_value": new_value,
        "percentage_change": round(change, 4),
    }


# -------------------
# Currency Tool
# -------------------
@mcp.tool()
def get_exchange_rate(base: str, target: str) -> dict:
    """
    Get the current exchange rate between two currencies.
    Examples: base='USD' target='INR', base='EUR' target='GBP'.
    """
    url = f"https://api.exchangerate-api.com/v4/latest/{base.upper()}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return {"error": "Failed to fetch exchange rates. Try again later."}
        data = response.json()
        rate = data["rates"].get(target.upper())
        if rate is None:
            return {"error": f"Currency code '{target.upper()}' not recognised."}
        return {
            "base": base.upper(),
            "target": target.upper(),
            "rate": rate,
            "result": f"1 {base.upper()} = {rate} {target.upper()}",
        }
    except requests.exceptions.RequestException as exc:
        return {"error": f"Network error: {str(exc)}"}


if __name__ == "__main__":
    mcp.run()
