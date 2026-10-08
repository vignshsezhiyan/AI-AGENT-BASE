import pytest

from agent.tools import add_numbers, divide_numbers, multiply_numbers, power_of_number, subtract_numbers
from app.tools.registry import ToolRegistry


@pytest.fixture
def registry() -> ToolRegistry:
    r = ToolRegistry(timeout_seconds=5, max_result_chars=1000, approval_patterns=[])
    for fn in (add_numbers, subtract_numbers, multiply_numbers, divide_numbers, power_of_number):
        r.register(fn.__agent_tool__)
    return r


@pytest.mark.parametrize("tool, args, expected", [
    ("add_numbers", {"a": 2, "b": 3}, '{"result": 5}'),
    ("subtract_numbers", {"a": 2, "b": 3.5}, '{"result": -1.5}'),
    ("multiply_numbers", {"a": 4, "b": 2.5}, '{"result": 10}'),
    ("divide_numbers", {"a": 1, "b": 4}, '{"result": 0.25}'),
    ("power_of_number", {"base": 2, "exponent": 10}, '{"result": 1024}'),
])
async def test_calculator(registry, tool, args, expected):
    result = await registry.execute(tool, args)
    assert result.ok and result.content == expected


@pytest.mark.parametrize("tool, args, error", [
    ("divide_numbers", {"a": 1, "b": 0}, "divide by zero"),
    ("power_of_number", {"base": 10, "exponent": 999}, "too large"),
    ("power_of_number", {"base": 2, "exponent": 5000}, "invalid arguments"),
    ("power_of_number", {"base": -8, "exponent": 0.5}, "no real result"),
    ("add_numbers", {"a": "ten", "b": 1}, "invalid arguments"),
])
async def test_calculator_errors(registry, tool, args, error):
    result = await registry.execute(tool, args)
    assert not result.ok and error in result.content
