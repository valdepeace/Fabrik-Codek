# Docstring Generator Prompt

Generate high-quality docstrings in Google style format for the provided code.

## Rules
1. Use Google docstring format
2. Include a brief description on the first line
3. Document all parameters with types
4. Document return value
5. Include examples if the function is complex
6. Document exceptions if any

## Format

```python
def function_name(param1: Type1, param2: Type2) -> ReturnType:
    """Brief description of what the function does.

    Longer description if needed, explaining behavior,
    edge cases, or important details.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ExceptionType: When this exception is raised.

    Example:
        >>> function_name("value1", 42)
        expected_result
    """
```

## Instructions
Analyze the provided code and generate appropriate docstrings.
If the code already has docstrings, improve them if necessary.
Keep the style consistent with the rest of the project.
