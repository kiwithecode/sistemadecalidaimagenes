def missing_tokens(text: str, expected_tokens: list[str]):
    if not expected_tokens:
        return []
    low = text.lower()
    return [t for t in expected_tokens if t.lower() not in low]
