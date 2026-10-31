"""Validate model answer IDs without discarding or overwriting mistakes."""

from collections import Counter


def inspect_answers(pairs, expected_ids):
    expected = set(expected_ids)
    counts = Counter(pair.key for pair in pairs)
    valid = {
        pair.key: pair.value.strip()
        for pair in pairs
        if pair.key in expected and counts[pair.key] == 1
        and pair.value.strip() and pair.value.strip().lower() != "none"
    }
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    invalid = sorted(set(counts) - expected)
    missing = sorted(expected - valid.keys())
    issues = []
    if duplicates:
        issues.append(f"Duplicate answer IDs: {duplicates}.")
    if invalid:
        issues.append(f"Unexpected answer IDs: {invalid}.")
    if missing:
        issues.append(f"Unresolved answer IDs: {missing}.")
    return valid, missing, issues
