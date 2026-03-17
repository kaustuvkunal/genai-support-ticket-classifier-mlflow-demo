from src.scorers import _extract_category, exact_category_match


def test_extract_category_exact_match():
    assert _extract_category("Incident") == "incident"
    assert _extract_category("ask for a Request") == "request"
    assert _extract_category("PROBLEM happened") == "problem"
    assert _extract_category("please do a Change") == "change"


def test_extract_category_unknown():
    assert _extract_category("no match here") == "unknown"


def test_exact_category_match_with_string_output():
    expectations = {"type": "Incident"}
    assert exact_category_match("Incident", expectations)
    assert exact_category_match("incident (outage)", expectations)


def test_exact_category_match_with_dict_output():
    expectations = {"type": "Request"}
    assert exact_category_match({"type": "Request"}, expectations)
