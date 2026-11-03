from actionproof.rollout.lite_format import extract_tool_calls_from_text as parse


def test_parses_cohere_list_and_ignores_trailing_text():
    t = '[{"tool_name": "lookup_order", "parameters": {"order_id": "O-1"}}]\nthen more'
    (c,) = parse(t)
    assert c["function"]["name"] == "lookup_order" and '"O-1"' in c["function"]["arguments"]


def test_no_call_returns_none():
    assert parse("The refund is complete.") is None
