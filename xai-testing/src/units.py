"""Parse the scored user message of a sample into units and rebuild it, optionally with one unit masked.

The scoring function sends only the FIRST user message (plan, Section 4), so units are taken from that message.
The user content is a JSON string written by one of four serialisers (indent 2 or compact, ASCII-escaped or not).
`detect_format` finds the one that reproduces the original byte for byte; re-serialising with it is then exact
(checked in S1 against the state string that call_openrouter_decision itself builds). Samples for which no
serialiser matches would need string surgery; S1 reports how many there are (0 in all three splits).
"""
import copy
import json

FORMATS = {  # name -> json.dumps kwargs
    "indent2_ascii": dict(indent=2, ensure_ascii=True),
    "compact_ascii": dict(indent=None, ensure_ascii=True),
    "indent2_utf8": dict(indent=2, ensure_ascii=False),
    "compact_utf8": dict(indent=None, ensure_ascii=False),
}
TOP_UNITS = ["user_task", "user_memory", "prior_steps", "service"]
CALL_UNITS = ["tool", "description"]
CALL_KEY = "assistant_proposed_tool_call"


def first_user_index(sample):
    return next(i for i, m in enumerate(sample["prompt"]) if m.get("role") == "user")


def scored_content(sample):
    return sample["prompt"][first_user_index(sample)]["content"]


def detect_format(content):
    obj = json.loads(content)
    for name, kw in FORMATS.items():
        if json.dumps(obj, **kw) == content:
            return name
    return None


def parse(sample):
    """-> (obj, fmt, units). units are ids such as 'user_task', 'tool', 'tool_args.invoice_name', in a fixed order."""
    content = scored_content(sample)
    obj = json.loads(content)
    fmt = detect_format(content)
    call = obj.get(CALL_KEY, {}) or {}
    units = [u for u in TOP_UNITS if u in obj]
    units += [u for u in CALL_UNITS if u in call]
    units += [f"tool_args.{k}" for k in (call.get("tool_args") or {})]
    return obj, fmt, units


def unit_type(unit):
    return "tool_args" if unit.startswith("tool_args.") else unit


def get_value(obj, unit):
    if unit in TOP_UNITS:
        return obj[unit]
    call = obj[CALL_KEY]
    if unit in CALL_UNITS:
        return call[unit]
    return call["tool_args"][unit.split(".", 1)[1]]


def apply_mask(obj, unit, strategy="remove", value=None):
    """Return a deep copy of obj with one unit masked. remove: the key is dropped (M1). replace: its value is set to
    `value` (M2; the donor value is chosen by the caller and logged there). Key order is preserved."""
    o = copy.deepcopy(obj)
    if unit in TOP_UNITS:
        parent, key = o, unit
    elif unit in CALL_UNITS:
        parent, key = o[CALL_KEY], unit
    else:
        parent, key = o[CALL_KEY]["tool_args"], unit.split(".", 1)[1]
    if key not in parent:
        raise KeyError(unit)
    if strategy == "remove":
        del parent[key]
    elif strategy == "replace":
        parent[key] = value
    else:
        raise ValueError(strategy)
    return o


def render(obj, fmt):
    if fmt is None:
        raise ValueError("no exact serialiser for this sample; string surgery would be needed")
    return json.dumps(obj, **FORMATS[fmt])


def with_content(sample, content):
    """Copy of the sample whose scored (first) user message has the given content; everything else unchanged."""
    s = copy.deepcopy(sample)
    s["prompt"][first_user_index(s)]["content"] = content
    return s


def variant(sample, unit=None, strategy="remove", value=None):
    """Sample to send for the baseline (unit=None) or for one masked unit."""
    obj, fmt, _ = parse(sample)
    if unit is not None:
        obj = apply_mask(obj, unit, strategy, value)
    return with_content(sample, render(obj, fmt))
