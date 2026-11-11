"""Players = input fields of one call; rebuilding the user message for a coalition of fields.

Players (at most 7): user_task, service, user_memory, prior_steps (top level of the user JSON) and tool, tool_args,
description (inside assistant_proposed_tool_call). Only fields present in the record are players. Removing a field
deletes its key; if tool, tool_args and description are all removed, assistant_proposed_tool_call is dropped. The
original key order is kept and the record's own JSON format is reused, so the full coalition is byte-identical to the
original user message (asserted in CallFields.__init__).
"""
import json

TOP_FIELDS = ("user_task", "service", "user_memory", "prior_steps")
APTC = "assistant_proposed_tool_call"
APTC_FIELDS = ("tool", "tool_args", "description")
ALL_FIELDS = TOP_FIELDS + APTC_FIELDS

# The 1,313 single-step test records use three serialisations (found by brute force over json.dumps options).
FORMATS = {
    "indent2_utf8": dict(indent=2, ensure_ascii=False),
    "compact_ascii": dict(ensure_ascii=True),
    "compact_utf8": dict(ensure_ascii=False),
}


def detect_format(user_content: str) -> str:
    obj = json.loads(user_content)
    for name, kw in FORMATS.items():
        if json.dumps(obj, **kw) == user_content:
            return name
    raise ValueError("no known JSON serialisation reproduces this user message byte for byte")


class CallFields:
    """One single-step record: its system message, field values and serialisation."""

    def __init__(self, record: dict):
        msgs = record["prompt"]
        users = [m["content"] for m in msgs if m.get("role") == "user"]
        assert len(users) == 1, "pilot uses single-step records only"
        self.system = next((m["content"] for m in msgs if m.get("role") == "system"), "")
        self.original_user = users[0]
        self.fmt = detect_format(self.original_user)
        self.obj = json.loads(self.original_user)
        unknown_top = set(self.obj) - set(TOP_FIELDS) - {APTC}
        assert not unknown_top, f"unexpected top-level keys {unknown_top}"
        aptc = self.obj.get(APTC, {})
        unknown_sub = set(aptc) - set(APTC_FIELDS)
        assert not unknown_sub, f"unexpected tool-call keys {unknown_sub}"
        # players in the order they appear in the original JSON
        self.players = []
        for k in self.obj:
            if k == APTC:
                self.players += [s for s in aptc if s in APTC_FIELDS]
            else:
                self.players.append(k)
        self.values = {p: (aptc[p] if p in APTC_FIELDS else self.obj[p]) for p in self.players}
        # byte-identical round trip of the full coalition
        assert self.user_content(self.players) == self.original_user, "round trip failed"

    def user_content(self, coalition) -> str:
        keep = set(coalition)
        assert keep <= set(self.players), f"unknown players {keep - set(self.players)}"
        out = {}
        for k, v in self.obj.items():
            if k == APTC:
                sub = {s: sv for s, sv in v.items() if s in keep}
                if sub:
                    out[APTC] = sub
            elif k in keep:
                out[k] = v
        return json.dumps(out, **FORMATS[self.fmt])
