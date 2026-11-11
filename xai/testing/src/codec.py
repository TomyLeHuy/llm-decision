"""llmSHAP PromptCodec and ValueFunction for field-level Shapley on the decision model."""
import json

from llmSHAP.data_handler import DataHandler
from llmSHAP.generation import Generation
from llmSHAP.prompt_codec import PromptCodec
from llmSHAP.value_functions import ValueFunction

from common import CLASSES, make_state
from fields import CallFields


def make_data_handler(call: CallFields, fixed=()) -> DataHandler:
    # one entry per present field, in original order. System message + questions are outside the data. Fields in
    # `fixed` are llmSHAP permanent keys: present in every coalition, not players (default: none, 7-player design).
    return DataHandler({p: call.values[p] for p in call.players}, permanent_keys=set(fixed) & set(call.players))


class FieldCodec(PromptCodec):
    """Builds the full state string from the fields of the coalition (not DataHandler.to_string)."""

    def __init__(self, call: CallFields):
        self.call = call

    def build_prompt(self, data_handler: DataHandler, indexes) -> str:
        present = data_handler.get_data(indexes, mask=False)          # {field: value} for the coalition
        return make_state(self.call.system, self.call.user_content(present.keys()))

    def parse_generation(self, model_output: str) -> Generation:
        return Generation(output=model_output)


class ClassProbability(ValueFunction):
    """v(S) = probability of one class for the coalition's generation; NaN if the provider refused / errored."""

    def __init__(self, cls: str):
        assert cls in CLASSES
        self.cls = cls

    def __call__(self, base_generation: Generation, coalition_generation: Generation) -> float:
        out = json.loads(coalition_generation.output)
        if out.get("status") != "ok":
            return float("nan")
        return float(out["p"][self.cls])
