"""Decision-model backends.

Strings resolve through the registry here: "hf:<repo id>[@revision]" for a
local open-weights model, "jev-latest" / "jev-preview" for TypeSafe's hosted Jev,
"fake-<version>" for the offline test model. Pass an instance for full
control over its parameters."""
from .base import Capabilities, DecisionModel, register_model, registered_prefixes, resolve_model
from .fake import FakeModel
from .jev import JevAPIError, JevModel
from .transformers import TransformersModel

register_model("jev-", lambda name: JevModel(name=name))
register_model("fake-", lambda name: FakeModel(version=name))


def _hf(spec: str) -> TransformersModel:
    name, _, revision = spec[len("hf:"):].partition("@")
    return TransformersModel(name, revision=revision or "main")


register_model("hf:", _hf)

__all__ = [
    "Capabilities",
    "DecisionModel",
    "FakeModel",
    "JevModel",
    "JevAPIError",
    "TransformersModel",
    "register_model",
    "registered_prefixes",
    "resolve_model",
]
