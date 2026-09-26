"""Decision-model backends. ``model="jev-1.13"`` style strings resolve through
the registry here; pass an instance for full control over its parameters."""
from .base import Capabilities, DecisionModel, register_model, registered_prefixes, resolve_model
from .fake import FakeModel
from .jev import JevAPIError, JevModel

register_model("jev-", lambda name: JevModel(name=name))
register_model("fake-", lambda name: FakeModel(version=name))

__all__ = [
    "Capabilities",
    "DecisionModel",
    "FakeModel",
    "JevModel",
    "JevAPIError",
    "register_model",
    "registered_prefixes",
    "resolve_model",
]
