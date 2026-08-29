"""Community-floaty boundary for the native scratch audio/video model.

The owner foundation in this module is the *entire* native model: audio codec,
vision encoder, timeline projections, talker, and the original HybridLM
thinker.  Only low-rank adapters inside the thinker are community trainable.

Attachment mutates the existing ``ScratchMultimodalLM.thinker`` reference once
so all native APIs continue to operate normally.  ``NativeFloatyBoundary`` is
intentionally not an ``nn.Module``; it cannot register the same owner tensors a
second time.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .foundation_floaty import (
    FloatyConfig,
    FoundationFloatyLM,
    ParameterPartition,
)
from .hybrid import HybridLM
from .multimodal import ScratchMultimodalLM


_NATIVE_FLOATY_PREFIX = "thinker.floaty."


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _framed(hasher: Any, value: bytes | memoryview) -> None:
    size = value.nbytes if isinstance(value, memoryview) else len(value)
    hasher.update(size.to_bytes(8, "big"))
    hasher.update(value)


def _array_bytes(value: mx.array) -> memoryview:
    raw = mx.contiguous(value).view(mx.uint8).reshape(-1)
    mx.eval(raw)
    return memoryview(raw)


def _flat_parameters(module: nn.Module) -> list[tuple[str, mx.array]]:
    return sorted(tree_flatten(module.parameters()))


def _flat_trainable_names(module: nn.Module) -> tuple[str, ...]:
    return tuple(
        sorted(name for name, _ in tree_flatten(module.trainable_parameters()))
    )


class NativeFloatyBoundary:
    """Non-registering ownership and optimizer boundary for a native model."""

    def __init__(
        self,
        model: ScratchMultimodalLM,
        thinker: FoundationFloatyLM,
    ) -> None:
        if type(model) is not ScratchMultimodalLM:
            raise TypeError("model must be an exact ScratchMultimodalLM instance")
        if type(thinker) is not FoundationFloatyLM or model.thinker is not thinker:
            raise TypeError("thinker must be the model's attached FoundationFloatyLM")
        self.model = model
        self.thinker = thinker
        self.floaty = thinker.floaty

    def parameter_partition(self) -> ParameterPartition:
        names = tuple(name for name, _ in _flat_parameters(self.model))
        floaty = tuple(name for name in names if name.startswith(_NATIVE_FLOATY_PREFIX))
        foundation = tuple(
            name for name in names if not name.startswith(_NATIVE_FLOATY_PREFIX)
        )
        return ParameterPartition(foundation=foundation, floaty=floaty)

    def foundation_parameter_names(self) -> tuple[str, ...]:
        return self.parameter_partition().foundation

    def floaty_parameter_names(self) -> tuple[str, ...]:
        """Return model-qualified names used by native gradient trees."""

        return self.parameter_partition().floaty

    def queue_floaty_parameter_names(self) -> tuple[str, ...]:
        """Return queue-qualified ``floaty.*`` names from the signed schema."""

        return tuple(
            tensor["name"] for tensor in self.thinker.config_manifest()["tensors"]
        )

    def community_to_native_parameter_names(self) -> dict[str, str]:
        """Map external ``floaty.*`` keys to internal native gradient names."""

        mapping = {
            community: f"thinker.{community}"
            for community in self.queue_floaty_parameter_names()
        }
        if tuple(mapping.values()) != self.floaty_parameter_names():
            raise RuntimeError("Community/native floaty name mapping is inconsistent")
        return mapping

    def trainable_parameter_names(self) -> tuple[str, ...]:
        return _flat_trainable_names(self.model)

    def foundation_parameter_count(self) -> int:
        return sum(
            value.size
            for name, value in _flat_parameters(self.model)
            if not name.startswith(_NATIVE_FLOATY_PREFIX)
        )

    def floaty_parameter_count(self) -> int:
        return sum(
            value.size
            for name, value in _flat_parameters(self.model)
            if name.startswith(_NATIVE_FLOATY_PREFIX)
        )

    def assert_parameter_partition(self) -> ParameterPartition:
        """Fail closed if owner tensors leak into the trainable surface."""

        self.thinker.assert_parameter_partition()
        partition = self.parameter_partition()
        if not partition.foundation or not partition.floaty:
            raise RuntimeError(
                "Native owner and floaty partitions must both be non-empty"
            )
        if set(partition.foundation) & set(partition.floaty):
            raise RuntimeError("Native owner and floaty parameter names overlap")
        actual = tuple(name for name, _ in _flat_parameters(self.model))
        if actual != partition.all:
            raise RuntimeError("Native model contains an unclassified parameter")

        expected_floaty = tuple(
            f"thinker.{name}" for name in self.thinker.floaty_parameter_names()
        )
        if partition.floaty != expected_floaty:
            raise RuntimeError("Native floaty partition differs from thinker adapters")
        trainable = set(self.trainable_parameter_names())
        if not trainable.issubset(partition.floaty):
            raise RuntimeError("A native owner-foundation parameter is trainable")

        parameters = dict(_flat_parameters(self.model))
        owner_ids = {id(parameters[name]) for name in partition.foundation}
        floaty_ids = {id(parameters[name]) for name in partition.floaty}
        if owner_ids & floaty_ids:
            raise RuntimeError("Native owner and floaty tensors alias")
        if len(owner_ids) != len(partition.foundation):
            raise RuntimeError("A native owner tensor is registered more than once")
        if len(floaty_ids) != len(partition.floaty):
            raise RuntimeError("A native floaty tensor is registered more than once")
        if self.floaty_parameter_count() != self.thinker.floaty_parameter_count():
            raise RuntimeError(
                "Native floaty parameter count differs from thinker adapters"
            )
        return partition

    def freeze_foundation(self) -> None:
        """Freeze every native tensor, then arm only the thinker floaty layer."""

        self.model.freeze()
        self.thinker.foundation.freeze()
        self.floaty.unfreeze()
        partition = self.assert_parameter_partition()
        if self.trainable_parameter_names() != tuple(sorted(partition.floaty)):
            raise RuntimeError("Native floaty whitelist was not fully armed")

    def floaty_parameters(self) -> dict[str, Any]:
        """Return the local adapter tree and nothing from the native foundation."""

        partition = self.assert_parameter_partition()
        parameters = self.floaty.trainable_parameters()
        local_names = tuple(sorted(name for name, _ in tree_flatten(parameters)))
        qualified = tuple(f"thinker.floaty.{name}" for name in local_names)
        if qualified != tuple(sorted(partition.floaty)):
            raise RuntimeError(
                "The native floaty optimizer surface is frozen; call freeze_foundation() first"
            )
        return parameters

    def config_manifest(self) -> dict[str, Any]:
        """Return the canonical native architecture plus queue-ready tensor schema."""

        mapping = self.community_to_native_parameter_names()
        manifest = {
            "schema": "tinydog.native-foundation-floaty.config.v1",
            "native_type": "mini_arithmetic.multimodal.ScratchMultimodalLM",
            "native_config": self.model.config.to_dict(),
            "modality_registry": self.model.registry.to_dict(),
            "thinker_floaty": self.thinker.config_manifest(),
            "community_to_native": [
                {"community_name": community, "native_name": native}
                for community, native in mapping.items()
            ],
        }
        return json.loads(_canonical_json(manifest))

    def config_hash(self) -> str:
        return hashlib.sha256(_canonical_json(self.config_manifest())).hexdigest()

    def foundation_hash(self) -> str:
        """Hash config and every non-floaty native tensor byte-for-byte."""

        partition = self.assert_parameter_partition()
        parameters = dict(_flat_parameters(self.model))
        owner_manifest = {
            "schema": "tinydog.native-owner-foundation.v1",
            "native_type": "mini_arithmetic.multimodal.ScratchMultimodalLM",
            "native_config": self.model.config.to_dict(),
            "modality_registry": self.model.registry.to_dict(),
        }
        hasher = hashlib.sha256()
        _framed(hasher, b"tinydog.native-owner-foundation.v1")
        _framed(hasher, _canonical_json(owner_manifest))
        _framed(hasher, str(len(partition.foundation)).encode("ascii"))
        for name in partition.foundation:
            value = parameters[name]
            _framed(hasher, name.encode("utf-8"))
            _framed(hasher, str(value.dtype).encode("ascii"))
            _framed(hasher, _canonical_json({"shape": list(value.shape)}))
            _framed(hasher, _array_bytes(value))
        return hasher.hexdigest()


def attach_floaty_thinker(
    native_model: ScratchMultimodalLM,
    config: FloatyConfig | None = None,
) -> NativeFloatyBoundary:
    """Attach or re-arm one community floaty thinker on a native model.

    The original HybridLM object becomes ``thinker.foundation``; it is not
    copied and is registered only through that one path.  Repeated attachment
    is idempotent only when the requested config is identical.
    """

    if type(native_model) is not ScratchMultimodalLM:
        raise TypeError("native_model must be an exact ScratchMultimodalLM instance")
    current = native_model.thinker
    if type(current) is FoundationFloatyLM:
        if config is not None and config != current.floaty_config:
            raise ValueError("Native model already has a different floaty config")
        boundary = NativeFloatyBoundary(native_model, current)
        boundary.freeze_foundation()
        return boundary
    if type(current) is not HybridLM:
        raise TypeError("Native thinker must be an unattached exact HybridLM instance")

    thinker = FoundationFloatyLM(current, config)
    native_model.thinker = thinker
    boundary = NativeFloatyBoundary(native_model, thinker)
    boundary.freeze_foundation()
    return boundary
