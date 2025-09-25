#!/usr/bin/env python3
from __future__ import annotations

from typing import Dict, Type

from .base import ProviderAdapter


_REGISTRY: Dict[str, Type[ProviderAdapter]] = {}


def register_provider(name: str, cls: Type[ProviderAdapter]) -> None:
    key = name.lower().strip()
    cls.name = key  # type: ignore[attr-defined]
    _REGISTRY[key] = cls


def get_provider(name: str, **kwargs) -> ProviderAdapter:
    key = name.lower().strip()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown provider: {name}. Registered: {list(_REGISTRY)}")
    return _REGISTRY[key](**kwargs)


def list_providers() -> Dict[str, Type[ProviderAdapter]]:
    return dict(_REGISTRY)
