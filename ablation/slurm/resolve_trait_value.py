"""Resolve a scalar value from a trait registry YAML file."""

from __future__ import annotations

import argparse

from omegaconf import OmegaConf


def resolve_trait_value(path: str, trait_name: str, field: str) -> str:
    config = OmegaConf.load(path)
    traits = config.get("traits", {})
    if trait_name not in traits:
        raise KeyError(f"Trait {trait_name!r} not found in {path}; available: {', '.join(traits)}")
    trait = traits[trait_name]
    if field not in trait or trait[field] is None:
        raise KeyError(f"Trait {trait_name!r} has no {field!r} in {path}")
    value = trait[field]
    if not isinstance(value, (str, int, float, bool)):
        raise TypeError(f"Trait field {field!r} for {trait_name!r} must be scalar, got {type(value).__name__}")
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trait", required=True)
    parser.add_argument("--field", required=True)
    parser.add_argument("--traits-config", default="ablation/configs/traits.yaml")
    args = parser.parse_args()
    print(resolve_trait_value(args.traits_config, args.trait, args.field))


if __name__ == "__main__":
    main()
