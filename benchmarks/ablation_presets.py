from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class AblationPreset:
    name: str
    description: str
    methods: Tuple[str, ...]


def preset_removal_all() -> AblationPreset:
    """
    Full + leave-one-component-out ablations (IWOA side), plus WOA baseline.
    """
    variants = (
        "full",
        "no_init",
        "no_qr",
        "no_lpsr",
        "no_restart",
        "no_crossover",
        "no_levy",
        "no_chaos",
        "no_nelder_mead",
    )
    methods: List[str] = ["woa"]
    for v in variants:
        if v == "full":
            methods.append("iwoa")
        else:
            methods.append(f"iwoa:{v}")
    return AblationPreset(
        name="ablation_removal",
        description="IWOA full + leave-one-out variants, with WOA baseline.",
        methods=tuple(methods),
    )


def preset_addition_ladder() -> AblationPreset:
    """
    Addition ladder: WOA baseline + WOAPlus variants that add IWOA components progressively,
    and IWOA full as a reference point.

    Note: WOAPlus is implemented as WOA dynamics + optional added operators; it is not identical to IWOA,
    but answers the question \"which operators added to WOA yield gains\".
    """
    ladder = (
        "woa",
        "woa_plus:init",
        "woa_plus:init_qr",
        "woa_plus:init_qr_crossover",
        "woa_plus:init_qr_crossover_restart",
        "woa_plus:init_qr_crossover_restart_perturb",
        "woa_plus:init_qr_crossover_restart_perturb_chaos",
        "woa_plus:init_qr_crossover_restart_perturb_chaos_nm",
        "woa_plus:all",
        "iwoa",
    )
    return AblationPreset(
        name="ablation_addition",
        description="WOA -> add operators progressively (WOAPlus ladder) -> IWOA full reference.",
        methods=tuple(ladder),
    )


def preset_interactions_core() -> AblationPreset:
    """
    Factorial interaction suite for {restart, crossover, nelder_mead}.
    Encoded as IWOA variants: r{0|1}_c{0|1}_nm{0|1} with other components enabled.
    """
    methods: List[str] = ["woa"]
    for r in (0, 1):
        for c in (0, 1):
            for nm in (0, 1):
                v = f"r{r}_c{c}_nm{nm}"
                if v == "r1_c1_nm1":
                    methods.append("iwoa")  # canonical full name
                else:
                    methods.append(f"iwoa:{v}")
    return AblationPreset(
        name="ablation_interactions",
        description="Factorial interactions for restart/crossover/nelder_mead (others on), plus WOA baseline.",
        methods=tuple(methods),
    )


def get_preset(name: str) -> AblationPreset:
    key = name.strip().lower()
    if key in {"ablation_removal", "removal", "removal_all"}:
        return preset_removal_all()
    if key in {"ablation_addition", "addition", "ladder"}:
        return preset_addition_ladder()
    if key in {"ablation_interactions", "interactions", "interaction", "core_interactions"}:
        return preset_interactions_core()
    raise ValueError(f"Unknown preset: {name}")


