"""The config hash is the audit trail. If it is not stable, nothing downstream is."""

from __future__ import annotations

import pytest

from bayline import config as config_module
from bayline.config import Config


def test_hash_is_insensitive_to_key_order(cfg):
    """Reordering YAML must not change the hash, or every reformat looks like a price change."""
    reversed_tree = {k: cfg.tree[k] for k in reversed(list(cfg.tree))}
    assert Config(reversed_tree, cfg.source).hash == cfg.hash


def test_hash_changes_when_a_price_changes(cfg):
    assert cfg.with_overrides(**{"economics.recovery_tow_eur": 781}).hash != cfg.hash


def test_overrides_do_not_mutate_the_original(cfg):
    before = cfg.get("fleet.vehicles")
    cfg.with_overrides(**{"fleet.vehicles": 7})
    assert cfg.get("fleet.vehicles") == before


def test_shares_must_sum_to_one(cfg):
    tree = cfg.tree
    tree["fleet"]["classes"][0]["share"] += 0.05
    with pytest.raises(ValueError, match="shares sum to"):
        config_module._validate(tree)


def test_planned_must_be_cheaper_than_unplanned(cfg):
    """If planned work costs as much as a breakdown there is no case for scheduling it.

    A typo here would not crash anything — it would quietly produce a plan that recommends
    doing nothing, and the product would look broken rather than misconfigured.
    """
    tree = cfg.tree
    tree["components"][0]["planned_eur"] = tree["components"][0]["repair_eur"]
    with pytest.raises(ValueError, match="planned cost is not below"):
        config_module._validate(tree)


def test_duplicate_component_ids_rejected(cfg):
    tree = cfg.tree
    tree["components"].append(dict(tree["components"][0]))
    with pytest.raises(ValueError, match="duplicate component"):
        config_module._validate(tree)


def test_every_component_is_typed_and_priced(cfg):
    for component in cfg.components:
        assert component.planned_eur < component.repair_eur
        assert component.bay_hours > 0
        assert component.weibull_shape > 1.0, (
            f"{component.id}: shape <= 1 is infant mortality or a constant hazard, neither of "
            "which is a wear-out process and neither of which is predictable from a trend"
        )
        assert component.deferrable is not component.safety_critical
