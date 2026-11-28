"""Split design (SPEC.md §5): determinism, registered hashes, template hold-out, sizes and disjointness.
The held-out split is generated and hashed here; no policy is run on it."""
from __future__ import annotations

import json

from actionproof.attacks import spec as S
from actionproof.attacks.generate import MANIFEST, check_disjoint, load_split, make_split, split_hash

MAN = json.loads(MANIFEST.read_text())
DEV = load_split("attack_dev")
HELD = make_split("attack_heldout")


def test_hashes_match_the_manifest():
    assert split_hash(DEV) == MAN["splits"]["attack_dev"]["sha256"]
    assert split_hash(HELD) == MAN["splits"]["attack_heldout"]["sha256"]
    assert split_hash(make_split("attack_dev")) == MAN["splits"]["attack_dev"]["sha256"]


def test_sizes():
    att = [t for t in HELD if t.meta["attack"]["variant"] == "attack"]
    assert len(att) >= 1000
    assert sum(t.meta["attack"]["family"] == "F9_compound" for t in att) == S.SIZES["attack_heldout"]["compound"]
    for f in S.FAMILIES:
        assert sum(t.meta["attack"]["family"] == f for t in att) == S.SIZES["attack_heldout"]["per_family"]
        assert sum(t.meta["attack"]["family"] == f and t.meta["attack"]["variant"] == "attack" for t in DEV) == \
            S.SIZES["attack_dev"]["per_family"]
    for split in (DEV, HELD):
        pairs = {t.meta["attack"]["pair_id"] for t in split if t.meta["attack"]["variant"] == "attack"}
        twins = {t.meta["attack"]["pair_id"] for t in split if t.meta["attack"]["variant"] == "twin"}
        assert pairs == twins


def test_template_holdout():
    dev_t = {c["template_id"] for t in DEV for c in t.meta["attack"]["components"]}
    held_t = {c["template_id"] for t in HELD for c in t.meta["attack"]["components"]}
    assert dev_t and held_t and not dev_t & held_t
    assert all(not S.is_heldout_template(c["family"], c["channel"], c["phrasing"])
               for t in DEV for c in t.meta["attack"]["components"])
    assert all(S.is_heldout_template(c["family"], c["channel"], c["phrasing"])
               for t in HELD for c in t.meta["attack"]["components"])
    assert not any(t.meta["attack"]["family"] in S.HELDOUT_ONLY_FAMILIES for t in DEV)
    # the training generator uses development templates only
    train = make_split("attack_train", per_family=9, compound=0)
    assert {c["template_id"] for t in train for c in t.meta["attack"]["components"]} <= dev_t


def test_every_heldout_template_is_used():
    used = {c["template_id"] for t in HELD for c in t.meta["attack"]["components"] if t.meta["attack"]["family"] !=
            "F9_compound"}
    for f in S.FAMILIES:
        assert {S.template_id(f, c, p) for c, p in S.heldout_templates(f)} <= used, f


def test_po_ids_disjoint_from_every_existing_split():
    out = check_disjoint({"dev": DEV, "held": HELD, "train": make_split("attack_train", per_family=9, compound=0)})
    assert out["existing_po_ids_checked"] > 1000
