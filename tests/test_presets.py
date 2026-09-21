"""Presets: vollständig, in den Grenzen, und jede Beispielkarte zeigt, was ihr Hilfetext behauptet (typische Ziehung, Median der 100 festen Karten)."""

import pytest

import om_constants as C
import om_evaluation as ev
import om_presets as P
from om_scenario import build

KEYS = set(P.PRESET_KEYS)


def _a(p):
    return ev.analyse(build(p["net"], p["n"], p["m"], p["reach"], p["ballung"], p["seed"]), p["arr"], p["rule"], p["w"], p["seed"], p["prio"] - 1)


def _dyn(p):
    return ev.analyse_dynamic(build("random", p["n"], p["m"], p["reach"], p["ballung"], p["seed"]), p["dur"], p["seed"], p["rule"], p["prio"] - 1)


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP) and len(C.PRESETS) == 8
    assert all(C.PRESET_HELP[name].strip() for name in C.PRESETS)
    for name, p in C.PRESETS.items():
        assert set(p) == KEYS, name


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_values_are_inside_the_bounds_and_on_the_step_grid(name):
    p = C.PRESETS[name]
    assert p["net"] in C.NETS and p["arr"] in C.ARR_LABELS and p["rule"] in C.RULE_LABELS and p["model"] in C.MODEL_LABELS
    for key, state_key in P.PRESET_KEYS.items():
        spec = P.SETTING_SPECS[state_key]
        if spec.lo is not None:
            assert spec.lo <= p[key] <= spec.hi, (name, key)
    assert (p["reach"] - C.REACH_MIN) % 5 == 0 and p["ballung"] % 25 == 0 and (p["dur"] - C.DUR_MIN) % 5 == 0
    assert not (p["model"] == "dynamic" and p["rule"] == "batch")                               # Batching gibt es nur ohne Zeit


def test_setting_specs_have_room_to_move():
    """Ein Regler mit lo == hi würde Streamlit abstürzen lassen."""
    assert all(spec.lo < spec.hi for spec in P.SETTING_SPECS.values() if spec.lo is not None)


def test_kept_keys_are_all_setting_keys():
    assert set(P.KEPT) <= set(P.SETTING_SPECS) and {"arr_radio", "model_radio", "w_slider", "prio_slider", "dur_slider"} <= set(P.KEPT)


def test_presets_use_seeds_outside_the_distribution_set():
    for name, p in C.PRESETS.items():
        assert p["seed"] not in C.DIST_SEEDS, name


def test_the_default_map_is_the_random_preset_and_shows_a_lost_pair():
    p = C.PRESETS["🗺️ Zufällige Karte"]
    assert (p["n"], p["m"], p["reach"], p["ballung"], p["seed"], p["arr"], p["rule"], p["model"]) == (20, 20, 40, 0, C.DEFAULT_SEED, C.DEFAULT_ARR, C.DEFAULT_RULE, C.DEFAULT_MODEL)
    level, code, d = ev.verdict(_a(p))
    assert (level, code) == ("warning", ev.FEWER_PAIRS) and (d["count"], d["opt_count"]) == (18, 20) and round(d["premium_pct"]) == 21


def test_the_presets_show_what_their_help_says():
    stairs = ev.verdict(_a(C.PRESETS["🪜 Die Treppe"]))[2]
    assert (stairs["count"], stairs["opt_count"]) == (4, 8) and stairs["quotient"] == 0.5
    assert round(float(ev.staircase_exact(8)), 1) == 5.3
    flex = ev.verdict(_a(C.PRESETS["😈 Flexible zuerst"]))[2]
    assert (flex["count"], flex["opt_count"]) == (17, 20)
    fen = ev.verdict(_a(C.PRESETS["🪟 Fenster von 5"]))
    assert fen[2]["count"] == 19 and fen[2]["premium_pct"] < ev.verdict(_a(C.PRESETS["🗺️ Zufällige Karte"]))[2]["premium_pct"]
    wide = ev.verdict(_a(C.PRESETS["🌐 Alles erreichbar"]))[2]
    assert wide["count"] == wide["opt_count"] == 20 and 15 < wide["premium_pct"] < 30
    short = ev.verdict(_a(C.PRESETS["📡 Knappe Reichweite"]))[2]
    assert short["count"] == short["opt_count"] == 8
    more = ev.verdict(_a(C.PRESETS["📐 Mehr Aufträge als Fahrzeuge"]))[2]
    assert more["count"] == more["opt_count"] == 10 and 40 < more["premium_pct"] < 60


def test_the_dynamic_preset_loses_a_few_orders():
    d = ev.verdict_dynamic(_dyn(C.PRESETS["⏱️ Fahrzeuge werden wieder frei"]))[2]
    assert (d["served"], d["opt_served"]) == (26, 28) and d["check"]["all_ok"]


def test_random_presets_are_typical_draws():
    """Die gezeigte Karte liegt nahe dem Median der 100 festen Karten derselben Einstellung (Prämie)."""
    for name in ("🗺️ Zufällige Karte", "😈 Flexible zuerst", "🌐 Alles erreichbar", "📐 Mehr Aufträge als Fahrzeuge"):
        p = C.PRESETS[name]
        d = ev.verdict(_a(p))[2]
        dist = ev.distribution(p["n"], p["m"], p["reach"], p["ballung"], p["arr"], p["w"])
        assert abs(d["premium_pct"] - dist["greedy"]["premium_median"]) <= 0.35 * dist["greedy"]["premium_median"], name
    p = C.PRESETS["⏱️ Fahrzeuge werden wieder frei"]
    q = ev.verdict_dynamic(_dyn(p))[2]["quotient"]
    assert abs(q - ev.dynamic_stats(p["n"], p["m"], p["reach"], p["ballung"], p["dur"])["greedy_q_median"]) <= 0.05


def test_fixed_presets_hide_the_random_controls():
    assert {n for n, p in C.PRESETS.items() if p["net"] in C.FIXED_NETS} == {"🪜 Die Treppe"}
