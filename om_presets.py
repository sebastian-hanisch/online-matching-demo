"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster des Portfolios, siehe gm_presets.py in greedy-matching-demo)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import om_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


def _choice(options):
    def cast(value):
        value = str(value)
        if value not in options:
            raise ValueError(value)
        return value
    return cast


SETTING_SPECS = {
    "net_select": SettingSpec("net", _choice(C.NETS), C.DEFAULT_NET),
    "arr_radio": SettingSpec("arr", _choice(C.ARR_LABELS), C.DEFAULT_ARR),
    "rule_radio": SettingSpec("rule", _choice(C.RULE_LABELS), C.DEFAULT_RULE),
    "w_slider": SettingSpec("w", int, C.DEFAULT_W, C.W_MIN, C.W_MAX),
    "prio_slider": SettingSpec("prio", int, C.DEFAULT_PRIO, C.PRIO_MIN, C.PRIO_MAX),
    "model_radio": SettingSpec("model", _choice(C.MODEL_LABELS), C.DEFAULT_MODEL),
    "dur_slider": SettingSpec("dur", int, C.DEFAULT_DUR, C.DUR_MIN, C.DUR_MAX),
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX),
    "m_slider": SettingSpec("m", int, C.DEFAULT_M, C.M_MIN, C.M_MAX),
    "reach_slider": SettingSpec("reach", int, C.DEFAULT_REACH, C.REACH_MIN, C.REACH_MAX),
    "ballung_slider": SettingSpec("ballung", int, C.DEFAULT_BALLUNG, C.BALLUNG_MIN, C.BALLUNG_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
}
PRESET_KEYS = {"net": "net_select", "arr": "arr_radio", "rule": "rule_radio", "w": "w_slider", "prio": "prio_slider", "model": "model_radio", "dur": "dur_slider", "n": "n_slider", "m": "m_slider",
               "reach": "reach_slider", "ballung": "ballung_slider", "seed": "seed_input"}
# Regler, die nicht immer gezeichnet werden (feste Karten, Regel ohne Fenster, statisches Modell ...): Streamlit löscht ihren Zustand, sobald sie nicht gezeichnet werden -
# der zuletzt gewählte Wert bleibt hier erhalten
KEPT = {key: f"_kept_{key}" for key in ("n_slider", "m_slider", "reach_slider", "ballung_slider", "seed_input", "w_slider", "prio_slider", "dur_slider", "arr_radio", "model_radio")}
STEPS = {"reach_slider": 5, "ballung_slider": 25, "dur_slider": 5}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = st.session_state.get(KEPT[state_key], spec.default) if state_key in KEPT else spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
                st.session_state[state_key] = value
                if state_key in KEPT:
                    st.session_state[KEPT[state_key]] = value
            except (ValueError, TypeError):
                pass
    for key, step in STEPS.items():
        if key in st.session_state:
            lo = SETTING_SPECS[key].lo
            st.session_state[key] = int(lo + round((st.session_state[key] - lo) / step) * step)
            if key in KEPT:
                st.session_state[KEPT[key]] = st.session_state[key]
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = C.PRESETS[name][key]
        if state_key in KEPT:
            st.session_state[KEPT[state_key]] = C.PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
