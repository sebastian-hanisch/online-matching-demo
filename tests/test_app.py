"""Rauchtests der Streamlit-Oberfläche per AppTest: Standard, jedes Preset, alle Kombinationen aus Regel, Ankunft und Modell (auch beim Abspielen auf mehrbildrigen Karten),
Randgrößen, Schritt-Zustand, ausgeblendete Regler, Permalink, Experimente auf Abruf, Schlüssel und Achsensperre."""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import om_constants as C
from om_presets import PRESET_KEYS

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"

# Anfang der Meldung zur gezeigten Karte (Streamlit legt das führende Emoji in `icon`, nicht in `value`)
EXPECTED = {
    "🪜 Die Treppe": "Die Regel findet 4 von 8 möglichen Paaren",
    "🗺️ Zufällige Karte": "Die Regel findet 18 von 20 möglichen Paaren",
    "😈 Flexible zuerst": "Die Regel findet 17 von 20 möglichen Paaren",
    "🪟 Fenster von 5": "Die Regel findet 19 von 20 möglichen Paaren",
    "🌐 Alles erreichbar": "Alle 20 möglichen Paare gefunden, aber 21,1 % teurer",
    "📡 Knappe Reichweite": "Alle 8 möglichen Paare gefunden, aber 1,7 % teurer",
    "📐 Mehr Aufträge als Fahrzeuge": "Alle 10 möglichen Paare gefunden, aber 49,3 % teurer",
    "⏱️ Fahrzeuge werden wieder frei": "Die Regel bedient 26 von 28 Aufträgen",
}
STATIC_COMBOS = [(rule, arr) for rule in C.RULE_LABELS for arr in C.ARR_LABELS]
DYN_RULES = ["greedy", "ranking"]


def _run(setup=None, timeout=600):
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if setup is not None:
        setup(at)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    return at


def _apply(at, p):
    for key, state_key in PRESET_KEYS.items():
        at.session_state[state_key] = p[key]


def _set(**kw):
    def setup(at):
        for k, v in kw.items():
            at.session_state[k] = v
    return setup


def _labels(at):
    return {w.label for w in list(at.sidebar.slider) + list(at.sidebar.selectbox) + list(at.sidebar.number_input) + list(at.sidebar.radio)}


def _texts(at):
    return [e.value for e in list(at.success) + list(at.warning) + list(at.info)]


def _has(at, prefix):
    return any(t.startswith(prefix) for t in _texts(at))


def _step(at):
    found = [s for s in at.slider if s.key == "om_step"]
    return found[0] if found else None


def _keys(at, kind):
    return [w.key for w in getattr(at, kind)]


def _play(at):
    [b for b in at.button if b.label == "▶️ Abspielen"][0].click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_default_renders_without_exception():
    at = _run()
    assert any("Regel und Ankunft" in m.value for m in at.markdown)
    assert _has(at, EXPECTED["🗺️ Zufällige Karte"]) and not at.error


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_renders_with_its_verdict(name):
    at = _run(lambda a: _apply(a, C.PRESETS[name]))
    assert _has(at, EXPECTED[name]), _texts(at)
    if C.PRESETS[name]["net"] in C.FIXED_NETS:
        assert any(t.startswith("Feste Karte") for t in _texts(at))
    else:
        assert any(m.label.startswith(("Gütequotient (Mittel", "Greedy: Anteil")) for m in at.metric)


@pytest.mark.parametrize("rule,arr", STATIC_COMBOS)
def test_every_static_combination_renders_every_kind_of_step(rule, arr):
    at = _run(_set(rule_radio=rule, arr_radio=arr))
    assert not at.error
    last = int(_step(at).max)
    assert last == 20 and _step(at).value == last
    for k in (0, 1, 2, 7, 10, 19, 20):
        _step(at).set_value(k)
        at.run()
        assert not at.exception, (rule, arr, k, [e.value for e in at.exception])


@pytest.mark.parametrize("rule", DYN_RULES)
def test_every_dynamic_combination_renders_every_kind_of_step(rule):
    at = _run(_set(model_radio="dynamic", rule_radio=rule, n_slider=10, m_slider=30))
    last = int(_step(at).max)
    assert last == 30
    for k in (0, 1, 2, 10, 29, 30):
        _step(at).set_value(k)
        at.run()
        assert not at.exception, (rule, k, [e.value for e in at.exception])


@pytest.mark.parametrize("rule", list(C.RULE_LABELS))
def test_play_runs_through_all_frames_without_duplicate_chart_keys(rule):
    """Beim Abspielen entstehen in einem Lauf mehrere Diagramme mit demselben Namen - die Schlüssel tragen deshalb den Schritt (Regression: StreamlitDuplicateElementKey bei mehr als einem Bild)."""
    at = _run(_set(rule_radio=rule))
    assert _step(at).max > 10
    _play(at)


@pytest.mark.parametrize("rule", DYN_RULES)
def test_play_in_the_dynamic_model(rule):
    at = _run(_set(model_radio="dynamic", rule_radio=rule, n_slider=10, m_slider=30))
    assert _step(at).max > 10
    _play(at)


@pytest.mark.parametrize("net", ["stairs", "p4"])
def test_play_on_fixed_maps(net):
    at = _run(_set(net_select=net))
    assert _step(at).max >= 2                                                                    # mindestens drei Bilder mit Diagrammen gleichen Namens
    _play(at)


def test_no_feasible_pair_renders_and_play_is_disabled():
    from om_scenario import generate
    seed = next(k for k in range(500) if not generate(C.N_MIN, C.M_MIN, C.REACH_MIN, 0, k).feasible.any())
    at = _run(_set(n_slider=C.N_MIN, m_slider=C.M_MIN, reach_slider=C.REACH_MIN, seed_input=seed))
    assert _step(at) is not None                                                                 # die Aufträge kommen trotzdem an (und werden abgelehnt)
    assert any(t.startswith("Kein einziges Paar ist möglich") for t in _texts(at))
    at = _run(_set(n_slider=C.N_MIN, m_slider=C.M_MIN, reach_slider=C.REACH_MIN, seed_input=seed, model_radio="dynamic"))
    assert any(t.startswith("Kein Auftrag ist erreichbar") for t in _texts(at))


def test_extreme_sizes_render():
    for n, m, reach in ((C.N_MIN, C.M_MIN, C.REACH_MIN), (C.N_MAX, C.M_MAX, C.REACH_MAX), (C.N_MIN, C.M_MAX, C.REACH_MIN), (C.N_MAX, C.M_MIN, C.REACH_MAX)):
        for model in ("static", "dynamic"):
            at = _run(_set(n_slider=n, m_slider=m, reach_slider=reach, model_radio=model))
            step = _step(at)
            assert step is not None and step.value == step.max


def test_hidden_controls_follow_the_net_and_the_rule():
    def labels_for(**kw):
        return _labels(_run(_set(**kw)))
    random_labels = labels_for()
    assert {"Karte", "Fahrzeuge", "Aufträge", "Reichweite [min]", "Ballung [%]", "Zufalls-Seed"} <= random_labels
    assert labels_for(net_select="stairs") == {"Karte"}                                          # keine toten Regler bei festen Karten
    at = _run()
    assert "w_slider" not in _keys(at, "slider") and "prio_slider" not in _keys(at, "slider") and "dur_slider" not in _keys(at, "slider")
    assert "w_slider" in _keys(_run(_set(rule_radio="batch")), "slider") and "prio_slider" in _keys(_run(_set(rule_radio="ranking")), "slider")
    dyn = _run(_set(model_radio="dynamic"))
    assert "dur_slider" in _keys(dyn, "slider") and "arr_radio" not in _keys(dyn, "radio")
    assert [r.options for r in dyn.radio if r.key == "rule_radio"][0] == [C.RULE_LABELS["greedy"], C.RULE_LABELS["ranking"]]         # kein Batching mit Zeit
    fixed = _run(_set(net_select="stairs"))
    assert "model_radio" not in _keys(fixed, "radio") and "arr_radio" in _keys(fixed, "radio")


def test_the_arrival_radio_and_sliders_keep_their_values_when_hidden_and_shown_again():
    at = _run(_set(arr_radio="flex", rule_radio="batch"))
    at.slider(key="w_slider").set_value(9)
    at.run()
    at.session_state["rule_radio"] = "greedy"
    at.run()
    assert "w_slider" not in _keys(at, "slider")
    at.session_state["rule_radio"] = "batch"
    at.run()
    assert not at.exception and at.slider(key="w_slider").value == 9
    at.session_state["model_radio"] = "dynamic"                                                  # die Ankunft ist im Modell mit Zeit ausgeblendet ...
    at.run()
    assert not at.exception and "arr_radio" not in _keys(at, "radio")
    assert [r.value for r in at.radio if r.key == "rule_radio"][0] == "greedy"                  # ... und Batching fällt auf Greedy zurück
    at.session_state["model_radio"] = "static"
    at.run()
    assert not at.exception and at.radio(key="arr_radio").value == "flex"                        # ... und kommt zurück


def test_hidden_slider_values_come_back_when_the_random_map_is_shown_again():
    at = _run(_set(reach_slider=90))
    at.session_state["net_select"] = "stairs"
    at.run()
    at.session_state["net_select"] = "random"
    at.run()
    assert not at.exception and at.slider(key="reach_slider").value == 90


def test_the_dynamic_model_survives_a_detour_over_a_fixed_map():
    at = _run(_set(model_radio="dynamic"))
    at.session_state["net_select"] = "stairs"
    at.run()
    assert not at.exception and "dur_slider" not in _keys(at, "slider")
    at.session_state["net_select"] = "random"
    at.run()
    assert not at.exception and at.radio(key="model_radio").value == "dynamic"


def test_step_slider_returns_to_the_last_step_when_the_map_or_a_mode_changes():
    at = _run()
    last = int(_step(at).max)
    assert _step(at).value == last
    _step(at).set_value(3)
    at.run()
    assert _step(at).value == 3
    at.session_state["net_select"] = "p4"
    at.run()
    assert not at.exception and _step(at).value == _step(at).max and _step(at).max < last
    at.session_state["net_select"] = "random"
    at.run()
    _step(at).set_value(2)
    at.run()
    at.session_state["rule_radio"] = "ranking"
    at.run()
    assert not at.exception and _step(at).value == _step(at).max


def test_first_step_shows_the_start_a_middle_step_the_options_and_the_last_the_check():
    at = _run()
    _step(at).set_value(0)
    at.run()
    assert not at.exception and any("Anfang: alle 20 Fahrzeuge sind frei" in m.value for m in at.markdown)
    assert not any("Regel eingehalten (neu nachgerechnet)" in t.value.to_dict("list").get("Prüfung", []) for t in at.table)          # die Prüfung erst im letzten Schritt
    _step(at).set_value(5)
    at.run()
    assert not at.exception and any("trifft ein" in m.value and "Freie Fahrzeuge in Reichweite" in m.value for m in at.markdown)
    assert any("Gewählt" in t.value.columns for t in at.table)
    _step(at).set_value(int(_step(at).max))
    at.run()
    check = next(t.value.to_dict("list") for t in at.table if "Regel eingehalten (neu nachgerechnet)" in t.value.to_dict("list").get("Prüfung", []))
    assert all(row == "✅" for row in check["Ergebnis"]) and len(check["Prüfung"]) == 6


def test_batching_shows_waiting_and_flush_steps():
    at = _run(_set(rule_radio="batch"))
    seen = set()
    for k in range(1, 11):
        _step(at).set_value(k)
        at.run()
        assert not at.exception
        seen |= {"wait" for m in at.markdown if "wartet im Stapel" in m.value or "Er wartet im Stapel" in m.value} | {"flush" for m in at.markdown if "Das Fenster ist voll" in m.value}
    assert seen == {"wait", "flush"}


def test_the_stairs_preset_shows_half_the_pairs_at_the_last_step():
    at = _run(lambda a: _apply(a, C.PRESETS["🪜 Die Treppe"]))
    assert any(m.label == "Paare (Regel)" and m.value == "4 von 8" for m in at.metric)


def test_permalink_parameters_are_clamped_and_snapped():
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.query_params["reach"] = "9999"
    at.query_params["ballung"] = "abc"
    at.query_params["n"] = "-5"
    at.query_params["dur"] = "22"
    at.query_params["model"] = "dynamic"
    at.run()
    assert not at.exception
    assert at.slider(key="reach_slider").value == C.REACH_MAX and at.slider(key="ballung_slider").value == C.DEFAULT_BALLUNG and at.slider(key="n_slider").value == C.N_MIN
    assert at.slider(key="dur_slider").value == 20                                               # auf die Regler-Schritte gerundet
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.query_params["reach"] = "42"
    at.query_params["ballung"] = "60"
    at.run()
    assert at.slider(key="reach_slider").value == 40 and at.slider(key="ballung_slider").value == 50


def test_permalink_keeps_valid_modes_and_falls_back_for_unknown_ones():
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.query_params["rule"] = "ranking"
    at.query_params["arr"] = "sweep"
    at.query_params["prio"] = "7"
    at.run()
    assert not at.exception and at.radio(key="rule_radio").value == "ranking" and at.radio(key="arr_radio").value == "sweep" and at.slider(key="prio_slider").value == 7
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.query_params["net"] = "ring"
    at.query_params["rule"] = "chaos"
    at.query_params["arr"] = "X"
    at.query_params["model"] = "warp"
    at.run()
    assert not at.exception and at.selectbox(key="net_select").value == C.DEFAULT_NET
    assert at.radio(key="rule_radio").value == C.DEFAULT_RULE and at.radio(key="arr_radio").value == C.DEFAULT_ARR and at.radio(key="model_radio").value == C.DEFAULT_MODEL


def test_a_batching_permalink_in_the_dynamic_model_falls_back_to_greedy():
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.query_params["rule"] = "batch"
    at.query_params["model"] = "dynamic"
    at.run()
    assert not at.exception and at.radio(key="rule_radio").value == "greedy"


def test_randomize_moves_the_seed_but_not_the_distribution():
    at = _run()
    before = {m.label: m.value for m in at.metric}
    seed_before = at.number_input(key="seed_input").value
    [b for b in at.sidebar.button if "Neue Karte" in b.label][0].click()
    at.run()
    assert not at.exception and at.number_input(key="seed_input").value != seed_before
    after = {m.label: m.value for m in at.metric}
    for label in ("Gütequotient (Mittel | Median)", "Prämie (Mittel | Median)", "Optimum im Mittel", "Greedy schlägt Ranking"):
        assert before[label] == after[label], label


def test_experiments_run_on_demand():
    at = _run()
    markers = ("Im statischen Modell kostet Warten nichts", "Auf der Treppe erreicht Auftrag t genau", "Mittlerer Gütequotient der Paare über 40 feste Karten", "Ein größeres Fenster kostet mehr Aufwand")
    assert not any(any(m in c.value for m in markers) for c in at.caption)
    for key in ("win_start", "stairs_start", "arr_start", "scan_start"):
        at.button(key=key).click()
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    text = " ".join(c.value for c in at.caption)
    assert all(m in text for m in markers)


def test_dynamic_experiment_and_fixed_map_experiments():
    at = _run(_set(model_radio="dynamic", n_slider=10, m_slider=30))
    assert "win_start" not in _keys(at, "button") and "dyn_start" in _keys(at, "button")
    at.button(key="dyn_start").click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Anteil der im Nachhinein bedienbaren Aufträge" in c.value for c in at.caption)
    fixed = _run(_set(net_select="stairs"))
    assert _keys(fixed, "button").count("stairs_start") == 1 and "win_start" not in _keys(fixed, "button")
    fixed.button(key="stairs_start").click()
    fixed.run()
    assert not fixed.exception


def _calls(src, name):
    """Der Text jedes Aufrufs `name(...)` einschließlich verschachtelter Klammern."""
    out = []
    for m in re.finditer(re.escape(name) + r"\(", src):
        depth, i = 1, m.end()
        while depth:
            depth += {"(": 1, ")": -1}.get(src[i], 0)
            i += 1
        out.append(src[m.start():i])
    return out


def test_every_plotly_chart_has_an_explicit_key_and_the_play_loop_keys_carry_the_step():
    calls = _calls(APP.read_text(encoding="utf-8"), "plotly_chart")
    assert len(calls) == 10 and all(re.search(r'key=f?"[a-z_]+(_\{\w+\})?"', c) for c in calls), calls
    keys = [re.search(r'key=f?"([a-z_]+?)(?:_\{\w+\})?"', c).group(1) for c in calls]
    stepped = [c for c in calls if 'key=f"' in c]
    assert len(stepped) == 3 and {re.search(r'key=f"([a-z_]+?)_\{', c).group(1) for c in stepped} == {"om_map", "om_progress"}    # die Karte (statisch und mit Zeit) und der Verlauf tragen den Schritt
    unstepped = [k for c, k in zip(calls, keys) if 'key=f"' not in c]
    assert len(set(unstepped)) == len(unstepped) == 7, unstepped                                  # jeder feste Schlüssel nur einmal
    viz = (ROOT / "om_visualization.py").read_text(encoding="utf-8")
    bodies = [b for b in viz.split(chr(10) + "def ") if b.startswith("build_")]
    assert "fixedrange=True" in viz and len(bodies) == 9 and all("_base(" in b or "_map_layout(" in b for b in bodies)


def test_maps_have_no_legend_and_keep_equal_scale_inside_the_domain():
    import om_dynamic as Dy
    import om_scenario as S
    import om_visualization as V
    from om_online import run_online
    sc = S.generate(20, 20, 40, 0, 11)
    figs = [V.build_online_map(sc, list(range(20)), run_online(sc, list(range(20))), k) for k in (0, 7, 20)]
    dsc = S.generate(10, 30, 40, 0, 18)
    figs.append(V.build_dynamic_map(dsc, Dy.run_dynamic(dsc, S.times(30, 120, 18), 20), 12))
    for fig in figs:
        assert fig.layout.showlegend is False and fig.layout.xaxis.constrain == "domain" and fig.layout.xaxis.fixedrange and fig.layout.yaxis.fixedrange


def test_app_text_has_no_links_to_repository_files():
    assert not re.search(r"\]\(\w+\.py\)", APP.read_text(encoding="utf-8"))


def test_footer_is_verbatim():
    src = APP.read_text(encoding="utf-8")
    assert "https://sebastianhanisch.net/kontakt.html" in src and "Interesse an einer maßgeschneiderten Lösung für" in src and "Operations Research und Machine Learning" in src


def test_runtime_needs_only_numpy_pandas_plotly_streamlit():
    """Konvention der Konzepte-Wurzeln und -Stücke: Referenzbibliotheken (scipy, networkx) nur als Testorakel."""
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "scipy" not in req and "networkx" not in req
    for path in ROOT.glob("*.py"):
        assert not re.search(r"^\s*(import|from)\s+(scipy|networkx)\b", path.read_text(encoding="utf-8"), re.M), path.name
