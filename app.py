"""Online-Matching - entscheiden, bevor man alles weiß - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Konzept - Zuordnen ohne Kenntnis der Zukunft - und lässt stattdessen das Beispiel wachsen.
Siebtes Stück der Matching-Linie der "Konzepte"-Reihe, unabhängiger Ast neben der Kostenlinie: der Kontrast zum Offline-Optimum der Ungarischen Methode. Siehe README.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import om_constants as C
import om_dynamic as Dy
import om_evaluation as ev
from om_online import state_at
from om_presets import (
    KEPT,
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from om_scenario import build, priority
from om_visualization import (
    build_arrival_bars,
    build_dynamic_map,
    build_dynamic_sweep,
    build_hist,
    build_online_map,
    build_progress,
    build_scan_bars,
    build_stair_curve,
    build_window_sweep,
)

st.set_page_config(page_title="Online-Matching – Sebastian Hanisch", layout="wide")


def _pct(x, digits=0):
    return "–" if x is None else f"{x:.{digits}f} %".replace(".", ",")


def _share(x):
    return f"{100 * x:.0f} %"


def _f(x, digits=2):
    return "–" if x is None else f"{x:.{digits}f}".replace(".", ",")


def _int(x):
    return f"{x:,.0f}".replace(",", " ")


@st.cache_resource(show_spinner=False, max_entries=16)
def _analysis(params):
    net, n, m, reach, ballung, seed, arr, rule, w, prio_k = params
    return ev.analyse(build(net, n, m, reach, ballung, seed), arr, rule, w, seed, prio_k)


@st.cache_resource(show_spinner=False, max_entries=16)
def _analysis_dyn(params):
    n, m, reach, ballung, seed, dur, rule, prio_k = params
    return ev.analyse_dynamic(build("random", n, m, reach, ballung, seed), dur, seed, rule, prio_k)


@st.cache_data(show_spinner=False, max_entries=8)
def _series(params):
    return ev.prefix_series(_analysis(params))


@st.cache_data(show_spinner=False, max_entries=8)
def _series_dyn(params):
    a = _analysis_dyn(params)
    online, best, cur = [0], [0], 0
    for k, e in enumerate(a.online.events, start=1):
        cur += e.kind == "assign"
        online.append(cur)
        best.append(Dy.offline_dynamic(a.scenario, a.times, a.dur, k)["served"])
    return online, best


@st.cache_data(show_spinner=False)
def _distribution(n, m, reach, ballung, arr, w):
    return ev.distribution(n, m, reach, ballung, arr, w)


@st.cache_data(show_spinner=False)
def _dyn_stats(n, m, reach, ballung, dur):
    return ev.dynamic_stats(n, m, reach, ballung, dur)


@st.cache_data(show_spinner=False)
def _window_sweep(n, m, reach, ballung, arr):
    return ev.window_sweep(n, m, reach, ballung, arr)


@st.cache_data(show_spinner=False)
def _stairs():
    return ev.staircase_curve(), ev.staircase_arrivals(20)


@st.cache_data(show_spinner=False)
def _arrivals(n, m, reach, ballung):
    return ev.arrival_comparison(n, m, reach, ballung)


@st.cache_data(show_spinner=False)
def _dyn_sweep(n, m, reach, ballung):
    return ev.dynamic_sweep(n, m, reach, ballung)


@st.cache_data(show_spinner=False)
def _scans(n, m, reach, ballung, arr):
    return ev.scan_table(n, m, reach, ballung, arr)


def _kept_widget(key, shown):
    """Ein Regler oder Umschalter, der nicht immer gezeichnet wird: kommt er wieder, kehrt sein zuletzt gewählter Wert zurück."""
    flag = f"_{key}_shown"
    if shown and not st.session_state.get(flag) and KEPT[key] in st.session_state:
        st.session_state[key] = st.session_state[KEPT[key]]
    st.session_state[flag] = shown


st.title("📬 Online-Matching – entscheiden, bevor man alles weiß")
st.markdown(
    """
Die Ungarische Methode kennt **alle** Aufträge vorab und findet die beste Zuordnung. In der Praxis **kommen Aufträge nacheinander**, und jede Zusage ist **sofort und unwiderruflich**: ein Fahrzeug wird zugeteilt, bevor der nächste Auftrag bekannt ist.
Das kostet etwas, und die Demo misst, wie viel. Die einfachste Regel, **Greedy** (das nächste freie Fahrzeug), findet mindestens **die Hälfte** der möglichen Paare - und auf der **Treppe** genau die Hälfte, weil sie immer das Fahrzeug verbraucht, das später am dringendsten gebraucht wird.
**Ranking** zieht dazu eine zufällige feste Rangfolge der Fahrzeuge und erreicht im Erwartungswert **1 − 1/e ≈ 63 %**, gegen jede Reihenfolge. **Batching** sammelt ein paar Aufträge und löst diesen Stapel optimal. Und wenn Fahrzeuge nach der Fahrt **wieder frei werden**, kommt eine Zeitdimension dazu.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - siebtes Stück der Matching-Linie der \"Konzepte\"-Reihe, unabhängiger Ast neben der Kostenlinie - **ein** Konzept an einem wachsenden Beispiel. "
    "Das Ziel ist lexikografisch wie in den Vorgängerdemos: **erst möglichst viele Paare, dann geringe Kosten**. Die Güte steht deshalb in zwei Zahlen: dem **Gütequotient der Paare** (Online-Paare geteilt durch die Paare des Optimums) und der **Prämie** der Kosten gegenüber der billigsten Paarung mit derselben Paarzahl. "
    "Die Referenz \"Offline-Optimum\" ist die Ungarische Methode aus der Demo dazu. Nächste Nachbarn der Matching-Linie, noch nicht gebaut: Blossom, Gewichteter Blossom und Stabile Mitbewohner."
)

with st.expander("So funktioniert Online-Matching", expanded=True):
    st.markdown(
        """
1. **Die Fahrzeuge sind bekannt, die Aufträge kommen nacheinander.** Beim Eintreffen sieht die Regel nur die Zeile des neuen Auftrags: welche freien Fahrzeuge ihn erreichen und wie lange sie brauchen.
2. **Sofort und unwiderruflich:** der Auftrag bekommt ein Fahrzeug oder wird abgelehnt. Nichts wird später umgehängt (mit Umhängen wäre es das Offline-Optimum).
3. **Greedy** nimmt das billigste freie Fahrzeug in Reichweite, **Ranking** das am besten platzierte einer zufällig gezogenen Rangfolge (kostenblind), **Batching** sammelt ein Fenster von Aufträgen und ordnet es den freien Fahrzeugen optimal zu (Ungarische Methode).
4. **Die Reihenfolge entscheidet:** zufällig, gegnerisch (flexible Aufträge zuerst: sie nehmen den starren die Fahrzeuge weg), starre zuerst oder von links nach rechts über die Karte.
5. **Mit Zeit:** ein zugeordnetes Fahrzeug ist eine feste Belegungszeit D belegt und steht danach am Ort des Auftrags wieder bereit. Der Maßstab ist dann das im Nachhinein beste Ergebnis (Min-Cost-Flow).
        """
    )

st.caption("🎯 Schnellstart – eine Beispielkarte laden:")
names = list(C.PRESETS.keys())
for row in range(0, len(names), 4):
    preset_cols = st.columns(4)
    for col, name in zip(preset_cols, names[row:row + 4]):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name] or None)

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    net_key = st.selectbox(
        "Karte", list(C.NETS), key="net_select", format_func=lambda k: C.NETS[k],
        help="Eine zufällige Karte mit Fahrzeugen und Aufträgen oder eine feste Lehrbuchkarte: die Treppe (Greedy findet nur die Hälfte) und ein Pfad aus vier Punkten (der Fall im Kleinen).",
    )
    if net_key == "random":
        n = st.slider("Fahrzeuge", *bounds("n_slider"), key="n_slider", help="Anzahl der Fahrzeuge (alle vorab bekannt).")
        st.session_state[KEPT["n_slider"]] = n
        m = st.slider("Aufträge", *bounds("m_slider"), key="m_slider", help="Anzahl der Aufträge, die nacheinander eintreffen.")
        st.session_state[KEPT["m_slider"]] = m
        reach = st.slider("Reichweite [min]", *bounds("reach_slider"), key="reach_slider", step=5,
                          help="Wie weit ein Paar höchstens auseinander liegen darf. Je kleiner, desto weniger Alternativen hat ein Auftrag - und desto weniger kann eine falsche Wahl anrichten (bei 10 und bei 150 ist Greedy fast oder ganz optimal bei den Paaren, bei 40 verliert er).")
        st.session_state[KEPT["reach_slider"]] = reach
        ballung = st.slider("Ballung [%]", *bounds("ballung_slider"), key="ballung_slider", step=25, help="0 = Fahrzeuge und Aufträge gleichmäßig verteilt, 100 = alle um drei Stadtteile gruppiert.")
        st.session_state[KEPT["ballung_slider"]] = ballung
        seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
        st.session_state[KEPT["seed_input"]] = seed
        st.button("🎲 Neue Karte generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Zufalls-Seed. Die Verteilung über 100 feste Karten weiter unten ändert sich dabei nicht.")
    else:
        n = int(st.session_state.get(KEPT["n_slider"], C.DEFAULT_N))
        m = int(st.session_state.get(KEPT["m_slider"], C.DEFAULT_M))
        reach = int(st.session_state.get(KEPT["reach_slider"], C.DEFAULT_REACH))
        ballung = int(st.session_state.get(KEPT["ballung_slider"], C.DEFAULT_BALLUNG))
        seed = int(st.session_state.get(KEPT["seed_input"], C.DEFAULT_SEED))
        st.caption("Diese Karte ist fest - es gibt nichts zu erzeugen. Zahl der Fahrzeuge und Aufträge, Reichweite, Ballung und Seed gehören zur zufälligen Karte.")

fixed = net_key in C.FIXED_NETS

# --- Regel und Ankunft --------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Regel und Ankunft")
step_col, model_col, rule_col, play_col = st.columns([4, 3, 4, 2])
with model_col:
    if fixed:
        model = "static"
        st.caption("Diese feste Karte gehört zum Modell „Ein Fahrzeug fährt einmal“.")
        _kept_widget("model_radio", False)
    else:
        _kept_widget("model_radio", True)
        model = st.radio("Modell", list(C.MODEL_LABELS), key="model_radio", format_func=lambda k: C.MODEL_LABELS[k],
                         help="Statisch: jedes Fahrzeug fährt höchstens einen Auftrag. Mit Zeit: nach D Minuten Belegung ist es wieder frei und steht am Ort des letzten Auftrags.")
        st.session_state[KEPT["model_radio"]] = model
dynamic = model == "dynamic"
with rule_col:
    rule_options = ["greedy", "ranking"] if dynamic else list(C.RULE_LABELS)
    if st.session_state.get("rule_radio") not in rule_options:
        st.session_state["rule_radio"] = C.DEFAULT_RULE
    rule = st.radio("Regel", rule_options, key="rule_radio", format_func=lambda k: C.RULE_LABELS[k],
                    help="Greedy: das billigste freie Fahrzeug. Ranking: das am besten platzierte einer zufälligen festen Rangfolge (kostenblind). Batching: Stapel von Aufträgen optimal lösen - im Modell mit Zeit nicht vorgesehen, weil eine verzögerte Zuordnung den Maßstab bricht.")
    if dynamic:
        st.caption("Batching gibt es nur ohne Zeit: eine verzögerte Zuordnung kann einen Auftrag noch bedienen, den der Sofort-Maßstab nicht kennt.")
ctl1, ctl2 = st.columns(2)
with ctl1:
    if dynamic:
        arr = st.session_state.get(KEPT["arr_radio"], C.DEFAULT_ARR)
        _kept_widget("arr_radio", False)
        st.caption("Die Aufträge treffen zu zufälligen Minuten zwischen 0 und 120 ein.")
    else:
        _kept_widget("arr_radio", True)
        arr = st.radio("Ankunft der Aufträge", list(C.ARR_LABELS), key="arr_radio", format_func=lambda k: C.ARR_LABELS[k],
                       help="Zufällig: eine gemischte Reihenfolge. Flexible zuerst: Aufträge mit vielen erreichbaren Fahrzeugen kommen vorn (gegnerisch). Starre zuerst: umgekehrt. Von links nach rechts: der Ort bestimmt die Reihenfolge.")
        st.session_state[KEPT["arr_radio"]] = arr
with ctl2:
    w = int(st.session_state.get(KEPT["w_slider"], C.DEFAULT_W))
    prio = int(st.session_state.get(KEPT["prio_slider"], C.DEFAULT_PRIO))
    dur = int(st.session_state.get(KEPT["dur_slider"], C.DEFAULT_DUR))
    _kept_widget("w_slider", rule == "batch" and not dynamic)
    _kept_widget("prio_slider", rule == "ranking")
    _kept_widget("dur_slider", dynamic)
    if rule == "batch" and not dynamic:
        w = st.slider("Fenster w", *bounds("w_slider"), key="w_slider", help="Wie viele Aufträge ein Stapel sammelt, bevor die Ungarische Methode ihn den freien Fahrzeugen zuordnet. Ein Fenster von 1 ist Greedy, ein Fenster ab m Aufträgen ist das Offline-Optimum.")
        st.session_state[KEPT["w_slider"]] = w
    if rule == "ranking":
        prio = st.slider("Rangfolge Nr.", *bounds("prio_slider"), key="prio_slider", help="Welche der 20 gezogenen zufälligen Rangfolgen der Fahrzeuge verwendet wird. Der Erwartungswert über alle 20 steht in der Verteilung unten.")
        st.session_state[KEPT["prio_slider"]] = prio
    if dynamic:
        dur = st.slider("Belegungszeit D [min]", *bounds("dur_slider"), key="dur_slider", step=5,
                        help="So lange ist ein Fahrzeug nach der Zuordnung belegt (die Fahrzeit ist nur Kosten). Bei 120 ist es praktisch das statische Modell.")
        st.session_state[KEPT["dur_slider"]] = dur

prio_k = int(prio) - 1
seed_eff = C.DEFAULT_SEED if fixed else int(seed)
if fixed:
    params = (net_key, C.DEFAULT_N, C.DEFAULT_M, C.DEFAULT_REACH, C.DEFAULT_BALLUNG, seed_eff, arr, rule, int(w), prio_k)
else:
    params = (net_key, int(n), int(m), int(reach), int(ballung), seed_eff, arr, rule, int(w), prio_k)
dparams = (int(n), int(m), int(reach), int(ballung), seed_eff, int(dur), rule, prio_k)
with st.spinner("Rechne..."):
    if dynamic:
        a = _analysis_dyn(dparams)
        sc, res = a.scenario, a.online
        level, code, d = ev.verdict_dynamic(a)
    else:
        a = _analysis(params)
        sc, res = a.scenario, a.online
        level, code, d = ev.verdict(a)
n_events = len(res.events)
owner = ("dyn", dparams) if dynamic else ("sta", params)
if st.session_state.get("om_step_owner") != owner:
    st.session_state["om_step"] = n_events
    st.session_state["om_step_owner"] = owner
with step_col:
    if n_events > 0:
        step = st.slider("Ankunft", 0, n_events, key="om_step", help="Wie viele Aufträge schon eingetroffen sind. Ganz rechts das Endergebnis.")
    else:
        step = 0
        st.caption("Hier gibt es keine Ankunft: keine Aufträge.")
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch", disabled=n_events == 0)
sync_query_params({"net_select": net_key, "arr_radio": arr, "rule_radio": rule, "w_slider": int(w), "prio_slider": int(prio), "model_radio": model, "dur_slider": int(dur),
                   "n_slider": int(n), "m_slider": int(m), "reach_slider": int(reach), "ballung_slider": int(ballung), "seed_input": int(seed)})
view_slot = st.empty()
online_series, best_series = _series_dyn(dparams) if dynamic else _series(params)


def _fs(i, c):
    return f"F{i + 1} ({c} min)"


def _event_text(k):
    if k == 0:
        return f"Anfang: alle {sc.n} Fahrzeuge sind frei, noch ist kein Auftrag eingetroffen."
    e = res.events[k - 1]
    j = e.order
    if dynamic:
        head = f"Minute {e.t}: A{j + 1} trifft ein ({e.n_busy} von {sc.n} Fahrzeugen sind belegt). "
        if e.kind == "reject":
            return head + "Kein freies Fahrzeug in Reichweite: der Auftrag wird abgelehnt."
        opts = ", ".join(_fs(i, c) for i, c in e.options)
        pick = "das billigste" if a.rule == "greedy" else f"das bestplatzierte (Rang {res.prio[e.vehicle] + 1} von {sc.n})"
        return head + f"Frei und in Reichweite: {opts}. Die Regel nimmt {pick}: F{e.vehicle + 1} fährt {e.cost} min und ist dann {a.dur} Minuten belegt."
    head = f"A{j + 1} trifft ein. "
    if e.kind == "reject":
        return head + "Kein freies Fahrzeug in Reichweite: der Auftrag wird endgültig abgelehnt."
    if e.kind == "assign":
        opts = ", ".join(_fs(i, c) for i, c in e.options)
        pick = "das billigste" if res.rule == "greedy" else f"das bestplatzierte (Rang {res.prio[e.chosen] + 1} von {sc.n})"
        return head + f"Freie Fahrzeuge in Reichweite: {opts}. Die Regel nimmt {pick}: F{e.chosen + 1}."
    if e.kind == "wait":
        return head + f"Er wartet im Stapel ({len(e.waiting)} von {res.w}); entschieden wird erst, wenn das Fenster voll ist."
    used_before = len(state_at(res, k - 1)[0])
    return head + f"Das Fenster ist voll ({len(e.assigned) + len(e.rejected)} Aufträge): die Ungarische Methode ordnet den Stapel den {sc.n - used_before} freien Fahrzeugen optimal zu - {len(e.assigned)} zugeordnet, {len(e.rejected)} abgelehnt."


def _check_table():
    ok = lambda b: "✅" if b else "❌"
    c = d["check"]
    if dynamic:
        rows = [("Jeder Auftrag höchstens einmal", ok(c["once"])), ("Fahrzeug war frei (Abstand ≥ D)", ok(c["free"])), ("Reichweite vom aktuellen Ort", ok(c["reach"])), ("Fahrzeiten stimmen", ok(c["cost"]) + " Summe " + ok(c["total"])),
                ("Regel eingehalten", ok(c["rule"]))]
    else:
        rows = [("Nur mögliche Paare", ok(c["feasible"])), ("Jedes Fahrzeug höchstens einmal", ok(c["unique_v"])), ("Jeder Auftrag höchstens einmal", ok(c["unique_o"])), ("Jeder Auftrag entschieden (Paar oder Ablehnung)", ok(c["all_decided"])),
                ("Kosten stimmen", ok(c["cost"])), ("Regel eingehalten (neu nachgerechnet)", ok(c["rule"]))]
    return {"Prüfung": [r[0] for r in rows], "Ergebnis": [r[1] for r in rows]}


def _render(k):
    """Zustand nach den ersten k Ankünften: links die Karte, rechts der Verlauf gegen die im Nachhinein beste Lösung."""
    with view_slot.container():
        c1, c2 = st.columns([3, 2])
        c1.markdown(f"**Nach {k} von {n_events} Ankünften** - " + _event_text(k))
        if dynamic:
            c1.plotly_chart(build_dynamic_map(sc, res, k), width="stretch", key=f"om_map_{k}")
        else:
            c1.plotly_chart(build_online_map(sc, a.order, res, k), width="stretch", key=f"om_map_{k}")
        c2.markdown("**Verlauf**")
        c2.plotly_chart(build_progress(online_series, best_series, k, online_name="bedient (Regel)" if dynamic else "zugeordnet (Regel)",
                                       best_name="Optimum im Nachhinein (ohne Zeitdruck)" if dynamic else "Optimum im Nachhinein", y_title="Aufträge" if dynamic else "Paare"),
                        width="stretch", key=f"om_progress_{k}")
        if dynamic:
            loc, t, free_at, pairs = Dy.state_at(sc, res, k)
            rows = [(i, free_at[i]) for i in range(min(sc.n, 12))]
            c2.table({"Fahrzeug": [f"F{i + 1}" for i, _ in rows], "Ort": [f"({loc[i][0]}, {loc[i][1]})" for i, _ in rows], "Stand": ["frei" if f <= t else f"belegt bis Minute {f}" for _, f in rows]})
        elif k > 0 and res.events[k - 1].kind == "assign":
            e = res.events[k - 1]
            c2.table({"Fahrzeug": [f"F{i + 1}" for i, _ in e.options], "Fahrzeit [min]": [c for _, c in e.options], "Gewählt": ["✅" if i == e.chosen else "" for i, _ in e.options]})
        if k == n_events:
            st.markdown("**Prüfung:** unabhängig nachgerechnet")
            st.table(_check_table())


if auto_play:
    frames = sorted({int(round(x)) for x in np.linspace(0, n_events, min(n_events, 40) + 1)})
    for k in frames:
        _render(k)
        time.sleep(min(0.6, 6.0 / max(len(frames), 1)))
    step = n_events
else:
    _render(step)

st.caption("Quadrate sind Fahrzeuge (hohl: frei, gefüllt: vergeben bzw. belegt), Kreise Aufträge (hellgrau: kommt noch, orange: zugeordnet, violett: wartet im Stapel, rotes Kreuz: abgelehnt). Blaue Linien sind feste Zuordnungen; beim letzten Ereignis zeigen grün gepunktete Linien die freien Fahrzeuge in Reichweite, die grüne Linie das gewählte. "
           "Im Verlauf steht die Regel gegen die im Nachhinein bestmögliche Paarzahl derselben ersten Aufträge - der Abstand ist der Preis der Unwiderruflichkeit.")

st.markdown("---")

# --- Wie gut? ---------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Wie gut ist die Regel?")
if dynamic:
    st.caption("Der Maßstab ist das im Nachhinein beste Ergebnis unter denselben Regeln der Belegung (Zuordnung bei Ankunft, feste Belegungszeit): erst möglichst viele bediente Aufträge, dann die geringste Fahrzeit (Min-Cost-Flow). Ein Verhältnis der Kosten gibt es nicht, weil sich die Zahl der bedienten Aufträge unterscheidet: die Fahrzeit steht je bedientem Auftrag.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Bediente Aufträge", f"{d['served']} von {d['opt_served']}", delta=f"{d['lost']} weniger als im Nachhinein" if d["lost"] else "so viele wie im Nachhinein", delta_color="off")
    m2.metric("Anteil", _f(d["quotient"]) if d["quotient"] is not None else "–", help="Bediente Aufträge der Regel geteilt durch die im Nachhinein bestmöglichen.")
    m3.metric("Fahrzeit je bedientem Auftrag", f"{_f(d['deadhead'], 1)} min", delta=f"im Nachhinein: {_f(d['opt_deadhead'], 1)} min", delta_color="off")
    m4.metric("Abgelehnt", str(d["rejected"]), delta=f"Belegungszeit D = {a.dur} min", delta_color="off")
    if code == "none":
        st.info("ℹ️ Kein Auftrag ist erreichbar - es gibt nichts zu bedienen.")
    elif code == "optimal":
        st.success(f"✅ Die Regel bedient {d['served']} von {d['opt_served']} Aufträgen - so viele wie im Nachhinein möglich.")
    else:
        st.warning(f"Die Regel bedient {d['served']} von {d['opt_served']} Aufträgen: {d['lost']} weniger, als im Nachhinein möglich gewesen wären.")
    if fixed:
        st.info("Feste Karte: es gibt nur diese eine Ziehung.")
    else:
        st.markdown(f"**Nicht nur diese eine Karte:** {len(C.DIST_SEEDS)} feste Karten mit denselben Einstellungen (Fahrzeuge {n}, Aufträge {m}, Reichweite {reach}, Ballung {ballung} %, Belegungszeit {dur} min), getrennt vom Seed oben.")
        ds = _dyn_stats(int(n), int(m), int(reach), int(ballung), int(dur))
        if ds["n_valid"] == 0:
            st.info("ℹ️ Bei dieser Reichweite ist auf keiner Karte ein Auftrag erreichbar.")
        else:
            p1, p2, p3 = st.columns(3)
            p1.metric("Greedy: Anteil (Mittel | Median)", f"{_f(ds['greedy_q_mean'], 3)} | {_f(ds['greedy_q_median'], 3)}", delta=f"schlechteste Karte {_f(ds['greedy_q_worst'], 3)}", delta_color="off")
            p2.metric("Ranking: Anteil (Mittel | Median)", f"{_f(ds['ranking_q_mean'], 3)} | {_f(ds['ranking_q_median'], 3)}", delta=f"schlechteste Karte {_f(ds['ranking_q_worst'], 3)}", delta_color="off", help="Erwartung über 10 Rangfolgen je Karte.")
            p3.metric("Fahrzeit je Auftrag", f"Greedy {_f(ds['deadhead_greedy'], 1)} | Ranking {_f(ds['deadhead_ranking'], 1)}", delta=f"im Nachhinein {_f(ds['deadhead_opt'], 1)} min", delta_color="off")
else:
    st.caption("**Gütequotient der Paare** = Paare der Regel geteilt durch die Paare des Offline-Optimums. **Prämie** = Mehrkosten gegenüber der billigsten Paarung mit derselben Paarzahl (Kosten sind über verschiedene Paarzahlen nicht vergleichbar). Ein Wettbewerbsverhältnis gibt es nur für die Paare; für die Kosten lässt sich keines angeben.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Paare (Regel)", f"{d['count']} von {d['opt_count']}", delta=f"{d['lost']} weniger als das Optimum" if d["lost"] else "so viele wie das Optimum", delta_color="off",
              help="Greedy und Ranking sind maximale Regeln (kein Auftrag wird bei freiem möglichen Fahrzeug abgelehnt), finden also mindestens die Hälfte der möglichen Paare.")
    m2.metric("Gütequotient", _f(d["quotient"]) if d["quotient"] is not None else "–")
    m3.metric("Kosten", f"{d['cost']} min", delta=(f"{_pct(d['premium_pct'], 1)} über der billigsten Paarung mit {d['count']} Paaren" if d["premium_pct"] is not None else "so billig wie möglich"), delta_color="off",
              help=f"Offline-Optimum: {d['opt_cost']} min mit {d['opt_count']} Paaren.")
    m4.metric("Aufwand (angesehene Kanten)", _int(d["scanned"]), delta=f"Offline: {_int(d['opt_scanned'])}", delta_color="off", help="Gezählt wird, nie Zeit: Greedy und Ranking sehen je Ankunft die möglichen Kanten zu den freien Fahrzeugen, Batching und das Optimum die der Ungarischen Methode.")
    if code == "none":
        st.info("ℹ️ Kein einziges Paar ist möglich – die Reichweite ist zu klein. Es gibt nichts zuzuordnen.")
    elif code == "optimal":
        st.success(f"✅ Die Regel findet {d['count']} Paare für {d['cost']} Minuten - genau das Offline-Optimum.")
    elif code == "worse_cost":
        st.info(f"Alle {d['count']} möglichen Paare gefunden, aber {_pct(d['premium_pct'], 1)} teurer als die billigste Paarung mit dieser Paarzahl ({d['cost']} gegen {d['opt_cost']} Minuten): der Verlust liegt hier nur bei den Kosten.")
    else:
        st.warning(f"Die Regel findet {d['count']} von {d['opt_count']} möglichen Paaren ({_f(d['quotient'])}): {d['lost']} Paare gingen verloren, weil früh eingetroffene Aufträge Fahrzeuge belegt haben, die später eintreffende dringender gebraucht hätten.")
    st.markdown("**Die Regeln im Vergleich auf dieser Karte (dieselbe Reihenfolge)**")
    cmp_rows = ev.compare_table(a)
    st.table({"Regel": [r["label"] for r in cmp_rows], "Paare": [r["count"] for r in cmp_rows], "Gütequotient": [_f(r["quotient"]) for r in cmp_rows], "Kosten [min]": [r["cost"] for r in cmp_rows],
              "Prämie": [_pct(r["premium_pct"], 1) for r in cmp_rows], "angesehene Kanten": [_int(r["scanned"]) for r in cmp_rows]})
    if fixed:
        st.info("Feste Karte: es gibt nur diese eine Ziehung. Für die Verteilung über viele Karten eine zufällige Karte wählen.")
    elif code != "none":
        st.markdown(f"**Nicht nur diese eine Karte:** {len(C.DIST_SEEDS)} feste Karten mit denselben Einstellungen (Fahrzeuge {n}, Aufträge {m}, Reichweite {reach}, Ballung {ballung} %, Ankunft: {C.ARR_LABELS[arr]}), getrennt vom Seed oben.")
        dist = _distribution(int(n), int(m), int(reach), int(ballung), arr, int(w))
        if dist["n_valid"] == 0:
            st.info("ℹ️ Bei dieser Reichweite gibt es auf keiner der Karten ein mögliches Paar.")
        else:
            key = {"greedy": "greedy", "ranking": "ranking", "batch": "batch"}[rule]
            s = dist[key]
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Gütequotient (Mittel | Median)", f"{_f(s['q_mean'], 3)} | {_f(s['q_median'], 3)}", delta=f"schlechteste Karte {_f(s['q_worst'], 3)}", delta_color="off", help="Paare der gewählten Regel geteilt durch die des Optimums, über die 100 Karten (Ranking: Erwartung über 20 Rangfolgen je Karte).")
            p2.metric("Prämie (Mittel | Median)", f"{_pct(s['premium_mean'], 1)} | {_pct(s['premium_median'], 1)}", delta=f"{_f(s['lost_mean'])} Paare weniger als das Optimum", delta_color="off")
            p3.metric("Optimum im Mittel", f"{_f(dist['opt_pairs_mean'], 1)} Paare", delta=f"{_f(dist['opt_cost_mean'], 0)} min", delta_color="off")
            p4.metric("Greedy schlägt Ranking", _share(dist["ranking_worse_share"]), delta=f"Ranking im Mittel {dist['ranking_minus_greedy_mean']:+.2f} Paare".replace(".", ","), delta_color="off",
                      help="Anteil der Karten, auf denen Ranking (Erwartung) weniger Paare findet als Greedy, und die mittlere Differenz.")
            st.table({"Regel": ["Greedy", "Ranking (Erwartung)", "Greedy nach Index", f"Batching, Fenster {w}"],
                      "Quotient Mittel": [_f(dist[k]["q_mean"], 3) for k in ("greedy", "ranking", "index", "batch")], "Median": [_f(dist[k]["q_median"], 3) for k in ("greedy", "ranking", "index", "batch")],
                      "schlechteste Karte": [_f(dist[k]["q_worst"], 3) for k in ("greedy", "ranking", "index", "batch")], "Prämie Mittel": [_pct(dist[k]["premium_mean"], 1) for k in ("greedy", "ranking", "index", "batch")],
                      "Prämie Median": [_pct(dist[k]["premium_median"], 1) for k in ("greedy", "ranking", "index", "batch")]})
            h1, h2 = st.columns(2)
            h1.plotly_chart(build_hist(s["q"], mean=s["q_mean"], median=s["q_median"], current=d["quotient"], x_title="Gütequotient der Paare"), width="stretch", key="om_hist_q")
            h1.caption("Gütequotient über die Karten (Regel wie oben); die rote Linie ist Ihre Ziehung.")
            h2.plotly_chart(build_hist([p for p in s["prem"]], mean=s["premium_mean"], median=s["premium_median"], current=d["premium_pct"], x_title="Prämie der Kosten [%]", bin_size=5, color="#d62728"), width="stretch", key="om_hist_p")
            h2.caption("Prämie über die Karten. Die Verteilung ist rechtsschief, deshalb Mittel und Median zusammen.")

st.markdown("---")

# --- Experimente auf Abruf ----------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Wovon hängt es ab?")
if not fixed and not dynamic:
    if st.button("Was ist Vorausschau wert? Batching mit wachsendem Fenster (100 Karten)", key="win_start"):
        st.session_state["win_on"] = (int(n), int(m), int(reach), int(ballung), arr)
    if st.session_state.get("win_on") == (int(n), int(m), int(reach), int(ballung), arr):
        with st.spinner("Rechne Fenster × 100 Karten..."):
            wrows = _window_sweep(int(n), int(m), int(reach), int(ballung), arr)
        c1, c2 = st.columns([3, 2])
        c1.plotly_chart(build_window_sweep(wrows), width="stretch", key="om_window_chart")
        c2.table({"Fenster w": [r["w"] for r in wrows], "Quotient Mittel": [_f(r["q_mean"], 3) for r in wrows], "schlechteste Karte": [_f(r["q_worst"], 3) for r in wrows], "Prämie Mittel": [_pct(r["premium_mean"], 1) for r in wrows],
                  "größeres Fenster schlechter": ["–" if r["fewer_than_prev"] is None else f"{r['fewer_than_prev']} Karten" for r in wrows], "angesehene Kanten": [_int(r["scanned"]) for r in wrows]})
        st.caption("Im statischen Modell kostet Warten nichts: mehr Fenster heißt mehr Information. Im Mittel wächst der Quotient mit dem Fenster (ab m Aufträgen ist es das Offline-Optimum), je Karte nicht immer: ein größeres Fenster kann weniger Paare finden. Der Aufwand wächst mit.")

if st.button("Die Treppe: wie schlecht wird Greedy, wie gut Ranking? (n bis 160)", key="stairs_start"):
    st.session_state["stairs_on"] = True
if st.session_state.get("stairs_on"):
    with st.spinner("Rechne die Treppe..."):
        srows, sarr = _stairs()
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(build_stair_curve(srows), width="stretch", key="om_stairs_chart")
    c2.table({"n": [r["n"] for r in srows], "Greedy": [r["greedy"] for r in srows], "Ranking (Erwartung)": [_f(r["ranking"], 2) + ("" if r["exact"] else " (MC)") for r in srows], "Ranking / n": [_f(r["ranking_ratio"], 3) for r in srows]})
    st.caption("Auf der Treppe erreicht Auftrag t genau die Fahrzeuge 0 bis n − t − 1. Greedy nimmt für jeden das nächste, also das flexibelste Fahrzeug, und findet ⌈n/2⌉ von n Paaren. Ranking liegt im Erwartungswert von oben über 1 − 1/e ≈ 0,632 (bis n = 8 exakt über alle Rangfolgen, darüber Monte Carlo mit 2000 Ziehungen).")
    st.table({"n = 20, abstrakte Treppe": ["Flexible zuerst", "Starre zuerst", "Zufällige Reihenfolge"] , "Greedy": [sarr["flex"]["greedy"], sarr["rigid"]["greedy"], _f(sarr["random"]["greedy"], 1)],
              "Ranking (Mittel)": [_f(sarr["flex"]["ranking"], 1), _f(sarr["rigid"]["ranking"], 1), _f(sarr["random"]["ranking"], 1)]})
    st.caption("Batching in der schlechtesten Reihenfolge (Fenster " + ", ".join(f"{wk}: {v}" for wk, v in sarr["batch"].items()) + " Paare von 20): das Batch-Optimum wählt die billigen Fahrzeuge - und die sind die flexiblen. Erst ein Fenster über die ganze Treppe hilft.")

if not fixed and not dynamic:
    if st.button("Ankunftsmodelle im Vergleich (40 Karten)", key="arr_start"):
        st.session_state["arr_on"] = (int(n), int(m), int(reach), int(ballung))
    if st.session_state.get("arr_on") == (int(n), int(m), int(reach), int(ballung)):
        with st.spinner("Rechne 5 Ankunftsmodelle × 40 Karten..."):
            arows = _arrivals(int(n), int(m), int(reach), int(ballung))
        c1, c2 = st.columns([3, 2])
        c1.plotly_chart(build_arrival_bars(arows, C.ARR_LABELS), width="stretch", key="om_arrival_chart")
        c2.table({"Ankunft": [C.ARR_LABELS.get(r["arr"], "Schlechteste gefundene") for r in arows], "Greedy": [_f(r["greedy"]["q_mean"], 3) for r in arows], "Greedy schlechteste Karte": [_f(r["greedy"]["q_worst"], 3) for r in arows],
                  "Ranking": [_f(r["ranking"]["q_mean"], 3) for r in arows], "Batching 10": [_f(r["batch"]["q_mean"], 3) for r in arows], "Greedy-Prämie": [_pct(r["greedy"]["premium_mean"], 1) for r in arows]})
        st.caption("Mittlerer Gütequotient der Paare über 40 feste Karten; Batching mit Fenster 10. Die letzte Zeile ist eine per Bergsteigen (300 Vertauschungen) gefundene schlechte Reihenfolge für Greedy - auf Zufallskarten fällt Greedy damit nicht annähernd auf 1/2; das gelingt nur der Treppe.")

if not fixed and dynamic:
    if st.button("Belegungszeit von 1 bis 121 Minuten durchfahren (100 Karten je Wert, dauert einige Sekunden)", key="dyn_start"):
        st.session_state["dyn_on"] = (int(n), int(m), int(reach), int(ballung))
    if st.session_state.get("dyn_on") == (int(n), int(m), int(reach), int(ballung)):
        with st.spinner("Rechne 11 Belegungszeiten × 100 Karten..."):
            drows = _dyn_sweep(int(n), int(m), int(reach), int(ballung))
        c1, c2 = st.columns([3, 2])
        c1.plotly_chart(build_dynamic_sweep([r for r in drows if r["n_valid"]]), width="stretch", key="om_dyn_chart")
        c2.table({"D [min]": [r["dur"] for r in drows], "Greedy Mittel": [_f(r.get("greedy_q_mean"), 3) for r in drows], "schlechteste": [_f(r.get("greedy_q_worst"), 3) for r in drows], "Ranking Mittel": [_f(r.get("ranking_q_mean"), 3) for r in drows]})
        st.caption("Anteil der im Nachhinein bedienbaren Aufträge, den die Regel bedient. Bei sehr kurzer Belegung sind die Fahrzeuge fast sofort wieder da, bei sehr langer wird nie ein Fahrzeug frei (statisches Modell): dazwischen ist der Verlust am größten. Wo, hängt von der Last (Aufträge je Fahrzeug) ab.")

if not fixed and not dynamic:
    if st.button("Aufwand: angesehene Kanten je Regel", key="scan_start"):
        st.session_state["scan_on"] = (int(n), int(m), int(reach), int(ballung), arr)
    if st.session_state.get("scan_on") == (int(n), int(m), int(reach), int(ballung), arr):
        srows2 = _scans(int(n), int(m), int(reach), int(ballung), arr)
        c1, c2 = st.columns([3, 2])
        c1.plotly_chart(build_scan_bars(srows2), width="stretch", key="om_scan_chart")
        c2.table({"Regel": [r["label"] for r in srows2], "angesehene Kanten": [_int(r["scanned"]) for r in srows2]})
        st.caption("Mittel über 40 feste Karten. Ein größeres Fenster kostet mehr Aufwand, weil die Ungarische Methode größere Stapel löst; das Fenster über alle Aufträge ist genau die Offline-Rechnung.")

st.markdown("---")

# --- Grenzen ---------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Fahrzeuge sind vorab bekannt** | Tauchen auch Fahrzeuge erst nach und nach auf, beidseitig online, ist das Problem schwerer (Vertex-Arrival-Varianten). | (nicht gebaut) |
| **Zusagen sind unwiderruflich** | Darf man umhängen (Verbesserungsweg je Ankunft), erreicht man auf jeder Reihenfolge das Offline-Optimum: der Preis ist die Unwiderruflichkeit, nicht die Reihenfolge. | **Verbesserungswege**, **Ungarische Methode** (gebaut) |
| **Es wird immer bedient, wenn möglich** | Wer bewusst ablehnt, um ein Fahrzeug für später zu sparen, braucht Wissen über die Zukunft. | (Vorhersage, nicht gebaut) |
| **Die Reihenfolge ist zufällig oder gegnerisch** | Bei i. i. d. gezogenen Aufträgen aus einer bekannten Verteilung geht mehr als 1 − 1/e; im Worst Case nicht. Auf Zufallskarten ist die gegnerische Reihenfolge harmlos, nur die Treppe erreicht 1/2. | (nicht gebaut) |
| **Feste Belegungszeit, Fahrzeug bleibt am Ort** | Hängt die Belegung von Fahrt und Service ab oder kehren Fahrzeuge zum Depot zurück, wird schon das Offline-Problem NP-schwer. | Routenplanung |
| **Im Modell mit Zeit wird sofort entschieden** | Wartet man auf ein frei werdendes Fahrzeug, kann man mehr Aufträge bedienen als der Sofort-Maßstab - der Vergleich wäre nicht mehr fair. | (nicht gebaut) |
| **Nur Paare zählen** | Für die Kosten gibt es kein festes Wettbewerbsverhältnis, und Ranking ist kostenblind (deutlich teurer als Greedy); kanten- und knotengewichtete Varianten sind eigene Verfahren. | (nicht gebaut) |
"""
)
st.caption("Die Nachbarn der Matching-Linie (noch nicht gebaut): Blossom, Gewichteter Blossom und Stabile Mitbewohner. Bereits gebaut: die Wurzel (Greedy-Matching), die Verbesserungswege, Hopcroft–Karp, die Ungarische Methode, der Auktionsalgorithmus, Gale–Shapley und diese Demo.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** Bipartiter Graph mit Fahrzeugen $V$ (vorab bekannt) und Aufträgen $O$, die in einer Reihenfolge $\pi$ eintreffen; ein Paar $(v,o)$ ist möglich, wenn die Fahrzeit höchstens die Reichweite beträgt. Beim Eintreffen von $o$ sieht die Regel nur seine Nachbarn $N(o)$ und die früheren Entscheidungen und wählt sofort und unwiderruflich ein freies $v\in N(o)$ oder lehnt ab. Ziel: erst $|M|$ maximal, dann $c(M)$ minimal (lexikografisch).

**Gütequotient.** $\rho = |M_{\text{online}}|/\nu$ mit $\nu$ = Größe des größtmöglichen Matchings (Offline). Für die Kosten gibt es kein festes Verhältnis; sie werden als Prämie $(c(M)-\sum_{i\le k}W_i)/\sum_{i\le k}W_i$ gegenüber der billigsten Paarung mit $k=|M|$ Paaren gemessen ($W_i$ = Grenzkosten der Ungarischen Methode).

**Greedy.** Jede maximale Regel liefert ein maximales Matching, also mindestens $\nu/2$ Paare. Das ist scharf: auf der **Treppe** ($o_t$ erreicht genau $v_0,\dots,v_{n-t-1}$, Kosten streng steigend im Index) nimmt Greedy für $o_0$ das billigste, also flexibelste $v_0$, für $o_1$ das Fahrzeug $v_1$ usw. und findet $\lceil n/2\rceil$ von $n$ Paaren; $\rho\to 1/2$. Kein deterministisches Verfahren ist besser als $1/2$ im Worst Case.

**Ranking (Karp–Vazirani–Vazirani 1990).** Ziehe eine gleichverteilte Rangfolge $\sigma$ der Fahrzeuge; jeder Auftrag nimmt das bestplatzierte freie Fahrzeug in $N(o)$. Dann ist $\mathbb{E}|M|\ge (1-1/e)\,\nu\approx 0{,}632\,\nu$ gegen jede Reihenfolge $\pi$ (Beweis über Primal-Dual, Devanur–Jain–Kleinberg 2013). Die Schranke ist scharf; auf der Treppe nähert sich $\mathbb{E}|M|/n$ von oben $1-1/e$. Ranking ignoriert die Kosten.

**Batching.** Alle $w$ Ankünfte löst die Ungarische Methode den Stapel optimal für die noch freien Fahrzeuge. Fenster $1$ ist Greedy, Fenster $\ge m$ das Offline-Optimum. Im Mittel wächst der Quotient mit $w$, je Karte nicht (Stapel-Optimum wählt die billigen, also flexiblen Fahrzeuge, die Falle kehrt wieder).

**Mit Zeit.** Auftrag $j$ trifft zur Minute $a_j$ ein; ein zugeordnetes Fahrzeug ist $D$ Minuten belegt und steht danach am Ort von $j$. Offline (Zuordnung bei Ankunft) ist ein Min-Cost-Flow: Quelle $\to$ Fahrzeug $\to(\text{in}_j\to\text{out}_j\to\text{in}_k\to\cdots)\to$ Senke, $\text{in}_j\to\text{out}_j$ mit Kosten $-L$, $L=1+m\cdot 142$, und $\text{out}_j\to\text{in}_k$ genau wenn $a_k\ge a_j+D$ und $k$ von $j$ aus erreichbar ist; $n$ Flusseinheiten, eine je Fahrzeug.

**Grenzen.** (1) Fahrzeuge vorab bekannt. (2) Unwiderruflich. (3) Immer bedienen. (4) Reihenfolge zufällig/gegnerisch. (5) Feste Belegungszeit. (6) Sofortentscheidung mit Zeit. (7) Kosten haben kein festes Wettbewerbsverhältnis.

Implementiert in `om_scenario.py` (Karten, eigener Zufallsgenerator, Ankunftsströme, Treppe), `om_online.py` (Greedy, Ranking, Batching, Ereignisprotokoll, Prüfer), `om_dynamic.py` (Modell mit Zeit, Min-Cost-Flow, Prüfer), `om_hungarian.py`, `om_augment.py` und `om_greedy.py` (aus den Vorgängerdemos), `om_evaluation.py` (Kennzahlen, Verteilungen, Sweeps).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
