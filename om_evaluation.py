"""Auswertung: eine Karte (`analyse`, `verdict`, `compare_table`), viele Karten (`distribution`, `window_sweep`, `arrival_comparison`), die Treppe (`staircase_curve`,
`staircase_arrivals`), die Zeitdimension (`analyse_dynamic`, `dynamic_sweep`) und der Aufwand. Alles ganzzahlig und deterministisch; nur die Anzeige-Statistiken sind Gleitkomma.

Das Ziel ist lexikografisch (erst Paare, dann Kosten). Die Güte einer Online-Regel steht deshalb in zwei Zahlen: dem **Gütequotient der Paare** (Online-Paare geteilt durch die Paare
des Offline-Optimums der Ungarischen Methode, je Karte) und der **Prämie** der Kosten gegenüber der billigsten Paarung mit derselben Paarzahl (Grenzkosten der Ungarischen Methode).
"Wettbewerbsverhältnis" gibt es nur für die Paare; für Kosten lässt sich kein festes Verhältnis angeben. Mittel, Median und die schlechteste Karte stehen zusammen.
"""

import math
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import permutations

import numpy as np

import om_constants as C
import om_dynamic as Dy
from om_hungarian import hungarian
from om_online import RULE_BATCH, RULE_GREEDY, RULE_RANKING, prefix_optimum, run_online, validate
from om_scenario import SplitMix64, arrival_order, build, generate, perm, priority, staircase_graph, times

OPTIMAL, WORSE_COST, FEWER_PAIRS, NONE = "optimal", "worse_cost", "fewer_pairs", "none"


@dataclass
class Analysis:
    scenario: object
    order: tuple
    arr: str
    rule: str
    w: int
    prio_k: int
    seed: int
    online: object        # gewählte Regel
    greedy: object
    ranking: object       # Rangfolge Nr. prio_k
    index: object         # Ranking mit der Identität als Rangfolge = "Greedy nach Index"
    batch: object
    hung: object          # Offline-Optimum
    kcost: tuple          # kcost[k] = Kosten der billigsten Paarung mit k Paaren


def _kcost(hung):
    out = [0]
    for w in hung.marginal:
        out.append(out[-1] + w)
    return tuple(out)


def cost_of(sc, pairs):
    return int(sum(sc.cost[i, j] for i, j in pairs))


def premium_of(cost, count, kcost):
    """Mehrkosten in Prozent gegenüber der billigsten Paarung mit derselben Paarzahl (None, wenn diese kostenlos ist)."""
    base = kcost[count] if count < len(kcost) else 0
    return None if not base else 100.0 * (cost - base) / base


def analyse(sc, arr=C.DEFAULT_ARR, rule=C.DEFAULT_RULE, w=C.DEFAULT_W, seed=0, prio_k=0, order=None):
    order = tuple(arrival_order(sc, arr, seed)) if order is None else tuple(order)
    prio = priority(sc.n, seed, prio_k)
    g = run_online(sc, order, RULE_GREEDY)
    rk = run_online(sc, order, RULE_RANKING, prio=prio)
    ix = run_online(sc, order, RULE_RANKING, prio=list(range(sc.n)))
    bt = run_online(sc, order, RULE_BATCH, w=w)
    hung = hungarian(sc, record=False)
    chosen = {RULE_GREEDY: g, RULE_RANKING: rk, RULE_BATCH: bt}[rule]
    return Analysis(sc, order, arr, rule, w, prio_k, seed, chosen, g, rk, ix, bt, hung, _kcost(hung))


def verdict(a):
    """(Stufe, Code, Zahlen) für die Anzeige; `Zahlen` enthält jede Zahl, die der Text nennt."""
    sc, r, h = a.scenario, a.online, a.hung
    prem = premium_of(r.cost, r.count, a.kcost)
    code = NONE if h.count == 0 else (OPTIMAL if (r.count, r.cost) == (h.count, h.cost) else (WORSE_COST if r.count == h.count else FEWER_PAIRS))
    check = validate(sc, r)
    data = {"count": r.count, "cost": r.cost, "opt_count": h.count, "opt_cost": h.cost, "lost": h.count - r.count, "quotient": (r.count / h.count) if h.count else None,
            "premium_pct": prem, "scanned": r.scanned_total, "opt_scanned": h.scanned_total, "check": check, "rejected": len(r.rejected),
            "n_ties": sum(1 for e in a.greedy.events if e.kind == "assign" and [c for _, c in e.options].count(min(c for _, c in e.options)) > 1)}
    level = {NONE: "info", OPTIMAL: "success", WORSE_COST: "info", FEWER_PAIRS: "warning"}[code]
    return level, code, data


def compare_table(a):
    rows = []
    for label, res in ((f"Greedy: das nächste freie Fahrzeug", a.greedy), (f"Ranking (Rangfolge Nr. {a.prio_k + 1})", a.ranking), (f"Batching, Fenster {a.batch.w}", a.batch)):
        rows.append({"label": label, "count": res.count, "cost": res.cost, "premium_pct": premium_of(res.cost, res.count, a.kcost), "quotient": res.count / a.hung.count if a.hung.count else None, "scanned": res.scanned_total})
    h = a.hung
    rows.append({"label": "Offline-Optimum (Ungarische Methode)", "count": h.count, "cost": h.cost, "premium_pct": 0.0 if h.count else None, "quotient": 1.0 if h.count else None, "scanned": h.scanned_total})
    return rows


def prefix_series(a):
    """Verlauf über die Ankünfte: Paare der gewählten Regel gegen die im Nachhinein optimalen Paare desselben Präfixes (Messlatte im Verlaufsdiagramm)."""
    online, best, cur = [0], [0], 0
    for k, e in enumerate(a.online.events, start=1):
        cur += len(e.assigned)
        online.append(cur)
        best.append(prefix_optimum(a.scenario, a.order, k)[0])
    return online, best


# --- viele Karten --------------------------------------------------------------------------------------------------------------

def worst_order(sc, seed, iters=C.HILL_SWEEPS):
    """Bergsteigen auf der Reihenfolge: `iters` zufällige Vertauschungen, behalten, wenn Greedy dadurch schlechter wird (weniger Paare, dann höhere Kosten)."""
    rng = SplitMix64(seed * 31 + 5)
    order = arrival_order(sc, "random", seed)
    best = run_online(sc, order, RULE_GREEDY).key()
    for _ in range(iters):
        i, j = rng.below(sc.m), rng.below(sc.m)
        if i == j:
            continue
        o2 = order[:]
        o2[i], o2[j] = o2[j], o2[i]
        k2 = run_online(sc, o2, RULE_GREEDY).key()
        if k2 < best:
            best, order = k2, o2
    return order


def _row(sc, order, seed, windows, k_draws):
    h = hungarian(sc, record=False)
    kc = _kcost(h)
    n = sc.n
    g = run_online(sc, order, RULE_GREEDY)
    gh = run_online(sc, order, RULE_GREEDY, tie="high")
    gi = run_online(sc, order, RULE_RANKING, prio=list(range(n)))
    rk = [run_online(sc, order, RULE_RANKING, prio=priority(n, seed, k)) for k in range(k_draws)]
    bt = {w: run_online(sc, order, RULE_BATCH, w=w) for w in windows}
    ties = decisions = 0
    for e in g.events:
        if e.kind == "assign":
            decisions += 1
            cs = sorted(c for _, c in e.options)
            ties += len(cs) > 1 and cs[0] == cs[1]
    return {"opt": (h.count, h.cost, h.scanned_total), "kc": kc, "greedy": (g.count, g.cost, g.scanned_total, tuple(sorted(g.pairs))), "greedy_high": (gh.count, gh.cost, tuple(sorted(gh.pairs))),
            "index": (gi.count, gi.cost, tuple(sorted(gi.pairs))), "ranking": tuple((r.count, r.cost, r.scanned_total) for r in rk),
            "batch": {w: (r.count, r.cost, r.scanned_total, tuple(sorted(r.pairs))) for w, r in bt.items()}, "ties": (ties, decisions)}


@lru_cache(maxsize=128)
def cell_rows(n, m, reach, ballung, arr, seeds, windows, k_draws):
    rows = []
    for sd in seeds:
        sc = generate(n, m, reach, ballung, sd)
        rows.append(_row(sc, arrival_order(sc, arr, sd), sd, windows, k_draws))
    return tuple(rows)


def _med(values):
    return float(np.median(values)) if len(values) else None


def _mean(values):
    return float(np.mean(values)) if len(values) else None


def _prem_list(prems):
    return [p for p in prems if p is not None]


def _rule_stats(rows, get):
    """Kennzahlen einer Regel über die Karten. `get(row)` liefert eine Liste von (Paare, Kosten) - eine Ziehung oder mehrere (Ranking: Mittel über die Ziehungen je Karte)."""
    q, lost, prem, cnt = [], [], [], []
    for r in rows:
        draws = get(r)
        oc = r["opt"][0]
        pairs = [d[0] for d in draws]
        q.append(float(np.mean(pairs)) / oc)
        lost.append(oc - float(np.mean(pairs)))
        cnt.append(float(np.mean(pairs)))
        pr = _prem_list([premium_of(d[1], d[0], r["kc"]) for d in draws])
        if pr:
            prem.append(float(np.mean(pr)))
    return {"q_mean": _mean(q), "q_median": _med(q), "q_worst": min(q), "lost_mean": _mean(lost), "pairs_mean": _mean(cnt), "premium_mean": _mean(prem), "premium_median": _med(prem),
            "optimal_share": _mean([l == 0 for l in lost]), "q": q, "prem": prem}


def distribution(n, m, reach, ballung, arr=C.DEFAULT_ARR, w=C.DEFAULT_W, seeds=C.DIST_SEEDS, k_draws=C.RANKING_K):
    """Verteilung über viele Karten: Gütequotient der Paare und Prämie der Kosten für Greedy, Ranking (Erwartung über `k_draws` Rangfolgen), Greedy nach Index und Batching (Fenster `w`)."""
    windows = tuple(sorted({w, 1, 2, 5, 10, m}))
    rows = [r for r in cell_rows(n, m, reach, ballung, arr, tuple(seeds), windows, k_draws) if r["opt"][0] > 0]
    if not rows:
        return {"n_seeds": len(seeds), "n_valid": 0}
    out = {"n_seeds": len(seeds), "n_valid": len(rows), "opt_pairs_mean": _mean([r["opt"][0] for r in rows]), "opt_cost_mean": _mean([r["opt"][1] for r in rows]),
           "greedy": _rule_stats(rows, lambda r: [r["greedy"][:2]]), "ranking": _rule_stats(rows, lambda r: [d[:2] for d in r["ranking"]]), "index": _rule_stats(rows, lambda r: [r["index"][:2]]),
           "batch": _rule_stats(rows, lambda r: [r["batch"][w][:2]]), "batch_all": _rule_stats(rows, lambda r: [r["batch"][m][:2]])}
    gp = [r["greedy"][0] for r in rows]
    rp = [float(np.mean([d[0] for d in r["ranking"]])) for r in rows]
    out["ranking_minus_greedy_mean"] = _mean([a - b for a, b in zip(rp, gp)])
    out["ranking_minus_greedy_median"] = _med([a - b for a, b in zip(rp, gp)])
    out["ranking_worse_share"] = _mean([a < b for a, b in zip(rp, gp)])
    out["greedy_index_same_pairs_share"] = _mean([r["greedy"][0] == r["index"][0] for r in rows])
    out["greedy_index_same_matching_share"] = _mean([r["greedy"][3] == r["index"][2] for r in rows])
    out["tie_share"] = sum(r["ties"][0] for r in rows) / max(sum(r["ties"][1] for r in rows), 1)
    out["tie_break_pairs_delta"] = _mean([r["greedy_high"][0] - r["greedy"][0] for r in rows])
    out["tie_break_differ_share"] = _mean([r["greedy_high"][2] != r["greedy"][3] for r in rows])
    out["greedy_scanned_mean"] = _mean([r["greedy"][2] for r in rows])
    out["opt_scanned_mean"] = _mean([r["opt"][2] for r in rows])
    return out


def window_sweep(n, m, reach, ballung, arr=C.DEFAULT_ARR, seeds=C.DIST_SEEDS, windows=C.WINDOWS):
    """Was ist Vorausschau wert? Batching mit wachsendem Fenster (das größte, Fenster >= m, ist die Offline-Lösung) - Quotient, Prämie, Aufwand und wie oft ein größeres Fenster weniger Paare bringt."""
    ws = tuple(sorted(set(windows) | {m}))
    rows = [r for r in cell_rows(n, m, reach, ballung, arr, tuple(seeds), ws, C.RANKING_K) if r["opt"][0] > 0]
    out, prev = [], None
    for w in ws:
        st = _rule_stats(rows, lambda r, w=w: [r["batch"][w][:2]])
        cur = [r["batch"][w][0] for r in rows]
        out.append({"w": w, "q_mean": st["q_mean"], "q_median": st["q_median"], "q_worst": st["q_worst"], "premium_mean": st["premium_mean"], "premium_median": st["premium_median"],
                    "scanned": _mean([r["batch"][w][2] for r in rows]), "fewer_than_prev": (None if prev is None else sum(a < b for a, b in zip(cur, prev))), "n_valid": len(rows)})
        prev = cur
    return out


def arrival_comparison(n, m, reach, ballung, seeds=C.SWEEP_SEEDS, w=10):
    """Greedy, Ranking und Batching unter den Ankunftsmodellen (zufällig, flexible zuerst, starre zuerst, links nach rechts) und unter der per Bergsteigen gefundenen schlechtesten Reihenfolge für Greedy."""
    out = []
    for arr in list(C.ARR_LABELS) + ["worst"]:
        rows = []
        for sd in seeds:
            sc = generate(n, m, reach, ballung, sd)
            if hungarian(sc, record=False).count == 0:
                continue
            order = worst_order(sc, sd) if arr == "worst" else arrival_order(sc, arr, sd)
            rows.append(_row(sc, order, sd, (w,), C.RANKING_K))
        g = _rule_stats(rows, lambda r: [r["greedy"][:2]])
        rk = _rule_stats(rows, lambda r: [d[:2] for d in r["ranking"]])
        bt = _rule_stats(rows, lambda r: [r["batch"][w][:2]])
        out.append({"arr": arr, "greedy": g, "ranking": rk, "batch": bt, "n_valid": len(rows)})
    return out


def scan_table(n, m, reach, ballung, arr=C.DEFAULT_ARR, seeds=C.SWEEP_SEEDS):
    """Aufwand in angesehenen Kanten (Mittel): Greedy, Batching mit wachsendem Fenster und das Offline-Optimum."""
    ws = tuple(sorted(set(C.WINDOWS) | {m}))
    rows = [r for r in cell_rows(n, m, reach, ballung, arr, tuple(seeds), ws, 1) if r["opt"][0] > 0]
    out = [{"label": "Greedy", "scanned": _mean([r["greedy"][2] for r in rows])}]
    out += [{"label": f"Batching, Fenster {w}" if w < m else f"Batching, Fenster {w} (alles)", "scanned": _mean([r["batch"][w][2] for r in rows])} for w in ws]
    out.append({"label": "Offline (Ungarische Methode)", "scanned": _mean([r["opt"][2] for r in rows])})
    return out


# --- die Treppe ---------------------------------------------------------------------------------------------------------------------

def _stair_pairs(ranks_matrix):
    """Ranking auf der Treppe für viele Rangfolgen gleichzeitig: Auftrag t erreicht die Fahrzeuge 0..n-t-1 und nimmt das bestplatzierte freie darunter.
    `ranks_matrix`: (K, n), Zeile = Rang je Fahrzeug. Rückgabe: Paarzahl je Zeile."""
    key = ranks_matrix.astype(np.float64)
    K, n = key.shape
    rows = np.arange(K)
    pairs = np.zeros(K, dtype=np.int64)
    for t in range(n):
        sub = key[:, :n - t]
        idx = np.argmin(sub, axis=1)
        ok = np.isfinite(sub[rows, idx])
        pairs += ok
        key[rows[ok], idx[ok]] = np.inf
    return pairs


@lru_cache(maxsize=16)
def staircase_exact(n):
    """Exakter Erwartungswert von Ranking auf der Treppe über alle n! Rangfolgen (Bruch)."""
    perms = np.array(list(permutations(range(n))), dtype=np.int64)
    return Fraction(int(_stair_pairs(perms).sum()), len(perms))


@lru_cache(maxsize=16)
def staircase_mc(n, draws=C.STAIR_MC_DRAWS):
    """Ranking auf der Treppe: Mittel über `draws` Rangfolgen aus dem eigenen Zufallsstrom (feste Seeds) - der Wert ist gepinnt, nicht 'zufällig'."""
    R = np.zeros((draws, n), dtype=np.int64)
    for k in range(draws):
        for pos, v in enumerate(perm(n, 5_000_000 + 1000 * n + k)):
            R[k, v] = pos
    p = _stair_pairs(R)
    return float(p.mean()), float(p.std())


def staircase_curve(ns=C.STAIR_NS):
    """Greedy und Ranking gegen n auf der Treppe: Greedy findet ceil(n/2) (Verhältnis -> 1/2), Ranking nähert sich von oben 1 - 1/e = 0,632."""
    out = []
    for n in ns:
        if n <= C.STAIR_EXACT_MAX:
            e, exact = float(staircase_exact(n)), True
        else:
            e, exact = staircase_mc(n)[0], False
        out.append({"n": n, "greedy": -(-n // 2), "greedy_ratio": -(-n // 2) / n, "ranking": e, "ranking_ratio": e / n, "exact": exact, "opt": n})
    return out


def staircase_arrivals(n=20, w_list=(1, 2, 5, 10), k_draws=200):
    """Die Treppe abstrakt (beliebiges n) unter verschiedenen Ankunftsreihenfolgen: flexible zuerst (Nummerierung), starre zuerst (umgekehrt), zufällig (Mittel über k Ziehungen);
    Greedy und Ranking (Mittel über die Rangfolgen) sowie Batching mit wachsendem Fenster in der schlechtesten Reihenfolge."""
    sc = staircase_graph(n)
    rank_mats = [priority(n, 777, k) for k in range(k_draws)]
    out = {"n": n}
    orders = {"flex": list(range(n)), "rigid": list(range(n))[::-1]}
    for kind in ("flex", "rigid"):
        o = orders[kind]
        out[kind] = {"greedy": run_online(sc, o, RULE_GREEDY).count, "ranking": float(np.mean([run_online(sc, o, RULE_RANKING, prio=pr).count for pr in rank_mats]))}
    rnd = [perm(n, 900 + k) for k in range(k_draws)]
    out["random"] = {"greedy": float(np.mean([run_online(sc, o, RULE_GREEDY).count for o in rnd])),
                     "ranking": float(np.mean([run_online(sc, o, RULE_RANKING, prio=rank_mats[k]).count for k, o in enumerate(rnd)]))}
    out["batch"] = {w: run_online(sc, orders["flex"], RULE_BATCH, w=w).count for w in tuple(w_list) + (n,)}
    return out


# --- die Zeitdimension --------------------------------------------------------------------------------------------------------

@dataclass
class DynAnalysis:
    scenario: object
    times: tuple
    dur: int
    seed: int
    rule: str
    prio_k: int
    online: object
    greedy: object
    ranking: object
    offline: dict       # {"served", "cost", "routes"}


def analyse_dynamic(sc, dur=C.DEFAULT_DUR, seed=0, rule=RULE_GREEDY, prio_k=0, horizon=C.HORIZON):
    ts = tuple(times(sc.m, horizon, seed))
    g = Dy.run_dynamic(sc, ts, dur, "greedy")
    rk = Dy.run_dynamic(sc, ts, dur, "ranking", priority(sc.n, seed, prio_k))
    return DynAnalysis(sc, ts, dur, seed, rule, prio_k, g if rule == RULE_GREEDY else rk, g, rk, Dy.offline_dynamic(sc, ts, dur))


def verdict_dynamic(a):
    r, off = a.online, a.offline
    served_ratio = r.served / off["served"] if off["served"] else None
    per = lambda cost, k: (cost / k) if k else None
    code = NONE if off["served"] == 0 else (OPTIMAL if r.served == off["served"] else FEWER_PAIRS)
    data = {"served": r.served, "opt_served": off["served"], "lost": off["served"] - r.served, "quotient": served_ratio, "cost": r.cost, "opt_cost": off["cost"],
            "deadhead": per(r.cost, r.served), "opt_deadhead": per(off["cost"], off["served"]), "check": Dy.validate(a.scenario, r),
            "rejected": a.scenario.m - r.served}
    return {NONE: "info", OPTIMAL: "success", FEWER_PAIRS: "warning"}[code], code, data


@lru_cache(maxsize=32)
def dynamic_rows(n, m, reach, ballung, dur, seeds, k_draws=10, horizon=C.HORIZON):
    rows = []
    for sd in seeds:
        sc = generate(n, m, reach, ballung, sd)
        ts = tuple(times(m, horizon, sd))
        off = Dy.offline_dynamic(sc, ts, dur)
        if off["served"] == 0:
            continue
        g = Dy.run_dynamic(sc, ts, dur, "greedy")
        rk = [Dy.run_dynamic(sc, ts, dur, "ranking", priority(n, sd, k)) for k in range(k_draws)]
        rows.append({"opt": (off["served"], off["cost"]), "greedy": (g.served, g.cost), "ranking": tuple((r.served, r.cost) for r in rk)})
    return tuple(rows)


def dynamic_stats(n, m, reach, ballung, dur, seeds=C.DIST_SEEDS, k_draws=10):
    rows = dynamic_rows(n, m, reach, ballung, dur, tuple(seeds), k_draws)
    if not rows:
        return {"n_valid": 0}
    qg = [r["greedy"][0] / r["opt"][0] for r in rows]
    qr = [float(np.mean([d[0] for d in r["ranking"]])) / r["opt"][0] for r in rows]
    per = lambda c, k: c / k if k else 0.0
    return {"n_valid": len(rows), "dur": dur, "greedy_q_mean": float(np.mean(qg)), "greedy_q_median": float(np.median(qg)), "greedy_q_worst": min(qg),
            "ranking_q_mean": float(np.mean(qr)), "ranking_q_median": float(np.median(qr)), "ranking_q_worst": min(qr),
            "opt_served_mean": float(np.mean([r["opt"][0] for r in rows])), "greedy_served_mean": float(np.mean([r["greedy"][0] for r in rows])),
            "deadhead_greedy": float(np.mean([per(r["greedy"][1], r["greedy"][0]) for r in rows])), "deadhead_opt": float(np.mean([per(r["opt"][1], r["opt"][0]) for r in rows])),
            "deadhead_ranking": float(np.mean([np.mean([per(d[1], d[0]) for d in r["ranking"]]) for r in rows]))}


def dynamic_sweep(n, m, reach, ballung, durs=C.DUR_SWEEP, seeds=C.DIST_SEEDS):
    return [dict(dynamic_stats(n, m, reach, ballung, d, seeds), dur=d) for d in durs]


def scenario_from_settings(net, n, m, reach, ballung, seed):
    return build(net, n, m, reach, ballung, seed)
