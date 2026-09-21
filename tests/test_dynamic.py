"""Zeitdimension: Fahrzeuge werden nach D Minuten wieder frei. Der Offline-Maßstab (Min-Cost-Flow) gegen Brute Force, networkx und die statische Ungarische Methode."""

import itertools

import networkx as nx
import pytest

import om_constants as C
import om_dynamic as Dy
import om_evaluation as ev
import om_scenario as S
from om_hungarian import hungarian
from om_online import run_online


def brute_force(sc, times, dur):
    """Alle (n+1)^m Zuordnungen: jedes Fahrzeug fährt seine Aufträge in Zeitreihenfolge, zwei aufeinanderfolgende mit Abstand >= D, Reichweite vom letzten Ort. Ergebnis (bediente, Kosten)."""
    n, m = sc.n, sc.m
    best = (0, 0)
    for assign in itertools.product(range(-1, n), repeat=m):
        ok, cost, served = True, 0, 0
        loc, last = list(sc.vehicles), [None] * n
        for j, i in enumerate(assign):
            if i < 0:
                continue
            if last[i] is not None and times[j] < last[i] + dur:
                ok = False
                break
            c, d2 = S.travel_cost(loc[i][0] - sc.orders[j][0], loc[i][1] - sc.orders[j][1])
            if d2 > sc.reach ** 2:
                ok = False
                break
            cost += c
            served += 1
            loc[i], last[i] = sc.orders[j], times[j]
        if ok and (served, -cost) > (best[0], -best[1]):
            best = (served, cost)
    return best


def networkx_flow(sc, times, dur):
    n, m = sc.n, sc.m
    L = 1 + m * Dy.MAX_LEG
    g = nx.DiGraph()
    g.add_node("s", demand=-n)
    g.add_node("t", demand=n)
    for i in range(n):
        g.add_edge("s", ("v", i), capacity=1, weight=0)
        g.add_edge(("v", i), "t", capacity=1, weight=0)
        for j in range(m):
            c, d2 = S.travel_cost(sc.vehicles[i][0] - sc.orders[j][0], sc.vehicles[i][1] - sc.orders[j][1])
            if d2 <= sc.reach ** 2:
                g.add_edge(("v", i), ("in", j), capacity=1, weight=c)
    for j in range(m):
        g.add_edge(("in", j), ("out", j), capacity=1, weight=-L)
        g.add_edge(("out", j), "t", capacity=1, weight=0)
        for k in range(m):
            if k != j and times[k] >= times[j] + dur:
                c, d2 = S.travel_cost(sc.orders[j][0] - sc.orders[k][0], sc.orders[j][1] - sc.orders[k][1])
                if d2 <= sc.reach ** 2:
                    g.add_edge(("out", j), ("in", k), capacity=1, weight=c)
    flow = nx.min_cost_flow(g)
    served = sum(flow[("in", j)][("out", j)] for j in range(m))
    return served, nx.cost_of_flow(g, flow) + served * L


def test_offline_flow_equals_brute_force_on_300_tiny_instances():
    for sd in range(300):
        n, m = 2 + sd % 2, 4 if sd % 3 else 5
        sc = S.generate(n, m, 30 + 10 * (sd % 5), 0, 500 + sd)
        ts = S.times(m, 30, 500 + sd)
        dur = 1 + (sd * 7) % 25
        off = Dy.offline_dynamic(sc, ts, dur)
        assert (off["served"], off["cost"]) == brute_force(sc, ts, dur), sd


def test_offline_flow_equals_networkx_on_medium_maps():
    for sd in C.DIST_SEEDS[:12]:
        sc = S.generate(8, 20, 40, 0, sd)
        ts = S.times(20, C.HORIZON, sd)
        for dur in (10, 30, 60):
            off = Dy.offline_dynamic(sc, ts, dur)
            assert (off["served"], off["cost"]) == networkx_flow(sc, ts, dur), (sd, dur)


def test_the_routes_of_the_flow_are_feasible_and_reproduce_served_and_cost():
    for sd in C.DIST_SEEDS[:20]:
        sc = S.generate(8, 24, 40, 0, sd)
        ts = S.times(24, C.HORIZON, sd)
        off = Dy.offline_dynamic(sc, ts, 25)
        seen, cost = set(), 0
        for i, chain in off["routes"].items():
            loc, last = sc.vehicles[i], None
            for j in chain:
                assert j not in seen and (last is None or ts[j] >= last + 25)
                c, d2 = S.travel_cost(loc[0] - sc.orders[j][0], loc[1] - sc.orders[j][1])
                assert d2 <= sc.reach ** 2
                cost += c
                seen.add(j)
                loc, last = sc.orders[j], ts[j]
        assert (len(seen), cost) == (off["served"], off["cost"])


def test_a_long_busy_time_is_the_static_model():
    """D > Horizont: kein Fahrzeug wird wieder frei. Fluss = Ungarische Methode auf der ganzen Karte, Regeln = statische Regeln in der Ankunftsreihenfolge."""
    for sd in C.DIST_SEEDS[:30]:
        sc = S.generate(10, 30, 40, 0, sd)
        ts = S.times(30, C.HORIZON, sd)
        D = C.HORIZON + 1
        h = hungarian(sc, record=False)
        off = Dy.offline_dynamic(sc, ts, D)
        assert (off["served"], off["cost"]) == (h.count, h.cost)
        assert sorted(Dy.run_dynamic(sc, ts, D, "greedy").pairs) == sorted(run_online(sc, list(range(30)), "greedy").pairs)
        prio = S.priority(10, sd, 0)
        assert sorted(Dy.run_dynamic(sc, ts, D, "ranking", prio).pairs) == sorted(run_online(sc, list(range(30)), "ranking", prio=prio).pairs)


def test_the_online_rules_never_beat_the_offline_optimum_and_pass_the_checker():
    for sd in C.DIST_SEEDS[:30]:
        sc = S.generate(10, 30, 40, 0, sd)
        ts = S.times(30, C.HORIZON, sd)
        for dur in (1, 20, 60):
            off = Dy.offline_dynamic(sc, ts, dur)
            for rule in ("greedy", "ranking"):
                r = Dy.run_dynamic(sc, ts, dur, rule, S.priority(10, sd, 1))
                assert r.served <= off["served"] and Dy.validate(sc, r)["all_ok"]


def test_the_prefix_optimum_grows_with_the_number_of_arrivals():
    sc = S.generate(10, 30, 40, 0, 100005)
    ts = S.times(30, C.HORIZON, 100005)
    vals = [Dy.offline_dynamic(sc, ts, 20, k)["served"] for k in range(0, 31)]
    assert vals == sorted(vals) and vals[-1] == Dy.offline_dynamic(sc, ts, 20)["served"]


def test_checker_notices_a_vehicle_that_is_still_busy():
    sc = S.generate(10, 30, 40, 0, 100007)
    ts = S.times(30, C.HORIZON, 100007)
    r = Dy.run_dynamic(sc, ts, 60, "greedy")
    assert Dy.validate(sc, r)["all_ok"]
    checked = Dy.validate(sc, Dy.DynOnline(r.rule, 200, r.times, r.prio, r.pairs, r.cost, r.events))          # dieselben Entscheidungen, aber Belegung 200: die Wiederverwendung war nicht erlaubt
    reuse = any(a[0] == b[0] for a, b in itertools.combinations(r.pairs, 2))
    assert reuse and not checked["all_ok"]


def test_negative_control_delayed_batching_can_beat_the_immediate_benchmark():
    """Wartet man mit der Zuordnung auf das Fensterende, kann ein inzwischen freies Fahrzeug einen Auftrag noch nehmen - der Sofort-Maßstab kennt das nicht (Quotient über 1). Deshalb gibt es im dynamischen Modell kein Batching."""
    better = 0
    for sd in C.DIST_SEEDS[:60]:
        sc = S.generate(10, 30, 40, 0, sd)
        ts = S.times(30, C.HORIZON, sd)
        dur, w = 20, 5
        off = Dy.offline_dynamic(sc, ts, dur)
        loc, free_at, served = list(sc.vehicles), [0] * sc.n, 0
        for s in range(0, sc.m, w):
            batch = list(range(s, min(s + w, sc.m)))
            t = ts[batch[-1]]
            free = [i for i in range(sc.n) if free_at[i] <= t]
            if not free:
                continue
            sub = S.from_points([loc[i] for i in free], [sc.orders[j] for j in batch], sc.reach)
            if not sub.feasible.any():
                continue
            for a, b in hungarian(sub, record=False).pairs:
                i, j = free[a], batch[b]
                loc[i], free_at[i] = sc.orders[j], t + dur
                served += 1
        better += served > off["served"]
    assert better > 0


def test_analysis_and_verdict_of_the_dynamic_model():
    sc = S.generate(10, 30, 40, 0, 100003)
    a = ev.analyse_dynamic(sc, 20, 100003)
    level, code, d = ev.verdict_dynamic(a)
    assert d["check"]["all_ok"] and d["served"] <= d["opt_served"] and level in ("success", "warning") and code in (ev.OPTIMAL, ev.FEWER_PAIRS)
    assert d["quotient"] == pytest.approx(d["served"] / d["opt_served"])
