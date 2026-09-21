"""Online-Regeln (Greedy, Ranking, Batching) gegen unabhängige Nachbauten, Brute Force, Identitäten und Negativkontrollen. Die Treppe hat eigene Tests unten."""

import itertools
from fractions import Fraction

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

import om_constants as C
import om_evaluation as ev
import om_scenario as S
from om_hungarian import hungarian
from om_online import run_online, state_at, validate

SEEDS = C.DIST_SEEDS[:30]


def _maps(n=12, m=12, reach=40):
    for sd in SEEDS:
        yield sd, S.generate(n, m, reach, 0, sd)


# --- unabhängige Nachbauten -----------------------------------------------------------------------------------------------------

def sim(sc, order, key):
    """Nachbau von Greedy/Ranking: `key(i, j)` = Sortierschlüssel, das kleinste mögliche freie Fahrzeug gewinnt."""
    free, pairs = set(range(sc.n)), []
    for j in order:
        cands = [i for i in free if sc.feasible[i, j]]
        if cands:
            i = min(cands, key=lambda i: key(i, j))
            free.discard(i)
            pairs.append((i, j))
    return pairs


def opt_scipy(sc, vs, os_):
    """Lexikografisches Optimum (Paare, Kosten) auf den Fahrzeugen vs und den Aufträgen os_ über scipy (Big-M)."""
    if not vs or not os_:
        return 0, 0
    big = 10_000
    a = np.array([[sc.cost[i, j] if sc.feasible[i, j] else big for j in os_] for i in vs])
    r, c = linear_sum_assignment(a)
    real = [(i, j) for i, j in zip(r, c) if a[i, j] < big]
    return len(real), int(sum(a[i, j] for i, j in real))


def brute_max_matching(sc):
    best = 0
    for k in range(min(sc.n, sc.m), 0, -1):
        for js in itertools.combinations(range(sc.m), k):
            for iv in itertools.permutations(range(sc.n), k):
                if all(sc.feasible[i, j] for i, j in zip(iv, js)):
                    return k
    return best


@pytest.mark.parametrize("arr", list(C.ARR_LABELS))
def test_greedy_and_ranking_match_independent_simulations(arr):
    for sd, sc in _maps():
        order = S.arrival_order(sc, arr, sd)
        g = run_online(sc, order, "greedy")
        assert list(g.pairs) == sim(sc, order, lambda i, j: (int(sc.cost[i, j]), i)), (arr, sd)
        prio = S.priority(sc.n, sd, 3)
        r = run_online(sc, order, "ranking", prio=prio)
        assert list(r.pairs) == sim(sc, order, lambda i, j: (prio[i], i)), (arr, sd)
        assert validate(sc, g)["all_ok"] and validate(sc, r)["all_ok"]


@pytest.mark.parametrize("w", [1, 2, 3, 5, 100])
def test_batching_every_window_is_optimal_for_the_free_vehicles_at_that_moment(w):
    for sd, sc in _maps():
        order = S.arrival_order(sc, "random", sd)
        b = run_online(sc, order, "batch", w=w)
        used, pending = set(), []
        for e in b.events:
            if e.kind == "wait":
                pending.append(e.order)
                continue
            batch, pending = pending + [e.order], []
            free = [i for i in range(sc.n) if i not in used]
            assert (len(e.assigned), sum(int(sc.cost[i, j]) for i, j in e.assigned)) == opt_scipy(sc, free, batch), (w, sd)
            used |= {i for i, _ in e.assigned}
        assert validate(sc, b)["all_ok"]


def test_the_decisions_are_never_revised():
    """Die Paare wachsen nur: der Zustand nach k+1 Ankünften enthält den nach k als Teilmenge."""
    for sd, sc in _maps():
        order = S.arrival_order(sc, "random", sd)
        for rule, kw in (("greedy", {}), ("ranking", {"prio": S.priority(sc.n, sd, 0)}), ("batch", {"w": 4})):
            res = run_online(sc, order, rule, **kw)
            prev = set()
            for k in range(len(order) + 1):
                pairs, waiting, rejected = state_at(res, k)
                assert prev <= set(pairs)
                assert not set(waiting) & {j for _, j in pairs} and not set(waiting) & set(rejected)
                prev = set(pairs)
            assert set(state_at(res, len(order))[0]) == set(res.pairs)


# --- Brute Force auf winzigen Karten --------------------------------------------------------------------------------------------

def test_offline_optimum_equals_brute_force_and_greedy_is_a_maximal_matching_within_a_factor_two():
    for sd in range(300, 340):
        sc = S.generate(4, 4, 45, 0, sd)
        best = brute_max_matching(sc)
        assert hungarian(sc, record=False).count == best
        for order in itertools.permutations(range(4)):
            g = run_online(sc, order, "greedy")
            assert 2 * g.count >= best and g.count <= best
            free = {i for i in range(sc.n)} - {i for i, _ in g.pairs}
            unmatched = set(range(sc.m)) - {j for _, j in g.pairs}
            assert not any(sc.feasible[i, j] for i in free for j in unmatched)      # maximal: kein Auftrag ist trotz freiem möglichen Fahrzeug abgelehnt


def test_greedy_pairs_are_at_least_half_the_optimum_on_all_hundred_maps_under_all_arrival_models():
    for arr in C.ARR_LABELS:
        for sd in C.DIST_SEEDS:
            sc = S.generate(20, 20, 40, 0, sd)
            opt = hungarian(sc, record=False).count
            assert 2 * run_online(sc, S.arrival_order(sc, arr, sd), "greedy").count >= opt, (arr, sd)
    for sd in C.SWEEP_SEEDS[:10]:
        sc = S.generate(20, 20, 40, 0, sd)
        assert 2 * run_online(sc, ev.worst_order(sc, sd), "greedy").count >= hungarian(sc, record=False).count


# --- Identitäten ------------------------------------------------------------------------------------------------------------------

def test_a_window_of_all_orders_is_the_offline_optimum():
    for sd, sc in _maps(12, 12, 40):
        h = hungarian(sc, record=False)
        for order in (list(range(sc.m)), S.arrival_order(sc, "random", sd)):
            b = run_online(sc, order, "batch", w=sc.m)
            assert (b.count, b.cost) == (h.count, h.cost)
            assert b.scanned_total == h.scanned_total or order != list(range(sc.m))
        assert sorted(run_online(sc, list(range(sc.m)), "batch", w=sc.m).pairs) == sorted(h.pairs)
        assert run_online(sc, list(range(sc.m)), "batch", w=sc.m + 5).count == h.count       # jedes größere Fenster ebenso


def test_a_window_of_one_is_greedy():
    for sd, sc in _maps(12, 12, 40):
        order = S.arrival_order(sc, "random", sd)
        assert sorted(run_online(sc, order, "batch", w=1).pairs) == sorted(run_online(sc, order, "greedy").pairs)


def test_ranking_with_the_identity_as_ranking_is_greedy_by_index():
    for sd, sc in _maps():
        order = S.arrival_order(sc, "random", sd)
        assert list(run_online(sc, order, "ranking", prio=list(range(sc.n))).pairs) == sim(sc, order, lambda i, j: i)


def test_ranking_priorities_are_permutations_and_reproducible():
    for sd in (1, 2, 3):
        for k in range(4):
            p = S.priority(15, sd, k)
            assert sorted(p) == list(range(15)) and p == S.priority(15, sd, k)
    assert S.priority(15, 1, 0) != S.priority(15, 1, 1) and S.priority(15, 1, 0) != S.priority(15, 2, 0)


# --- die Treppe -------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("n", [2, 3, 4, 5, 8, 10, 12])
def test_greedy_gets_only_half_of_the_stairs_without_any_tie_break(n):
    sc = S.staircase(n)
    assert [int(x) for x in sc.feasible.sum(axis=0)] == list(range(n, 0, -1))                 # Auftrag t erreicht die Fahrzeuge 0..n-t-1
    for t in range(n):                                                                       # die Kosten steigen streng mit dem Fahrzeugindex: es gibt nie einen Gleichstand
        cs = [int(sc.cost[i, t]) for i in range(n - t)]
        assert all(a < b for a, b in zip(cs, cs[1:]))
    assert hungarian(sc, record=False).count == n
    g = run_online(sc, list(range(n)), "greedy")
    assert g.count == -(-n // 2)
    assert run_online(sc, list(range(n)), "greedy", tie="high").count == -(-n // 2)         # ein umgekehrter Tie-Break hilft hier nicht
    assert run_online(sc, list(range(n)), "ranking", prio=list(range(n))).count == -(-n // 2)   # feste Rangfolge nach Index: dasselbe
    assert run_online(sc, list(range(n))[::-1], "greedy").count == n                          # starre zuerst: perfekt


def test_stairs_fit_the_map_and_the_abstract_graph_has_the_same_structure():
    for n in (8, 12):
        sc = S.staircase(n)
        pts = sc.vehicles + sc.orders
        assert all(0 <= x <= 100 and 0 <= y <= 100 for x, y in pts)
        assert (S.staircase_graph(n).feasible == sc.feasible).all()
    with pytest.raises(ValueError):
        S.staircase(S.STAIRCASE_MAX_N + 1)


def test_on_an_abstract_graph_with_equal_costs_only_the_tie_break_decides():
    """Die obere Dreiecksmatrix mit gleichen Kosten: kleinster Index wählt die flexibelsten Fahrzeuge (n/2), größter die starren (alle n) - deshalb ist die Falle geometrisch gebaut."""
    n = 8
    g = S.staircase_graph(n)
    flat = S.Scenario(g.vehicles, g.orders, g.reach, np.ones((n, n), dtype=np.int64), g.feasible)
    assert run_online(flat, list(range(n)), "greedy", tie="low").count == n // 2
    assert run_online(flat, list(range(n)), "greedy", tie="high").count == n


def test_exact_expected_pairs_of_ranking_on_the_stairs():
    """E[Ranking] = Mittel über alle n! Rangfolgen, hier mit dem echten Ranking auf der geometrischen Treppe nachgerechnet."""
    for n in (3, 4, 5, 6):
        sc = S.staircase(n)
        total = sum(run_online(sc, list(range(n)), "ranking", prio=list(p)).count for p in itertools.permutations(range(n)))
        assert Fraction(total, len(list(itertools.permutations(range(n))))) == ev.staircase_exact(n)
    assert [str(ev.staircase_exact(n)) for n in (3, 4, 5)] == ["13/6", "67/24", "137/40"]
    assert [round(float(ev.staircase_exact(n)), 4) for n in (6, 7, 8)] == [4.0569, 4.6891, 5.3212]


def test_ranking_on_the_stairs_approaches_one_minus_one_over_e_from_above():
    vals = {n: ev.staircase_mc(n)[0] / n for n in (10, 20, 40, 80, 160)}
    assert [round(vals[n], 4) for n in (10, 20, 40, 80, 160)] == [0.6548, 0.6454, 0.6398, 0.6358, 0.6334]
    assert all(v >= 1 - 1 / np.e - 0.002 for v in vals.values())
    assert all(vals[a] > vals[b] for a, b in ((10, 20), (20, 40), (40, 80), (80, 160)))
    assert all(round(ev.staircase_exact(n) / n, 3) > 1 - 1 / np.e for n in range(3, 9))


# --- Negativkontrollen ------------------------------------------------------------------------------------------------------------

def _augmenting_online(sc, order):
    """Umhängen erlaubt: nach jeder Ankunft sucht ein Verbesserungsweg (Kuhn) ab dem neuen Auftrag. Nachbau, ohne om_augment."""
    match_of_v, seen_orders = {}, []

    def try_order(j, visited):
        for i in range(sc.n):
            if sc.feasible[i, j] and i not in visited:
                visited.add(i)
                if i not in match_of_v or try_order(match_of_v[i], visited):
                    match_of_v[i] = j
                    return True
        return False
    counts = []
    for j in order:
        try_order(j, set())
        counts.append(len(match_of_v))
    return counts


def test_negative_control_allowing_reassignment_reaches_the_optimum_under_every_arrival_order():
    """Nicht die Reihenfolge ist das Problem, sondern die Unwiderruflichkeit: dieselben Ankünfte mit Umhängen liefern immer die größte Paarzahl."""
    for arr in C.ARR_LABELS:
        gaps = 0
        for sd in C.DIST_SEEDS[:30]:
            sc = S.generate(20, 20, 40, 0, sd)
            order = S.arrival_order(sc, arr, sd)
            counts = _augmenting_online(sc, order)
            assert counts[-1] == hungarian(sc, record=False).count
            assert all(b - a in (0, 1) for a, b in zip(counts, counts[1:]))
            gaps += run_online(sc, order, "greedy").count < counts[-1]
        assert gaps > 0                                                                       # und die unwiderrufliche Greedy-Regel bleibt auf einigen Karten dahinter
    sc = S.staircase(8)
    assert _augmenting_online(sc, list(range(8)))[-1] == 8 > run_online(sc, list(range(8)), "greedy").count


def test_negative_control_fixed_index_priority_is_no_better_than_greedy_on_the_worst_order_of_the_stairs():
    for n in (4, 8, 12):
        sc = S.staircase(n)
        assert run_online(sc, list(range(n)), "ranking", prio=list(range(n))).count == run_online(sc, list(range(n)), "greedy").count == -(-n // 2)
        reverse = list(range(n))[::-1]
        assert run_online(sc, list(range(n)), "ranking", prio=reverse).count == n           # die "richtige" feste Rangfolge (starre Fahrzeuge zuerst) wäre perfekt - man kennt sie nur nicht vorab
