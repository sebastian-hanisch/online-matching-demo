"""Karten, Zufallsströme, Wächter der kopierten Bausteine und die Plumbing der Auswertung (Verdict, Tabellen, Verteilungen, Sweeps)."""

import dataclasses

import pytest

import om_constants as C
import om_evaluation as ev
import om_scenario as S
from om_augment import augment
from om_greedy import optimum, run_rule
from om_hungarian import certificate, hungarian
from om_online import run_online


def test_splitmix_vector_and_the_seed_2_map_of_the_predecessors():
    rng = S.SplitMix64(0)
    assert [rng.next() for _ in range(2)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4]
    sc = S.generate(20, 20, 40, 0, 2)
    g, o = run_rule(sc, "edge"), optimum(sc)
    assert (g.count, g.cost, o.count, o.cost) == (17, 232, 20, 316)                            # Wächter: die kopierten Bausteine rechnen wie in den Vorgängerdemos
    h = hungarian(sc, record=False)
    assert (h.count, h.cost) == (o.count, o.cost) and certificate(sc, hungarian(sc))["all_ok"]
    assert augment(sc).count == o.count


def test_streams_are_deterministic_permutations():
    for m in (1, 5, 20):
        for seed in (0, 7, 100003):
            assert S.perm(m, seed) == S.perm(m, seed) and sorted(S.perm(m, seed)) == list(range(m))
    sc = S.generate(20, 20, 40, 0, 5)
    assert S.arrival_order(sc, "random", 5) == S.arrival_order(sc, "random", 5) != S.arrival_order(sc, "random", 6)
    assert S.arrival_order(sc, "random", 5) != S.perm(20, 5)                                   # der Ankunftsstrom ist vom Kartenstrom getrennt
    ts = S.times(30, 120, 9)
    assert ts == sorted(ts) and len(ts) == 30 and min(ts) >= 0 and max(ts) <= 120 and ts == S.times(30, 120, 9)


def test_arrival_kinds():
    sc = S.generate(20, 20, 40, 0, 11)
    deg = [S.degree(sc, j) for j in range(sc.m)]
    flex, rigid = S.arrival_order(sc, "flex"), S.arrival_order(sc, "rigid")
    assert [deg[j] for j in flex] == sorted(deg, reverse=True) and [deg[j] for j in rigid] == sorted(deg)
    sweep = S.arrival_order(sc, "sweep")
    assert [sc.orders[j][0] for j in sweep] == sorted(x for x, _ in sc.orders)
    assert S.arrival_order(sc, "index") == list(range(20))
    for kind in S.ARRIVALS:
        assert sorted(S.arrival_order(sc, kind, 3)) == list(range(20))
    with pytest.raises(ValueError):
        S.arrival_order(sc, "no-such-kind")


def test_build_fixed_nets_ignore_random_parameters():
    a, b = S.build("stairs", 5, 5, 99, 100, 123), S.build("stairs", 30, 30, 10, 0, 1)
    assert a.vehicles == b.vehicles and a.n == 8 == a.m
    p4 = S.build("p4", 1, 1, 1, 1, 1)
    assert p4.n == 2 == p4.m
    assert S.build("random", 7, 9, 40, 0, 3).n == 7


def test_p4_is_the_two_way_trap_in_the_small():
    """O1 (flexibel) kommt zuerst und nimmt V2, das O2 allein hätte erreichen können: Greedy 1, Optimum 2; in der anderen Reihenfolge ist Greedy perfekt."""
    sc = S.p4_chain(1)
    assert run_online(sc, [0, 1], "greedy").count == 1 and hungarian(sc, record=False).count == 2
    assert run_online(sc, [1, 0], "greedy").count == 2


def test_verdict_of_the_fixed_nets():
    a = ev.analyse(S.build("stairs", 20, 20, 40, 0, 2), "flex", "greedy", 5, 2)
    level, code, d = ev.verdict(a)
    assert (level, code) == ("warning", ev.FEWER_PAIRS) and (d["count"], d["opt_count"], d["lost"]) == (4, 8, 4) and d["quotient"] == 0.5 and d["check"]["all_ok"]
    a = ev.analyse(S.build("stairs", 20, 20, 40, 0, 2), "rigid", "greedy", 5, 2)
    level, code, d = ev.verdict(a)
    assert d["count"] == 8 and code in (ev.OPTIMAL, ev.WORSE_COST)
    a = ev.analyse(S.build("p4", 1, 1, 1, 1, 1), "flex", "batch", 2, 1)
    level, code, d = ev.verdict(a)
    assert (level, code) == ("success", ev.OPTIMAL) and (d["count"], d["opt_count"]) == (2, 2)                # ein Fenster von 2 sieht beide Aufträge


def test_verdict_none_without_any_feasible_edge():
    sc = next(s for s in (S.generate(3, 3, 10, 0, k) for k in range(200)) if not s.feasible.any())
    level, code, d = ev.verdict(ev.analyse(sc, "random", "greedy", 5, 0))
    assert (level, code) == ("info", ev.NONE) and d["count"] == 0 and d["quotient"] is None


def test_verdict_codes_optimal_worse_cost_fewer_pairs():
    seen = set()
    for sd in C.DIST_SEEDS[:40]:
        seen.add(ev.verdict(ev.analyse(S.generate(20, 20, 40, 0, sd), "random", "greedy", 5, sd))[1])
    assert seen == {ev.OPTIMAL, ev.WORSE_COST, ev.FEWER_PAIRS} or seen == {ev.WORSE_COST, ev.FEWER_PAIRS}


def test_premium_is_measured_against_the_cheapest_matching_with_the_same_number_of_pairs():
    a = ev.analyse(S.generate(20, 20, 40, 0, 97), "random", "greedy", 5, 97)
    k = a.online.count
    assert a.kcost[a.hung.count] == a.hung.cost and a.kcost[k] <= a.online.cost
    assert ev.premium_of(a.online.cost, k, a.kcost) > 0
    assert ev.premium_of(a.hung.cost, a.hung.count, a.kcost) == 0.0


def test_compare_table_rows():
    a = ev.analyse(S.generate(20, 20, 40, 0, 97), "random", "ranking", 5, 97, prio_k=2)
    rows = ev.compare_table(a)
    assert [r["label"] for r in rows] == ["Greedy: das nächste freie Fahrzeug", "Ranking (Rangfolge Nr. 3)", "Batching, Fenster 5", "Offline-Optimum (Ungarische Methode)"]
    assert rows[-1]["quotient"] == 1.0 and rows[-1]["premium_pct"] == 0.0 and all(0 < r["quotient"] <= 1 for r in rows)
    assert a.online is a.ranking


def test_prefix_series_ends_at_the_final_counts():
    a = ev.analyse(S.generate(15, 15, 40, 0, 97), "random", "greedy", 5, 97)
    online, best = ev.prefix_series(a)
    assert online[0] == best[0] == 0 and online[-1] == a.online.count and best[-1] == a.hung.count and len(online) == len(best) == 16
    assert all(o <= b for o, b in zip(online, best)) and best == sorted(best)


def test_distribution_fields_and_shapes():
    d = ev.distribution(20, 20, 40, 0, "random", 5)
    assert d["n_seeds"] == 100 == d["n_valid"]
    for rule in ("greedy", "ranking", "index", "batch", "batch_all"):
        s = d[rule]
        assert 0 < s["q_worst"] <= s["q_median"] <= 1 and 0 < s["q_mean"] <= 1 and len(s["q"]) == 100
    assert d["batch_all"]["q_mean"] == 1.0 and d["batch_all"]["premium_mean"] == 0.0


def test_distribution_is_repeatable_and_without_feasible_pairs():
    assert ev.distribution(12, 12, 40, 25, "random", 5) == ev.distribution(12, 12, 40, 25, "random", 5)
    bad = tuple(s for s in range(60) if not S.generate(3, 3, 10, 0, s).feasible.any())
    assert len(bad) >= 3
    assert ev.distribution(3, 3, 10, 0, "random", 5, seeds=bad)["n_valid"] == 0


def test_cell_rows_are_integers():
    rows = ev.cell_rows(10, 10, 40, 0, "random", tuple(C.SWEEP_SEEDS[:5]), (1, 5, 10), 3)
    for row in rows:
        assert all(isinstance(x, int) for x in row["opt"]) and all(isinstance(x, int) for x in row["kc"])
        assert all(isinstance(x, int) for x in row["greedy"][:3]) and all(isinstance(x, int) for d in row["ranking"] for x in d)


def test_window_sweep_shape():
    rows = ev.window_sweep(12, 12, 40, 0, "random", seeds=C.SWEEP_SEEDS[:10], windows=(1, 2, 5))
    assert [r["w"] for r in rows] == [1, 2, 5, 12] and rows[-1]["q_mean"] == 1.0 and rows[0]["fewer_than_prev"] is None
    assert all(r["scanned"] > 0 for r in rows)


def test_arrival_comparison_shape():
    rows = ev.arrival_comparison(12, 12, 40, 0, seeds=C.SWEEP_SEEDS[:6])
    assert [r["arr"] for r in rows] == ["random", "flex", "rigid", "sweep", "worst"]
    assert rows[4]["greedy"]["q_mean"] <= rows[0]["greedy"]["q_mean"]                          # die schlechteste gefundene Reihenfolge ist nicht besser als eine zufällige


def test_worst_order_is_a_permutation_and_never_better_than_its_start():
    for sd in C.SWEEP_SEEDS[:5]:
        sc = S.generate(20, 20, 40, 0, sd)
        wo = ev.worst_order(sc, sd)
        assert sorted(wo) == list(range(20))
        assert run_online(sc, wo, "greedy").key() <= run_online(sc, S.arrival_order(sc, "random", sd), "greedy").key()


def test_staircase_tables():
    rows = ev.staircase_curve((3, 8, 20))
    assert [r["n"] for r in rows] == [3, 8, 20] and rows[0]["exact"] and not rows[2]["exact"] and [r["greedy"] for r in rows] == [2, 4, 10]
    sa = ev.staircase_arrivals(12, w_list=(1, 2), k_draws=10)
    assert sa["flex"]["greedy"] == 6 and sa["rigid"]["greedy"] == 12 and sa["batch"][12] == 12 and sa["batch"][1] == sa["batch"][2] == 6


def test_scan_table_shape():
    rows = ev.scan_table(12, 12, 40, 0, seeds=C.SWEEP_SEEDS[:5])
    assert rows[0]["label"] == "Greedy" and rows[-1]["label"].startswith("Offline") and all(r["scanned"] > 0 for r in rows)
    assert rows[-2]["scanned"] == pytest.approx(rows[-1]["scanned"], rel=0.03)                # das Fenster "alles" ist die Offline-Rechnung (nur die Spaltenreihenfolge im Heap ändert wenige Kanten)
    same = ev.scan_table(12, 12, 40, 0, arr="index", seeds=C.SWEEP_SEEDS[:5])
    assert same[-2]["scanned"] == same[-1]["scanned"]                                            # in der Nummerierung als Ankunftsreihenfolge sogar genau


def test_dynamic_stats_and_sweep_shape():
    s = ev.dynamic_stats(8, 20, 40, 0, 30, seeds=C.SWEEP_SEEDS[:8])
    assert s["n_valid"] > 0 and 0 < s["greedy_q_worst"] <= s["greedy_q_median"] <= 1
    rows = ev.dynamic_sweep(8, 20, 40, 0, durs=(10, 121), seeds=C.SWEEP_SEEDS[:8])
    assert [r["dur"] for r in rows] == [10, 121]


def test_the_verdict_flags_a_tampered_result():
    a = ev.analyse(S.generate(20, 20, 40, 0, 97), "random", "greedy", 5, 97)
    broken = dataclasses.replace(a.online, pairs=a.online.pairs[1:])
    assert not ev.verdict(dataclasses.replace(a, online=broken))[2]["check"]["all_ok"]
