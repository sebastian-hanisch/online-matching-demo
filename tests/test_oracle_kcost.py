"""Orakel für die Bezugsgröße der Prämie: `kcost[k]` (Summe der Grenzkosten der Ungarischen Methode) muss die Kosten der billigsten Paarung
mit genau k Paaren sein - hier durch reine Aufzählung aller k-Paarungen auf winzigen Karten, ohne Potenziale und ohne Dijkstra."""

import itertools

import om_scenario as S
from om_evaluation import _kcost, premium_of
from om_hungarian import hungarian


def _cheapest_with_k_pairs(sc, k):
    best = None
    for js in itertools.combinations(range(sc.m), k):
        for iv in itertools.permutations(range(sc.n), k):
            if all(sc.feasible[i, j] for i, j in zip(iv, js)):
                c = sum(int(sc.cost[i, j]) for i, j in zip(iv, js))
                best = c if best is None else min(best, c)
    return best


def test_kcost_is_the_cheapest_matching_with_exactly_k_pairs():
    checked = 0
    for sd in range(500, 560):
        n, m = 2 + sd % 4, 2 + (sd // 4) % 4
        sc = S.generate(n, m, (25, 40, 60)[sd % 3], 0, sd)
        h = hungarian(sc, record=False)
        kc = _kcost(h)
        assert len(kc) == h.count + 1 and kc[0] == 0 and kc[-1] == h.cost
        for k in range(1, h.count + 1):
            assert kc[k] == _cheapest_with_k_pairs(sc, k), (sd, k)
            checked += 1
        assert _cheapest_with_k_pairs(sc, h.count + 1) is None          # h.count ist wirklich das Maximum
    assert checked > 100


def test_premium_of_the_cheapest_matching_is_zero():
    sc = S.generate(5, 5, 40, 0, 521)
    h = hungarian(sc, record=False)
    assert premium_of(h.cost, h.count, _kcost(h)) == 0.0
