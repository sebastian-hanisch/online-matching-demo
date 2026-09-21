"""Jede Zahl in den Hilfetexten, Presets, Tabellen und Grenzen der App und der README ist hier über die 100 festen Karten (DIST_SEEDS) belegt.
Alles rechnet mit ganzen Zahlen und einem eigenen Zufallsgenerator - die Werte sind auf jeder Plattform dieselben; die Toleranzen decken nur die Rundung auf die im Text genannten Stellen.
Positive UND negative Aussagen: wo Greedy fast nichts verliert (kurze Reichweite, alles erreichbar) und wo viel (flexible zuerst, die Treppe). Mittel und Median stehen zusammen."""

import pytest

import om_evaluation as ev


def near(value, expected, tol):
    assert abs(value - expected) <= tol, f"{value:.4f} statt {expected}"


@pytest.fixture(scope="module")
def r40():
    return ev.distribution(20, 20, 40, 0, "random", 5)


# --- zufällige Karte, zufällige Reihenfolge --------------------------------------------------------------------------------------------

def test_greedy_on_random_maps(r40):
    g = r40["greedy"]
    near(r40["opt_pairs_mean"], 19.46, 0.005)
    near(g["q_mean"], 0.890, 0.0005)                                                           # "im Mittel 89 %"
    near(g["q_median"], 0.900, 0.0005)
    near(g["q_worst"], 0.737, 0.0005)                                                          # "schlechteste Karte 0,74"
    near(g["premium_mean"], 18.8, 0.05)
    near(g["premium_median"], 19.2, 0.05)                                                      # "Median 19,2 %"
    near(g["lost_mean"], 2.13, 0.005)
    near(g["optimal_share"], 0.02, 0.0001)                                                     # nur auf 2 von 100 Karten optimal


def test_ranking_wins_a_little_on_pairs_and_loses_a_lot_on_cost(r40):
    rk, g = r40["ranking"], r40["greedy"]
    near(rk["q_mean"], 0.911, 0.0005)
    near(rk["q_worst"], 0.847, 0.0005)                                                         # die schlechteste Karte ist bei Ranking besser
    near(r40["ranking_minus_greedy_mean"], 0.39, 0.005)                                        # +0,39 Paare
    near(r40["ranking_minus_greedy_median"], 0.35, 0.005)
    near(r40["ranking_worse_share"], 0.27, 0.0001)                                             # auf 27 von 100 Karten schlechter als Greedy
    near(rk["premium_mean"], 71.0, 0.05)
    near(rk["premium_median"], 70.2, 0.05)
    assert rk["premium_median"] > 3 * g["premium_median"]                                     # kostenblind: mehr als dreimal so teuer
    assert rk["q_mean"] > g["q_mean"] and rk["premium_mean"] > g["premium_mean"]


def test_greedy_by_index_is_not_the_same_matching_as_greedy_by_cost(r40):
    ix = r40["index"]
    near(ix["q_mean"], 0.906, 0.0005)
    near(ix["premium_median"], 71.3, 0.05)
    near(r40["greedy_index_same_pairs_share"], 0.40, 0.0001)                                   # gleiche Paarzahl nur auf 40 von 100 Karten
    assert r40["greedy_index_same_matching_share"] == 0.0                                      # dieselbe Paarung auf keiner Karte
    near(r40["tie_share"] * 100, 4.1, 0.05)                                                    # 4,1 % der Entscheidungen haben einen Kostengleichstand
    assert r40["tie_break_pairs_delta"] == 0.0                                                 # der Tie-Break ändert die mittlere Paarzahl nicht ...
    near(r40["tie_break_differ_share"], 0.52, 0.0001)                                          # ... aber die Paarung auf 52 von 100 Karten


def test_batching_window_5(r40):
    b = r40["batch"]
    near(b["q_mean"], 0.906, 0.0005)                                                           # "im Mittel 0,906 (Greedy 0,890)"
    near(b["premium_mean"], 14.6, 0.05)                                                        # "14,6 % Prämie statt 18,8 %"
    assert b["q_worst"] == pytest.approx(0.75) and b["q_mean"] > r40["greedy"]["q_mean"] and b["premium_mean"] < r40["greedy"]["premium_mean"]
    assert r40["batch_all"]["q_mean"] == 1.0 and r40["batch_all"]["premium_mean"] == 0.0        # ein Fenster über alle Aufträge ist das Optimum


def test_window_sweep_is_monotone_on_average_but_not_per_map():
    rows = ev.window_sweep(20, 20, 40, 0, "random")
    assert [r["w"] for r in rows] == [1, 2, 5, 10, 20]
    q = [r["q_mean"] for r in rows]
    assert all(a < b for a, b in zip(q, q[1:]))                                                # im Mittel monoton
    for r, (qm, pm, sc) in zip(rows, ((0.890, 18.8, 144), (0.896, 17.3, 214), (0.906, 14.6, 436), (0.934, 10.9, 836), (1.0, 0.0, 1698))):
        near(r["q_mean"], qm, 0.0006)
        near(r["premium_mean"], pm, 0.05)
        near(r["scanned"], sc, 1)                                                              # angesehene Kanten
    assert [r["fewer_than_prev"] for r in rows] == [None, 0, 8, 2, 0]                          # je Karte nicht: 8 von 100 Karten verlieren von Fenster 2 auf 5, 2 von 5 auf 10
    near(rows[3]["q_median"], 0.9487, 0.0001)
    near(rows[3]["q_worst"], 0.75, 0.0001)


# --- Ankunftsmodelle -------------------------------------------------------------------------------------------------------------------

def test_flexible_first_hurts_greedy_but_nowhere_near_one_half():
    d = ev.distribution(20, 20, 40, 0, "flex", 5)
    g = d["greedy"]
    near(g["q_mean"], 0.832, 0.0005)
    near(g["q_median"], 0.850, 0.0005)
    near(g["q_worst"], 0.650, 0.0005)                                                          # "schlechteste Karte 0,65"
    assert g["q_worst"] > 0.5
    near(g["premium_median"], 23.9, 0.05)
    near(d["ranking"]["q_mean"], 0.850, 0.0006)
    near(d["batch"]["q_mean"], 0.850, 0.0005)


def test_rigid_first_and_left_to_right():
    rigid = ev.distribution(20, 20, 40, 0, "rigid", 5)
    near(rigid["greedy"]["q_mean"], 0.947, 0.0005)
    near(rigid["batch"]["q_mean"], 0.967, 0.0005)
    sweep = ev.distribution(20, 20, 40, 0, "sweep", 5)
    near(sweep["greedy"]["q_mean"], 0.896, 0.0006)                                             # kaum besser als zufällig (0,890) ...
    near(sweep["greedy"]["premium_mean"], 29.9, 0.05)                                          # ... aber teurer (29,9 % gegen 18,8 %)
    near(sweep["ranking"]["q_mean"], 0.880, 0.0006)                                            # Ranking wird schlechter
    near(sweep["batch"]["q_mean"], 0.925, 0.0006)                                              # Batching profitiert
    assert sweep["greedy"]["premium_mean"] > ev.distribution(20, 20, 40, 0, "random", 5)["greedy"]["premium_mean"]


def test_arrival_comparison_on_40_maps():
    rows = {r["arr"]: r for r in ev.arrival_comparison(20, 20, 40, 0)}
    expect = {"random": (0.889, 0.778, 0.910, 0.929), "flex": (0.831, 0.684, 0.846, 0.867), "rigid": (0.945, 0.895, 0.952, 0.981), "sweep": (0.905, 0.750, 0.888, 0.963), "worst": (0.826, 0.737, 0.890, 0.918)}
    for arr, (g, gw, rk, bt) in expect.items():
        near(rows[arr]["greedy"]["q_mean"], g, 0.0006)
        near(rows[arr]["greedy"]["q_worst"], gw, 0.0006)
        near(rows[arr]["ranking"]["q_mean"], rk, 0.0006)
        near(rows[arr]["batch"]["q_mean"], bt, 0.0006)
    assert all(r["greedy"]["q_worst"] > 0.5 for r in rows.values())                            # keine Karte fällt unter 1/2
    near(rows["worst"]["greedy"]["premium_mean"], 55.5, 0.05)


# --- andere Reichweiten und Größen -------------------------------------------------------------------------------------------------

def test_short_reach_greedy_is_almost_always_optimal():
    d = ev.distribution(20, 20, 10, 0, "random", 5)
    near(d["opt_pairs_mean"], 7.28, 0.005)
    near(d["greedy"]["q_mean"], 0.980, 0.0005)
    assert d["greedy"]["q_median"] == 1.0
    near(d["greedy"]["optimal_share"], 0.85, 0.0001)                                           # optimal auf 85 von 100 Karten
    near(d["greedy"]["premium_mean"], 5.8, 0.05)
    near(d["greedy"]["premium_median"], 2.1, 0.05)                                             # "Median 2,1 %"


def test_all_reachable_the_loss_is_pure_cost():
    d = ev.distribution(20, 20, 150, 0, "random", 5)
    assert d["greedy"]["q_mean"] == 1.0 and d["greedy"]["q_worst"] == 1.0 and d["greedy"]["optimal_share"] == 1.0      # auf allen 100 Karten alle 20 Paare
    assert d["ranking"]["q_worst"] == 1.0
    near(d["greedy"]["premium_median"], 21.2, 0.05)
    near(d["greedy"]["premium_mean"], 22.7, 0.05)
    near(d["ranking"]["premium_median"], 160.0, 0.05)                                          # "im Median 160 %"
    near(d["ranking"]["premium_mean"], 164.7, 0.05)


def test_fewer_vehicles_than_orders_and_the_other_way_round():
    d = ev.distribution(10, 20, 40, 0, "random", 5)
    near(d["greedy"]["q_mean"], 0.985, 0.0005)
    assert d["greedy"]["q_median"] == 1.0
    near(d["greedy"]["optimal_share"], 0.87, 0.0001)                                           # alle möglichen Paare auf 87 von 100 Karten
    near(d["greedy"]["premium_median"], 49.3, 0.05)
    near(d["greedy"]["premium_mean"], 49.9, 0.05)
    e = ev.distribution(20, 10, 40, 0, "random", 5)
    near(e["ranking"]["q_mean"], 0.991, 0.0005)                                                # bei mehr Fahrzeugen als Aufträgen schlägt Ranking Greedy bei den Paaren
    near(e["index"]["q_mean"], 0.997, 0.0005)
    near(e["greedy"]["q_mean"], 0.985, 0.0005)
    assert e["ranking"]["q_mean"] > e["greedy"]["q_mean"] and e["index"]["q_mean"] > e["greedy"]["q_mean"]
    near(e["ranking"]["premium_mean"], 87.0, 0.05)
    near(e["index"]["premium_mean"], 85.8, 0.05)
    near(e["greedy"]["premium_mean"], 7.9, 0.05)


# --- die Treppe ------------------------------------------------------------------------------------------------------------------------

def test_staircase_table():
    rows = {r["n"]: r for r in ev.staircase_curve()}
    assert [rows[n]["greedy"] for n in (3, 4, 5, 6, 8, 10, 20, 40, 80, 160)] == [2, 2, 3, 3, 4, 5, 10, 20, 40, 80]
    assert [round(rows[n]["ranking"], 4) for n in (3, 4, 5, 6, 8)] == [2.1667, 2.7917, 3.425, 4.0569, 5.3212]
    near(rows[8]["ranking_ratio"], 0.665, 0.0005)                                              # "0,665"
    assert rows[8]["ranking_ratio"] > 1 - 1 / 2.718281828459045
    near(rows[160]["greedy_ratio"], 0.5, 1e-12)
    assert rows[10]["exact"] is False and rows[8]["exact"] is True


def test_staircase_arrival_models_at_n_20():
    sa = ev.staircase_arrivals(20)
    assert sa["flex"]["greedy"] == 10 and sa["rigid"]["greedy"] == 20 and sa["rigid"]["ranking"] == 20.0
    near(sa["flex"]["ranking"], 12.9, 0.05)
    near(sa["random"]["greedy"], 12.9, 0.05)
    near(sa["random"]["ranking"], 16.3, 0.05)
    assert sa["batch"] == {1: 10, 2: 10, 5: 10, 10: 10, 20: 20}                                # Fenster 1 bis 10 ändern nichts, erst das über alle 20


# --- Aufwand ----------------------------------------------------------------------------------------------------------------------------

def test_effort_in_scanned_edges():
    rows = {r["label"]: r["scanned"] for r in ev.scan_table(20, 20, 40, 0)}
    near(rows["Greedy"], 73.6, 0.05)
    near(rows["Batching, Fenster 1"], 144.5, 0.05)
    near(rows["Batching, Fenster 2"], 215.5, 0.05)
    near(rows["Batching, Fenster 5"], 436.3, 0.05)
    near(rows["Batching, Fenster 10"], 834.7, 0.05)
    near(rows["Batching, Fenster 20 (alles)"], 1676.0, 0.05)
    near(rows["Offline (Ungarische Methode)"], 1685.7, 0.05)
    assert rows["Greedy"] < rows["Batching, Fenster 1"] < rows["Batching, Fenster 5"] < rows["Offline (Ungarische Methode)"]


# --- die Zeitdimension -------------------------------------------------------------------------------------------------------------------

def test_dynamic_loss_is_hump_shaped_over_the_busy_time():
    vals = {D: ev.dynamic_stats(10, 30, 40, 0, D) for D in (1, 10, 20, 30, 60, 121)}
    for D, (mean, med, worst) in {1: (0.988, 1.0, 0.867), 10: (0.952, 0.967, 0.767), 20: (0.915, 0.925, 0.741), 30: (0.922, 0.923, 0.793), 60: (0.946, 0.947, 0.812), 121: (0.998, 1.0, 0.900)}.items():
        near(vals[D]["greedy_q_mean"], mean, 0.0006)
        near(vals[D]["greedy_q_median"], med, 0.0006)
        near(vals[D]["greedy_q_worst"], worst, 0.0006)
    m = [vals[D]["greedy_q_mean"] for D in (1, 10, 20, 30, 60, 121)]
    assert min(m) == m[2] and m[0] > m[1] > m[2] and m[2] < m[3] < m[4] < m[5]                 # Minimum bei D = 20, davor fallend, danach steigend
    near(vals[20]["ranking_q_mean"], 0.918, 0.0006)                                            # Ranking ähnlich
    near(vals[20]["opt_served_mean"], 28.05, 0.005)
    near(vals[20]["greedy_served_mean"], 25.68, 0.005)


def test_dynamic_hump_moves_with_the_load():
    q = {D: ev.dynamic_stats(20, 40, 40, 0, D)["greedy_q_mean"] for D in (10, 20, 30, 40, 50, 60, 90)}
    assert min(q, key=q.get) == 40                                                             # bei 20 Fahrzeugen und 40 Aufträgen liegt das Minimum bei D = 40
    near(q[40], 0.938, 0.0006)
    assert min(ev.dynamic_stats(10, 30, 40, 0, D)["greedy_q_mean"] for D in (10, 20, 30, 40, 50, 60, 90)) < q[40]   # und der Verlust ist kleiner als bei 10 Fahrzeugen und 30 Aufträgen


def test_dynamic_all_reachable_the_loss_moves_into_the_deadhead():
    s = ev.dynamic_stats(10, 30, 150, 0, 60)
    assert s["greedy_q_mean"] == 1.0 and s["greedy_q_worst"] == 1.0                            # jeder im Nachhinein bedienbare Auftrag wird bedient
    near(s["deadhead_greedy"], 36.9, 0.05)                                                     # aber mit 36,9 statt 23,1 Minuten Fahrzeit je Auftrag
    near(s["deadhead_opt"], 23.1, 0.05)
    near(s["deadhead_ranking"], 53.3, 0.05)
    s20 = ev.dynamic_stats(10, 30, 40, 0, 20)
    near(s20["deadhead_greedy"], 19.7, 0.05)
    near(s20["deadhead_opt"], 20.1, 0.05)                                                      # bei Reichweite 40 ist Greedy je Auftrag sogar knapp billiger: er bedient weniger, aber die leichteren
