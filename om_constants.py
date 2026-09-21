"""Konstanten, Regler-Grenzen, Presets und feste Seed-Mengen der Demo "Online-Matching"."""

# --- Regler (wie in den Vorgängerdemos) ---------------------------------------------------------------------------------
N_MIN, N_MAX, DEFAULT_N = 3, 40, 20          # Fahrzeuge
M_MIN, M_MAX, DEFAULT_M = 3, 40, 20          # Aufträge
REACH_MIN, REACH_MAX, DEFAULT_REACH = 10, 150, 40   # Reichweite in Minuten; ab 142 ist auf der 100x100-Karte alles erreichbar
BALLUNG_MIN, BALLUNG_MAX, DEFAULT_BALLUNG = 0, 100, 0   # ganze Prozent, Schritt 25
DEFAULT_SEED = 11                               # eine typische Karte: Greedy 18 von 20 Paaren, Prämie 21 %, Batching mit Fenster 5 findet 19
SEED_MAX = 2_000_000_000

NETS = {"random": "Zufällige Karte", "stairs": "Die Treppe (8×8)", "p4": "Pfad aus vier Punkten (2×2)"}
DEFAULT_NET = "random"
FIXED_NETS = ("stairs", "p4")

ARR_LABELS = {"random": "Zufällig", "flex": "Flexible zuerst (gegnerisch)", "rigid": "Starre zuerst", "sweep": "Von links nach rechts"}
DEFAULT_ARR = "random"

RULE_LABELS = {"greedy": "Greedy: das nächste freie Fahrzeug", "ranking": "Ranking: zufällige feste Rangfolge", "batch": "Batching: Stapel optimal lösen"}
DEFAULT_RULE = "greedy"
W_MIN, W_MAX, DEFAULT_W = 2, 20, 5           # Fenstergröße beim Batching
PRIO_MIN, PRIO_MAX, DEFAULT_PRIO = 1, 20, 1   # Nummer der gezogenen Rangfolge beim Ranking (1..K)
RANKING_K = 20                              # Ziehungen für den Erwartungswert von Ranking

MODEL_LABELS = {"static": "Ein Fahrzeug fährt einmal", "dynamic": "Fahrzeuge werden wieder frei"}
DEFAULT_MODEL = "static"
DUR_MIN, DUR_MAX, DEFAULT_DUR = 5, 120, 20   # Belegungszeit D in Minuten (dynamisches Modell)
HORIZON = 120                               # Ankunftsminuten 0..T im dynamischen Modell (D > T: nie wieder frei, das statische Modell)

# --- feste Seed-Mengen (dieselben wie in den Vorgängerdemos; unabhängig vom Nutzer-Seed) --------------------------------------------
DIST_SEEDS = tuple(range(100000, 100100))
SWEEP_SEEDS = DIST_SEEDS[:40]
WINDOWS = (1, 2, 5, 10)                      # dazu "alle" (Fenster >= m ist genau die Offline-Lösung)
STAIR_NS = (3, 4, 5, 6, 8, 10, 20, 40, 80, 160)
STAIR_EXACT_MAX = 8                         # bis hierhin exakt über alle Rangfolgen, darüber Monte Carlo
STAIR_MC_DRAWS = 2000
DUR_SWEEP = (1, 5, 10, 15, 20, 25, 30, 40, 60, 90, 121)
HILL_SWEEPS = 300                           # Vertauschungen bei der Suche nach der schlechtesten Reihenfolge

COLORS = {"matched": "#1f77b4", "assign": "#2ca02c", "reject": "#d62728", "vehicle": "#111111", "order": "#ff7f0e", "future": "#c8c8c8", "wait": "#9467bd",
          "opt": "#d62728", "greedy": "#8c564b", "trail": "#7f7f7f"}

# --- Presets ------------------------------------------------------------------------------------------------------------------
_BASE = dict(net="random", arr=DEFAULT_ARR, rule=DEFAULT_RULE, w=DEFAULT_W, prio=DEFAULT_PRIO, model=DEFAULT_MODEL, dur=DEFAULT_DUR, n=DEFAULT_N, m=DEFAULT_M, reach=DEFAULT_REACH,
             ballung=DEFAULT_BALLUNG, seed=DEFAULT_SEED)
PRESETS = {
    "🪜 Die Treppe": {**_BASE, "net": "stairs", "arr": "flex"},
    "🗺️ Zufällige Karte": {**_BASE},
    "😈 Flexible zuerst": {**_BASE, "arr": "flex"},
    "🪟 Fenster von 5": {**_BASE, "rule": "batch"},
    "🌐 Alles erreichbar": {**_BASE, "reach": 150, "seed": 16},
    "📡 Knappe Reichweite": {**_BASE, "reach": 10, "seed": 92},
    "📐 Mehr Aufträge als Fahrzeuge": {**_BASE, "n": 10, "m": 20, "seed": 3},
    "⏱️ Fahrzeuge werden wieder frei": {**_BASE, "model": "dynamic", "n": 10, "m": 30, "seed": 18},
}
# Jede Zahl in diesen Texten ist in tests/test_claims.py über die 100 festen Karten (DIST_SEEDS) belegt
PRESET_HELP = {
    "🪜 Die Treppe": "Acht Fahrzeuge in einer Reihe, Auftrag t erreicht genau die Fahrzeuge 0 bis 7 − t. Kommen die flexiblen Aufträge zuerst (wie hier), nimmt Greedy für jeden das nächste - das flexibelste - Fahrzeug und findet 4 von 8 Paaren, genau die Hälfte, ohne dass ein Tie-Break im Spiel wäre (die Kosten steigen streng). Ranking erreicht hier im Erwartungswert 5,3 von 8 (0,665); für große n nähert sich das von oben 1 − 1/e ≈ 0,632.",
    "🗺️ Zufällige Karte": "20 Fahrzeuge, 20 Aufträge, Reichweite 40, zufällige Reihenfolge: Greedy findet im Mittel 89 % der möglichen Paare (Median 0,90, schlechteste Karte 0,74) und zahlt 19 % mehr als die billigste Paarung derselben Paarzahl (Median 19,2 %). Diese Karte: 18 von 20 Paaren, und Batching mit Fenster 5 findet 19.",
    "😈 Flexible zuerst": "Dieselbe Karte, aber die flexiblen Aufträge (viele erreichbare Fahrzeuge) kommen zuerst und nehmen den starren die Fahrzeuge weg: Greedy fällt im Mittel auf 0,83 (Median 0,85, schlechteste Karte 0,65) - nicht annähernd auf 1/2. Das gelingt nur der Treppe.",
    "🪟 Fenster von 5": "Batching sammelt 5 Aufträge und ordnet sie optimal zu: im Mittel 0,906 der Paare (Greedy 0,890) bei 14,6 % Prämie statt 18,8 %. Je Karte nicht immer besser: auf 8 von 100 Karten findet Fenster 5 weniger Paare als Fenster 2.",
    "🌐 Alles erreichbar": "Bei Reichweite 150 kann jeder Auftrag jedes Fahrzeug nehmen: Greedy findet auf allen 100 Karten alle 20 Paare - der Verlust liegt allein bei den Kosten (Prämie im Median 21,2 %). Ranking, das die Kosten ignoriert, zahlt im Median 160 %.",
    "📡 Knappe Reichweite": "Wenige mögliche Paare: bei Reichweite 10 ist Greedy auf 85 von 100 Karten optimal (Mittel 0,980, Median 1,000). Wo es kaum Alternativen gibt, kann eine falsche Wahl wenig anrichten.",
    "📐 Mehr Aufträge als Fahrzeuge": "10 Fahrzeuge, 20 Aufträge: die Fahrzeuge sind knapp, aber Greedy findet auf 87 von 100 Karten alle möglichen Paare (Mittel 0,985). Die Kosten sind der Preis: Prämie im Median 49,3 %.",
    "⏱️ Fahrzeuge werden wieder frei": "10 Fahrzeuge, 30 Aufträge innerhalb von 120 Minuten, jedes Fahrzeug nach der Zuordnung 20 Minuten belegt: Greedy bedient im Mittel 91 % der im Nachhinein bedienbaren Aufträge (Median 0,92, schlechteste Karte 0,74). Bei D = 1 sind es 99 %, bei D = 121 (nie wieder frei) 100 %: dazwischen liegt der Buckel.",
}
