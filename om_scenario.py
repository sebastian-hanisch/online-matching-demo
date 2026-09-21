"""Szenario: Fahrzeuge und Aufträge auf einer Karte - beim Online-Matching kommen die Aufträge nacheinander.

Alles ist ganzzahlig und läuft über einen eigenen Zufallsgenerator (SplitMix64 auf Python-Ints) statt über
`numpy.random`: numpy garantiert keine über Versionen stabilen Zufallsströme, die CI installiert aber wöchentlich die
neueste Version. So sind Voreinstellungen, Seeds und jede im Text genannte Zahl auf Windows und Linux dieselben.
Kosten = auf ganze Minuten aufgerundete Entfernung (per `isqrt`, ohne Gleitkomma); ein Paar ist möglich, wenn die
Entfernung höchstens die Reichweite beträgt.

Dazu kommen hier drei Dinge, die es nur beim Online-Matching gibt: die Ankunftsreihenfolge der Aufträge (`arrival_order`,
eigener Zufallsstrom, getrennt vom Kartenstrom), die Ankunftsminuten (`times`) und die Treppe (`staircase`), auf der Greedy
nachweislich nur die Hälfte der möglichen Paare findet.
"""

from dataclasses import dataclass
from math import isqrt

import numpy as np

_MASK = (1 << 64) - 1
MAP_SIZE = 100
N_CENTRES = 3
STAIRCASE_MAX_N = 12                    # größte geometrische Treppe auf der 100x100-Karte (Abstand 6, Reichweite 6n)


class SplitMix64:
    """Kleiner, gut gemischter 64-Bit-Zufallsgenerator (Vigna); reine Ganzzahl-Arithmetik."""

    def __init__(self, seed):
        self.state = seed & _MASK

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
        return z ^ (z >> 31)

    def below(self, n):
        """Ganzzahl in 0..n-1 (die Modulo-Verzerrung bei n <= 101 liegt um 1e-17)."""
        return self.next() % n


def travel_cost(dx, dy):
    """Aufgerundete Entfernung in Minuten und das Quadrat der Entfernung (beides ganzzahlig)."""
    d2 = dx * dx + dy * dy
    r = isqrt(d2)
    return r + (1 if d2 > r * r else 0), d2


@dataclass(frozen=True)
class Scenario:
    vehicles: tuple      # ((x, y), ...)
    orders: tuple
    reach: int
    cost: np.ndarray     # (n, m) int64, Anfahrtszeit in Minuten
    feasible: np.ndarray  # (n, m) bool, Entfernung <= Reichweite

    @property
    def n(self):
        return len(self.vehicles)

    @property
    def m(self):
        return len(self.orders)

    def edges(self):
        """Alle möglichen Paare als (Kosten, Fahrzeug, Auftrag), aufsteigend sortiert (Gleichstand: kleinster Index)."""
        return sorted((int(self.cost[i, j]), i, j) for i in range(self.n) for j in range(self.m) if self.feasible[i, j])


def from_points(vehicles, orders, reach):
    n, m = len(vehicles), len(orders)
    cost = np.zeros((n, m), dtype=np.int64)
    feasible = np.zeros((n, m), dtype=bool)
    for i, (vx, vy) in enumerate(vehicles):
        for j, (ox, oy) in enumerate(orders):
            c, d2 = travel_cost(vx - ox, vy - oy)
            cost[i, j] = c
            feasible[i, j] = d2 <= reach * reach
    return Scenario(tuple(map(tuple, vehicles)), tuple(map(tuple, orders)), int(reach), cost, feasible)


def generate(n, m, reach, ballung, seed):
    """Zufällige Karte. `ballung` in ganzen Prozent: 0 = gleichmäßig verteilt, 100 = alle Punkte um drei Zentren."""
    rng = SplitMix64(seed)
    lo, hi = 15, MAP_SIZE - 15
    centres = [(lo + rng.below(hi - lo + 1), lo + rng.below(hi - lo + 1)) for _ in range(N_CENTRES)]

    def point():
        ux, uy = rng.below(MAP_SIZE + 1), rng.below(MAP_SIZE + 1)
        cx, cy = centres[rng.below(N_CENTRES)]
        jx, jy = rng.below(21) - 10, rng.below(21) - 10
        x = ((100 - ballung) * ux + ballung * (cx + jx)) // 100
        y = ((100 - ballung) * uy + ballung * (cy + jy)) // 100
        return min(max(x, 0), MAP_SIZE), min(max(y, 0), MAP_SIZE)

    vehicles = [point() for _ in range(n)]
    orders = [point() for _ in range(m)]
    return from_points(vehicles, orders, reach)


# --- Zufallsströme des Online-Modells -----------------------------------------------------------------------------------------

def perm(n, seed):
    """Zufällige Permutation von 0..n-1 (Fisher-Yates mit SplitMix64)."""
    rng = SplitMix64(seed)
    p = list(range(n))
    for i in range(n - 1, 0, -1):
        k = rng.below(i + 1)
        p[i], p[k] = p[k], p[i]
    return p


def arrival_seed(seed):
    return seed * 7919 + 13


def priority_seed(seed, k):
    return seed * 104729 + 31 * k + 7


def priority(n, seed, k=0):
    """Zufällige Rangfolge der Fahrzeuge für Ranking: `rank[i]` = Rang von Fahrzeug i (klein = zuerst)."""
    rank = [0] * n
    for pos, v in enumerate(perm(n, priority_seed(seed, k))):
        rank[v] = pos
    return rank


def degree(sc, j):
    """Wie viele Fahrzeuge Auftrag j erreichen können - seine Flexibilität."""
    return int(sc.feasible[:, j].sum())


ARRIVALS = ("random", "flex", "rigid", "sweep")


def arrival_order(sc, kind, seed=0):
    """Reihenfolge, in der die Aufträge eintreffen. random = Permutation aus dem Ankunftsstrom; flex = flexible zuerst (viele erreichbare Fahrzeuge),
    rigid = starre zuerst; sweep = von links nach rechts über die Karte; index = die Nummerierung selbst. Gleichstand: kleinster Index."""
    m = sc.m
    if kind == "random":
        return perm(m, arrival_seed(seed))
    if kind == "flex":
        return sorted(range(m), key=lambda j: (-degree(sc, j), j))
    if kind == "rigid":
        return sorted(range(m), key=lambda j: (degree(sc, j), j))
    if kind == "sweep":
        return sorted(range(m), key=lambda j: (sc.orders[j][0], sc.orders[j][1], j))
    if kind == "index":
        return list(range(m))
    raise ValueError(kind)


def times(m, horizon, seed):
    """Ankunftsminuten der m Aufträge im dynamischen Modell, aufsteigend sortiert (Auftrag j ist der j-te, der ankommt)."""
    rng = SplitMix64(seed * 6151 + 3)
    return sorted(rng.below(horizon + 1) for _ in range(m))


# --- feste Karten -------------------------------------------------------------------------------------------------------------

def _fan_orders(vehicles, reach, x_max):
    """Für t = 0..n-1 ein ganzzahliger Punkt links von den Fahrzeugen, der genau die Fahrzeuge 0..n-t-1 erreicht und dessen Kosten zu ihnen streng steigen
    (kein Gleichstand, den ein Tie-Break entscheiden könnte). Unter allen solchen Punkten wird der zu einem Zielpunkt (x = 15, y = 4 t) nächste genommen: so fächern sich die
    Aufträge in einer Spalte auf, statt sich auf der Geraden der Fahrzeuge zu überdecken."""
    n = len(vehicles)
    out = []
    for t in range(n):
        k = n - t
        best = None
        for y in range(MAP_SIZE + 1):
            for x in range(x_max + 1):
                ds = [travel_cost(vx - x, vy - y) for vx, vy in vehicles]
                if any((d2 <= reach * reach) != (i < k) for i, (_c, d2) in enumerate(ds)):
                    continue
                if all(ds[i][0] < ds[i + 1][0] for i in range(k - 1)):
                    key = ((x - 15) ** 2 + (y - 4 * t) ** 2, y, x)
                    if best is None or key < best[0]:
                        best = (key, (x, y))
        if best is None:
            raise ValueError(f"keine Treppe für n = {n}")
        out.append(best[1])
    return out


def staircase(n=8):
    """Die Treppe: Fahrzeuge auf einer Geraden bei x = 30 + 6 i; Auftrag t (t = 0..n-1) liegt links davon und erreicht genau die Fahrzeuge 0..n-t-1 - Auftrag 0 alle,
    der letzte nur Fahrzeug 0. Das nächste freie Fahrzeug ist für jeden Auftrag das mit dem kleinsten Index, also das flexibelste: Greedy verbraucht immer das, was die
    späteren Aufträge am nötigsten brauchen, und findet ceil(n/2) der n möglichen Paare - ohne dass ein Tie-Break im Spiel wäre (die Kosten steigen streng).
    Die Aufträge sind in Ankunftsreihenfolge nummeriert: kommen sie in der Nummerierung, ist das die schlechteste Reihenfolge (flexible zuerst)."""
    if not 2 <= n <= STAIRCASE_MAX_N:
        raise ValueError(n)
    vehicles = [(30 + 6 * i, 0) for i in range(n)]
    reach = 6 * n
    return from_points(vehicles, _fan_orders(vehicles, reach, 30), reach)


def staircase_graph(n):
    """Dieselbe Erreichbarkeitsstruktur ohne Geometrie, für beliebig große n (Auftrag t erreicht die Fahrzeuge 0..n-t-1; Kosten steigen mit dem Index).
    Die Punkte sind nur Platzhalter auf einer Geraden."""
    cost = np.zeros((n, n), dtype=np.int64)
    feasible = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for t in range(n):
            cost[i, t] = 1 + i + t
            feasible[i, t] = i < n - t
    return Scenario(tuple((i, 0) for i in range(n)), tuple((i, 1) for i in range(n)), n, cost, feasible)


def p4_chain(k=1):
    """k getrennte Pfade V1-O1-V2-O2 (Versatz 30 in y, außer Reichweite füreinander). Die mittlere Kante ist die
    billigste, Greedy nimmt sie zuerst und findet k Paare, das Optimum 2k (genau die Hälfte) - der Zwei-Wege-Fall im Kleinen."""
    vehicles, orders = [], []
    for c in range(k):
        y = 5 + 30 * c
        vehicles += [(0, y), (12, y)]
        orders += [(10, y), (22, y)]
    return from_points(vehicles, orders, reach=10)


NETS = {
    "p4": lambda: p4_chain(1),
    "stairs": lambda: staircase(8),
}


def build(net, n, m, reach, ballung, seed):
    """Karte zu den Einstellungen; feste Karten ignorieren die Zufallsparameter."""
    if net in NETS:
        return NETS[net]()
    return generate(n, m, reach, ballung, seed)
