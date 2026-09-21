"""Online-Matching mit Zeit: ein Fahrzeug wird nach seiner Fahrt wieder frei.

Auftrag j trifft in der ganzzahligen Minute a_j ein (aufsteigend sortiert, Horizont T; Auftrag j ist der j-te Eintreffende). Wird ein Fahrzeug einem Auftrag zugeordnet,
fährt es zu ihm und ist ab der Zuordnung genau `dur` (D) Minuten belegt; danach ist es frei und steht am Ort dieses Auftrags. Die Fahrzeit ist nur Kosten, sie verlängert
die Belegung nicht (feste Belegungszeit). Ein Fahrzeug ist für einen Auftrag möglich, wenn es frei ist und dessen Ort innerhalb der Reichweite von seinem AKTUELLEN Ort liegt.
Ist D größer als der Horizont, wird nie ein Fahrzeug wieder frei: das ist genau das statische Modell.

Regeln: Greedy (billigstes mögliches freies Fahrzeug, Gleichstand kleinster Index) und Ranking (feste Zufallsrangfolge). Batching gehört nicht hierher: eine verzögerte Zuordnung
kann Fahrzeuge zu Zeitpunkten belegen, die der Sofort-Maßstab nicht kennt, und der Vergleich mit dem Optimum wäre nicht mehr fair (sichtbar an Quotienten über 1).

Der Maßstab ist das im Nachhinein beste Ergebnis unter denselben Regeln der Belegung (Zuordnung bei Ankunft, feste Belegung D): erst möglichst viele bediente Aufträge, dann die
geringste Summe der Fahrzeiten. Das ist ein Min-Cost-Flow: Quelle -> Fahrzeug -> (Auftrag_ein -> Auftrag_aus -> nächster Auftrag_ein ...) -> Senke; die Kante Auftrag_ein -> Auftrag_aus kostet
-L mit L = 1 + m * 142 (größer als jede Summe von Fahrzeiten), ein Auftrag k folgt auf j, wenn a_k >= a_j + D und k von j aus erreichbar ist. Genau n Flusseinheiten, eine je Fahrzeug
(ein unbenutztes Fahrzeug fließt direkt zur Senke). Kürzeste Wege per Bellman-Ford (die Kosten sind negativ). Alles ganzzahlig.
"""

from dataclasses import dataclass

from om_scenario import travel_cost

INF = 1 << 60
MAX_LEG = 142                       # größte mögliche Fahrzeit auf der 100x100-Karte (aufgerundete Diagonale)


@dataclass(frozen=True)
class DEvent:
    k: int              # Zahl der bisher eingetroffenen Aufträge (1-basiert)
    order: int
    t: int              # Ankunftsminute
    kind: str           # "assign" | "reject"
    vehicle: int        # gewähltes Fahrzeug oder -1
    cost: int           # Fahrzeit des gewählten Fahrzeugs
    origin: tuple       # dessen Ort vor der Fahrt
    options: tuple      # ((i, Kosten), ...): freie Fahrzeuge in Reichweite
    n_busy: int         # wie viele Fahrzeuge zu diesem Zeitpunkt belegt waren


@dataclass(frozen=True)
class DynOnline:
    rule: str
    dur: int
    times: tuple
    prio: tuple
    pairs: tuple        # ((i, j), ...) in Ankunftsreihenfolge
    cost: int
    events: tuple

    @property
    def count(self):
        return len(self.pairs)

    @property
    def served(self):
        return len(self.pairs)


def _leg(p, q):
    return travel_cost(p[0] - q[0], p[1] - q[1])


def run_dynamic(sc, times, dur, rule="greedy", prio=None):
    """Lässt die Aufträge 0..m-1 zu den Minuten `times` eintreffen (aufsteigend). Rückgabe `DynOnline`."""
    n, m = sc.n, sc.m
    assert len(times) == m and list(times) == sorted(times) and dur >= 1
    if rule == "ranking":
        assert prio is not None and sorted(prio) == list(range(n))
    loc = list(sc.vehicles)
    free_at = [0] * n
    pairs, events, total = [], [], 0
    for j in range(m):
        t, o = times[j], sc.orders[j]
        opts = []
        for i in range(n):
            if free_at[i] > t:
                continue
            c, d2 = _leg(loc[i], o)
            if d2 <= sc.reach * sc.reach:
                opts.append((i, c))
        n_busy = sum(1 for i in range(n) if free_at[i] > t)
        if not opts:
            events.append(DEvent(j + 1, j, t, "reject", -1, 0, (), (), n_busy))
            continue
        i, c = min(opts, key=(lambda x: (x[1], x[0])) if rule == "greedy" else (lambda x: (prio[x[0]], x[0])))
        events.append(DEvent(j + 1, j, t, "assign", i, c, loc[i], tuple(opts), n_busy))
        loc[i] = o
        free_at[i] = t + dur
        pairs.append((i, j))
        total += c
    return DynOnline(rule, dur, tuple(times), tuple(prio) if rule == "ranking" else (), tuple(pairs), total, tuple(events))


def state_at(sc, res, k):
    """Zustand nach k Ankünften: (Orte der Fahrzeuge, Minute, bis wann jedes Fahrzeug belegt ist, Paare)."""
    loc = list(sc.vehicles)
    free_at = [0] * sc.n
    pairs = []
    for e in res.events[:k]:
        if e.kind == "assign":
            loc[e.vehicle] = sc.orders[e.order]
            free_at[e.vehicle] = e.t + res.dur
            pairs.append((e.vehicle, e.order))
    t = res.events[k - 1].t if k else 0
    return tuple(loc), t, tuple(free_at), tuple(pairs)


# --- Offline-Maßstab: Min-Cost-Flow ------------------------------------------------------------------------------------------------

class _Flow:
    def __init__(self, n_nodes):
        self.g = [[] for _ in range(n_nodes)]

    def add(self, u, v, cap, cost):
        self.g[u].append([v, cap, cost, len(self.g[v]), cap])
        self.g[v].append([u, 0, -cost, len(self.g[u]) - 1, 0])

    def push(self, s, t):
        """Ein kürzester Weg s -> t (Bellman-Ford mit Warteschlange), Fluss um 1 erhöhen. Rückgabe Wegkosten oder None."""
        n = len(self.g)
        dist, prev, inq = [INF] * n, [None] * n, [False] * n
        dist[s] = 0
        queue, head = [s], 0
        while head < len(queue):
            u = queue[head]
            head += 1
            inq[u] = False
            for idx, (v, cap, cost, _rev, _c0) in enumerate(self.g[u]):
                if cap > 0 and dist[u] + cost < dist[v]:
                    dist[v] = dist[u] + cost
                    prev[v] = (u, idx)
                    if not inq[v]:
                        inq[v] = True
                        queue.append(v)
        if dist[t] >= INF:
            return None
        v = t
        while v != s:
            u, idx = prev[v]
            self.g[u][idx][1] -= 1
            self.g[v][self.g[u][idx][3]][1] += 1
            v = u
        return dist[t]


def offline_dynamic(sc, times, dur, k=None):
    """Bestes Ergebnis im Nachhinein für die ersten k Aufträge (Standard: alle). Rückgabe dict: served, cost (Fahrzeiten), routes {Fahrzeug: [Aufträge in Reihenfolge]}."""
    n = sc.n
    m = sc.m if k is None else k
    if m == 0:
        return {"served": 0, "cost": 0, "routes": {}}
    L = 1 + m * MAX_LEG
    s, t = 0, 1
    V = lambda i: 2 + i
    IN = lambda j: 2 + n + 2 * j
    OUT = lambda j: 3 + n + 2 * j
    f = _Flow(2 + n + 2 * m)
    reach2 = sc.reach * sc.reach
    for i in range(n):
        f.add(s, V(i), 1, 0)
        f.add(V(i), t, 1, 0)
        for j in range(m):
            c, d2 = _leg(sc.vehicles[i], sc.orders[j])
            if d2 <= reach2:
                f.add(V(i), IN(j), 1, c)
    for j in range(m):
        f.add(IN(j), OUT(j), 1, -L)
        f.add(OUT(j), t, 1, 0)
        for k2 in range(m):
            if k2 != j and times[k2] >= times[j] + dur:
                c, d2 = _leg(sc.orders[j], sc.orders[k2])
                if d2 <= reach2:
                    f.add(OUT(j), IN(k2), 1, c)
    total = 0
    for _ in range(n):
        d = f.push(s, t)
        if d is None:
            break
        total += d
    served = sum(1 for j in range(m) for e in f.g[IN(j)] if e[0] == OUT(j) and e[1] == 0 and e[4] == 1)
    routes = {}
    nxt = {}
    for j in range(m):
        for v, cap, _c, _rev, cap0 in f.g[OUT(j)]:
            if cap0 == 1 and cap == 0 and v >= 2 + n and (v - 2 - n) % 2 == 0:
                nxt[j] = (v - 2 - n) // 2
    for i in range(n):
        for v, cap, _c, _rev, cap0 in f.g[V(i)]:
            if cap0 == 1 and cap == 0 and v >= 2 + n:
                j = (v - 2 - n) // 2
                chain = [j]
                while chain[-1] in nxt:
                    chain.append(nxt[chain[-1]])
                routes[i] = chain
    return {"served": served, "cost": total + served * L, "routes": routes}


def validate(sc, res):
    """Unabhängige Prüfung durch neues Abspielen: jeder Auftrag höchstens einmal, ein Fahrzeug nur, wenn es frei war (Abstand >= D zwischen zwei seiner Aufträge),
    Fahrzeit und Reichweite vom aktuellen Ort, Regel eingehalten (Greedy: billigstes freies mögliches Fahrzeug; Ranking: bestplatziertes)."""
    loc = list(sc.vehicles)
    last = [None] * sc.n
    ok = {"once": len({j for _, j in res.pairs}) == len(res.pairs), "free": True, "reach": True, "cost": True, "rule": True}
    total = 0
    for e in res.events:
        cands = []
        for i in range(sc.n):
            if last[i] is not None and e.t < last[i] + res.dur:
                continue
            c, d2 = _leg(loc[i], sc.orders[e.order])
            if d2 <= sc.reach * sc.reach:
                cands.append((i, c))
        best = -1
        if cands:
            best = min(cands, key=(lambda x: (x[1], x[0])) if res.rule == "greedy" else (lambda x: (res.prio[x[0]], x[0])))[0]
        if e.kind == "reject":
            ok["rule"] &= best == -1
            continue
        i = e.vehicle
        ok["free"] &= last[i] is None or e.t >= last[i] + res.dur
        c, d2 = _leg(loc[i], sc.orders[e.order])
        ok["reach"] &= d2 <= sc.reach * sc.reach
        ok["cost"] &= c == e.cost
        ok["rule"] &= i == best
        total += c
        loc[i] = sc.orders[e.order]
        last[i] = e.t
    ok["total"] = total == res.cost
    ok["all_ok"] = all(ok.values())
    return ok
