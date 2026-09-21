"""Online-Matching: die Aufträge kommen nacheinander und jeder wird sofort und unwiderruflich einem Fahrzeug zugeordnet.

Die Fahrzeuge sind vorab bekannt, ein Fahrzeug fährt höchstens einen Auftrag. Bei der Ankunft eines Auftrags sieht die Regel nur dessen Zeile
(welche Fahrzeuge erreichen ihn, zu welchen Kosten) und die bisherigen Entscheidungen - nicht die späteren Aufträge. Drei Regeln:

- Greedy: das billigste mögliche freie Fahrzeug (Gleichstand: kleinster Index).
- Ranking (Karp, Vazirani, Vazirani 1990): eine zufällig gezogene, feste Rangfolge der Fahrzeuge; der Auftrag nimmt das bestplatzierte mögliche freie
  Fahrzeug. Die Kosten spielen keine Rolle (`prio` = `rank[i]`, klein = zuerst; mit der Identität `range(n)` ist es "Greedy nach Index").
- Batching (Fenster w): w Ankünfte werden gesammelt, dann löst die Ungarische Methode die Zuordnung dieses Stapels zu den noch freien Fahrzeugen
  optimal; wer dabei leer ausgeht, wird endgültig abgelehnt. Der letzte Stapel wird zum Schluss abgerechnet.

Alle drei bedienen, sobald es ein mögliches freies Fahrzeug gibt (maximale Regeln). Aufwand wird in angesehenen Kanten gezählt, nie in Sekunden:
Greedy und Ranking sehen je Ankunft die möglichen Kanten zu den noch freien Fahrzeugen, Batching die der Ungarischen Methode (`scanned_total`).
Der Ablauf ist ein Ereignisprotokoll (`Event`), aus dem `state_at` jeden Zwischenzustand herstellt.
"""

from dataclasses import dataclass

import numpy as np

from om_hungarian import hungarian
from om_scenario import Scenario

RULE_GREEDY, RULE_RANKING, RULE_BATCH = "greedy", "ranking", "batch"
RULES = (RULE_GREEDY, RULE_RANKING, RULE_BATCH)


@dataclass(frozen=True)
class Event:
    k: int              # Zahl der bisher eingetroffenen Aufträge (1-basiert)
    order: int          # der eben eingetroffene Auftrag
    kind: str           # "assign" | "reject" | "wait" (Stapel wird noch gesammelt) | "flush" (dieser Auftrag schließt den Stapel)
    assigned: tuple     # neue Paare ((i, j), ...) in diesem Schritt
    rejected: tuple     # endgültig abgelehnte Aufträge in diesem Schritt
    waiting: tuple      # Aufträge im offenen Stapel nach diesem Schritt
    options: tuple      # ((i, Kosten), ...): die freien Fahrzeuge in Reichweite des eintreffenden Auftrags (bei "flush": die des Stapels als Vereinigung leer)
    chosen: int         # gewähltes Fahrzeug (Greedy/Ranking), sonst -1
    scanned: int        # in diesem Schritt angesehene Kanten


@dataclass(frozen=True)
class Online:
    rule: str
    w: int
    order: tuple        # Ankunftsreihenfolge
    prio: tuple         # Rangfolge der Fahrzeuge (nur Ranking, sonst ())
    pairs: tuple        # ((i, j), ...) in der Reihenfolge der Entscheidungen
    cost: int
    events: tuple
    rejected: tuple
    tie: str = "low"    # Gleichstand bei Greedy: "low" = kleinster Index (Standard), "high" = größter (nur für die Gegenprobe)

    @property
    def count(self):
        return len(self.pairs)

    @property
    def scanned_total(self):
        return sum(e.scanned for e in self.events)

    def key(self):
        return self.count, -self.cost


def _sub(sc, vs, os_):
    """Teilkarte mit den Fahrzeugen `vs` und den Aufträgen `os_` (Indexlisten)."""
    return Scenario(tuple(sc.vehicles[i] for i in vs), tuple(sc.orders[j] for j in os_), sc.reach,
                    sc.cost[np.ix_(vs, os_)], sc.feasible[np.ix_(vs, os_)])


def _options(sc, free, j):
    return tuple((i, int(sc.cost[i, j])) for i in free if sc.feasible[i, j])


def run_online(sc, order, rule=RULE_GREEDY, w=1, prio=None, tie="low"):
    """Lässt die Aufträge in `order` eintreffen und wendet die Regel an. `w` gilt nur für Batching, `prio` nur für Ranking, `tie` nur für Greedy."""
    order = tuple(order)
    assert sorted(order) == list(range(sc.m)), "order ist keine Permutation der Aufträge"
    if rule == RULE_RANKING:
        assert prio is not None and sorted(prio) == list(range(sc.n)), "Ranking braucht eine Rangfolge der Fahrzeuge"
    used = set()
    pairs, events, rejected_all, window = [], [], [], []
    for k, j in enumerate(order, start=1):
        free = [i for i in range(sc.n) if i not in used]
        if rule in (RULE_GREEDY, RULE_RANKING):
            opts = _options(sc, free, j)
            if not opts:
                rejected_all.append(j)
                events.append(Event(k, j, "reject", (), (j,), (), (), -1, 0))
                continue
            if rule == RULE_GREEDY:
                i = min(opts, key=(lambda o: (o[1], o[0])) if tie == "low" else (lambda o: (o[1], -o[0])))[0]
            else:
                i = min(opts, key=lambda o: (prio[o[0]], o[0]))[0]
            used.add(i)
            pairs.append((i, j))
            events.append(Event(k, j, "assign", ((i, j),), (), (), opts, i, len(opts)))
            continue
        window.append(j)
        if len(window) < w and k < len(order):
            events.append(Event(k, j, "wait", (), (), tuple(window), _options(sc, free, j), -1, 0))
            continue
        batch, window = window, []
        new, rej, scanned = [], [], 0
        if free:
            ss = _sub(sc, free, batch)
            if ss.feasible.any():
                res = hungarian(ss, record=False)
                scanned = res.scanned_total
                new = [(free[a], batch[b]) for a, b in res.pairs]
        got = {j2 for _, j2 in new}
        rej = [j2 for j2 in batch if j2 not in got]
        for i, j2 in new:
            used.add(i)
        pairs += new
        rejected_all += rej
        events.append(Event(k, j, "flush", tuple(new), tuple(rej), (), (), -1, scanned))
    cost = int(sum(sc.cost[i, j] for i, j in pairs))
    return Online(rule, w if rule == RULE_BATCH else 1, order, tuple(prio) if rule == RULE_RANKING else (), tuple(pairs), cost, tuple(events), tuple(rejected_all), tie)


def state_at(res, k):
    """Zustand nach k Ankünften: (Paare, wartende Aufträge, abgelehnte Aufträge)."""
    pairs, rejected, waiting = [], [], ()
    for e in res.events[:k]:
        pairs += e.assigned
        rejected += e.rejected
        waiting = e.waiting
    return tuple(pairs), tuple(waiting), tuple(rejected)


def prefix_optimum(sc, order, k):
    """Größte Paarzahl und (kleinste Kosten dazu) der ersten k Ankünfte, im Nachhinein und mit Umhängen berechnet - die Messlatte im Verlaufsdiagramm."""
    if k == 0:
        return 0, 0
    os_ = list(order[:k])
    ss = _sub(sc, list(range(sc.n)), os_)
    if not ss.feasible.any():
        return 0, 0
    res = hungarian(ss, record=False)
    return res.count, res.cost


def validate(sc, res):
    """Unabhängige Prüfung: jedes Paar möglich, jedes Fahrzeug und jeder Auftrag höchstens einmal, jeder Auftrag genau einmal entschieden (Paar oder Ablehnung),
    Entscheidungen nie nachträglich geändert (das Ereignisprotokoll baut die Paare nur auf), und die Regel wurde eingehalten (Greedy: billigstes freies mögliches Fahrzeug;
    Ranking: bestplatziertes; Batching: jeder Stapel optimal für die zu diesem Zeitpunkt freien Fahrzeuge)."""
    pairs = list(res.pairs)
    ok = {"feasible": all(sc.feasible[i, j] for i, j in pairs),
          "unique_v": len({i for i, _ in pairs}) == len(pairs), "unique_o": len({j for _, j in pairs}) == len(pairs),
          "all_decided": sorted([j for _, j in pairs] + list(res.rejected)) == list(range(sc.m)),
          "cost": res.cost == int(sum(sc.cost[i, j] for i, j in pairs))}
    used, rule_ok, pending = set(), True, []
    for e in res.events:
        free = [i for i in range(sc.n) if i not in used]
        if e.kind in ("assign", "reject"):
            opts = _options(sc, free, e.order)
            if e.kind == "reject":
                rule_ok &= not opts
            else:
                i = e.assigned[0][0]
                best = min(opts, key=lambda o: ((o[1], o[0] if res.tie == "low" else -o[0]) if res.rule == RULE_GREEDY else (res.prio[o[0]], o[0])))[0] if opts else -1
                rule_ok &= i == best
        elif e.kind == "wait":
            pending.append(e.order)
        elif e.kind == "flush":
            batch, pending = pending + [e.order], []
            rule_ok &= sorted([j for _, j in e.assigned] + list(e.rejected)) == sorted(batch)
            if free and batch:
                ss = _sub(sc, free, batch)
                if ss.feasible.any():
                    opt = hungarian(ss, record=False)
                    rule_ok &= (len(e.assigned), sum(int(sc.cost[i, j]) for i, j in e.assigned)) == (opt.count, opt.cost)
        used |= {i for i, _ in e.assigned}
    ok["rule"] = rule_ok
    ok["all_ok"] = all(ok.values())
    return ok
