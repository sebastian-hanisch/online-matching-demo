"""Plotly-Abbildungen des Online-Matchings: Karte nach k Ankünften (statisch und mit Zeit), Verlauf gegen die im Nachhinein beste Lösung, Verteilung über die Karten, Fenster-Sweep,
Treppe, Ankunftsmodelle, Belegungszeit-Sweep und Aufwand. Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen.
Karten haben keine Legende (sie würde in schmalen Spalten die Zeichenfläche auf fast null drücken); die Farben stehen in der Erklärung unter der Karte."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import om_constants as C
import om_dynamic as Dy
from om_online import state_at


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.08), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _collinear(points):
    pts = list(points)
    (x0, y0), (x1, y1) = pts[0], next((p for p in pts if p != pts[0]), pts[0])
    return all((x1 - x0) * (y - y0) - (y1 - y0) * (x - x0) == 0 for x, y in pts)


def _path(p, q, curved, side, steps=14):
    """Punkte einer Verbindung: gerade, oder (bei Punkten auf einer Geraden) als Bogen, dessen Seite wechselt."""
    (vx, vy), (ox, oy) = p, q
    if not curved:
        return [vx, ox], [vy, oy]
    dx, dy = ox - vx, oy - vy
    cx, cy = (vx + ox) / 2 - side * 0.35 * dy, (vy + oy) / 2 + side * 0.35 * dx
    ts = [k / steps for k in range(steps + 1)]
    return ([(1 - t) ** 2 * vx + 2 * (1 - t) * t * cx + t * t * ox for t in ts], [(1 - t) ** 2 * vy + 2 * (1 - t) * t * cy + t * t * oy for t in ts])


def _segments(links, curved):
    """Linienspur für eine Liste von (Punkt, Punkt, Seite); None trennt die Segmente."""
    x, y = [], []
    for p, q, side in links:
        px, py = _path(p, q, curved, side)
        x += px + [None]
        y += py + [None]
    return x, y


def _map_layout(fig, pts, curved, height):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    pad = 8
    if curved:
        span = max(max(xs) - min(xs), max(ys) - min(ys), 1)
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        if max(xs) - min(xs) >= max(ys) - min(ys):
            fig.update_xaxes(visible=False, range=[min(xs) - pad, max(xs) + pad])
            fig.update_yaxes(visible=False, range=[cy - span * 0.22, cy + span * 0.22])
        else:
            fig.update_xaxes(visible=False, range=[cx - span * 0.22, cx + span * 0.22])
            fig.update_yaxes(visible=False, range=[min(ys) - pad, max(ys) + pad])
        fig = _base(fig, min(height, 320))
    else:
        # constrain="domain": bei gleichem Maßstab schrumpft die Zeichenfläche statt dass sich der Wertebereich aufbläht (sonst kann eine Karte, die beim ersten Zeichnen in einer noch
        # schmalen Spalte steht, dauerhaft auf einen Punkt kollabieren)
        fig.update_xaxes(visible=False, range=[min(xs) - pad, max(xs) + pad], scaleanchor="y", scaleratio=1, constrain="domain")
        fig.update_yaxes(visible=False, range=[min(ys) - pad, max(ys) + pad], constrain="domain")
        fig = _base(fig, height)
    fig.update_layout(showlegend=False)
    return fig


def _markers(fig, pts, idx, symbol, color, prefix, tpos, name, small, open_=False, size=12, opacity=1.0):
    if idx:
        fig.add_trace(go.Scatter(x=[pts[k][0] for k in idx], y=[pts[k][1] for k in idx], mode="markers+text" if small else "markers", name=name, opacity=opacity,
                                 text=[f"{prefix}{k + 1}" for k in idx] if small else None, textposition=tpos, hovertext=[f"{name} {prefix}{k + 1}" for k in idx], hoverinfo="text",
                                 marker=dict(symbol=symbol + ("-open" if open_ else ""), size=size, color=color, line=dict(width=2, color=color))))


def build_online_map(sc, order, res, k, height=430):
    """Karte nach k Ankünften (statisches Modell). Aufträge: hellgrau = noch nicht da, orange gefüllt = zugeordnet, violett = wartet im Stapel, rotes Kreuz = abgelehnt; blaue Linien = feste
    Zuordnungen. Das letzte Ereignis: gestrichelte graue Linien zu den freien Fahrzeugen in Reichweite, grüne Linie zum gewählten, roter Ring bei einer Ablehnung, grüner Ring um den Auftrag."""
    pairs, waiting, rejected = state_at(res, k)
    arrived = set(order[:k])
    assigned = {j for _, j in pairs}
    used = {i for i, _ in pairs}
    pts = list(sc.vehicles + sc.orders)
    curved = _collinear(pts)
    fig = go.Figure()
    faint = [(sc.vehicles[i], sc.orders[j], 1 if (i + j) % 2 == 0 else -1) for i in range(sc.n) for j in range(sc.m) if sc.feasible[i, j]]
    fx, fy = _segments(faint, curved)
    fig.add_trace(go.Scatter(x=fx, y=fy, mode="lines", line=dict(color="rgba(150,150,150,0.22)", width=1), hoverinfo="skip"))
    lx, ly = _segments([(sc.vehicles[i], sc.orders[j], 1 if (i + j) % 2 == 0 else -1) for i, j in pairs], curved)
    fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines", line=dict(color=C.COLORS["matched"], width=3.5), hoverinfo="skip"))
    ev = res.events[k - 1] if k else None
    if ev is not None and ev.kind in ("assign", "reject"):
        j = ev.order
        opts = [(sc.vehicles[i], sc.orders[j], 1 if (i + j) % 2 == 0 else -1) for i, _c in ev.options if i != ev.chosen]
        if opts:
            ox, oy = _segments(opts, curved)
            fig.add_trace(go.Scatter(x=ox, y=oy, mode="lines", line=dict(color="rgba(44,160,44,0.6)", width=2, dash="dot"), hoverinfo="skip"))
        if ev.chosen >= 0:
            cx, cy = _segments([(sc.vehicles[ev.chosen], sc.orders[j], 1 if (ev.chosen + j) % 2 == 0 else -1)], curved)
            fig.add_trace(go.Scatter(x=cx, y=cy, mode="lines", line=dict(color=C.COLORS["assign"], width=5), hoverinfo="skip"))
    small = sc.n + sc.m <= 24
    _markers(fig, sc.vehicles, [i for i in range(sc.n) if i in used], "square", "#2e7d32", "F", "top center", "Fahrzeug vergeben", small)
    _markers(fig, sc.vehicles, [i for i in range(sc.n) if i not in used], "square", "#2e7d32", "F", "top center", "Fahrzeug frei", small, open_=True)
    _markers(fig, sc.orders, [j for j in range(sc.m) if j not in arrived], "circle", C.COLORS["future"], "A", "bottom center", "Auftrag (kommt noch)", small, open_=True)
    _markers(fig, sc.orders, sorted(assigned), "circle", C.COLORS["order"], "A", "bottom center", "Auftrag zugeordnet", small)
    _markers(fig, sc.orders, sorted(waiting), "circle", C.COLORS["wait"], "A", "bottom center", "Auftrag wartet", small)
    rj = sorted(rejected)
    if rj:
        fig.add_trace(go.Scatter(x=[sc.orders[j][0] for j in rj], y=[sc.orders[j][1] for j in rj], mode="markers+text" if small else "markers", text=[f"A{j + 1}" for j in rj] if small else None,
                                 textposition="bottom center", hovertext=[f"Auftrag abgelehnt A{j + 1}" for j in rj], hoverinfo="text", marker=dict(symbol="x", size=12, color=C.COLORS["reject"], line=dict(width=2))))
    if ev is not None:
        color = C.COLORS["reject"] if ev.kind == "reject" else (C.COLORS["wait"] if ev.kind == "wait" else C.COLORS["assign"])
        p = sc.orders[ev.order]
        fig.add_trace(go.Scatter(x=[p[0]], y=[p[1]], mode="markers", hoverinfo="skip", marker=dict(symbol="circle-open", size=28, color=color, line=dict(width=3, color=color))))
    return _map_layout(fig, pts, curved, height)


def build_dynamic_map(sc, res, k, height=430):
    """Karte nach k Ankünften (mit Zeit): Fahrzeuge stehen dort, wo sie zuletzt einen Auftrag bedient haben (grau gefüllt = noch belegt, hohl = frei), graue Spur = bisherige Fahrten,
    Aufträge wie im statischen Modell; grüne Linie = die Fahrt zum eben eingetroffenen Auftrag."""
    loc, t, free_at, pairs = Dy.state_at(sc, res, k)
    served = {j for _, j in pairs}
    rejected = {e.order for e in res.events[:k] if e.kind == "reject"}
    arrived = set(range(k))
    pts = list(sc.vehicles + sc.orders)
    curved = _collinear(pts)
    fig = go.Figure()
    trail = [(sc.vehicles[e.vehicle] if e.origin == sc.vehicles[e.vehicle] else e.origin, sc.orders[e.order], 1) for e in res.events[:k] if e.kind == "assign"]
    tx, ty = _segments(trail, curved)
    fig.add_trace(go.Scatter(x=tx, y=ty, mode="lines", line=dict(color="rgba(120,120,120,0.55)", width=2, dash="dot"), hoverinfo="skip"))
    ev = res.events[k - 1] if k else None
    if ev is not None and ev.kind == "assign":
        cx, cy = _segments([(ev.origin, sc.orders[ev.order], 1)], curved)
        fig.add_trace(go.Scatter(x=cx, y=cy, mode="lines", line=dict(color=C.COLORS["assign"], width=5), hoverinfo="skip"))
    small = sc.n + sc.m <= 24
    busy = [i for i in range(sc.n) if free_at[i] > t]
    idle = [i for i in range(sc.n) if free_at[i] <= t]
    _markers(fig, sc.orders, [j for j in range(sc.m) if j not in arrived], "circle", C.COLORS["future"], "A", "bottom center", "Auftrag (kommt noch)", small, open_=True)
    _markers(fig, sc.orders, sorted(served), "circle", C.COLORS["order"], "A", "bottom center", "Auftrag bedient", small)
    vpts = list(loc)                                       # Fahrzeuge zuletzt: sie stehen nach einer Fahrt genau auf dem Auftrag und sollen nicht überdeckt werden
    _markers(fig, vpts, busy, "square", "#7f7f7f", "F", "top center", "Fahrzeug belegt", small, size=13)
    _markers(fig, vpts, idle, "square", "#2e7d32", "F", "top center", "Fahrzeug frei", small, open_=True, size=13)
    rj = sorted(rejected)
    if rj:
        fig.add_trace(go.Scatter(x=[sc.orders[j][0] for j in rj], y=[sc.orders[j][1] for j in rj], mode="markers+text" if small else "markers", text=[f"A{j + 1}" for j in rj] if small else None,
                                 textposition="bottom center", hovertext=[f"Auftrag abgelehnt A{j + 1}" for j in rj], hoverinfo="text", marker=dict(symbol="x", size=12, color=C.COLORS["reject"], line=dict(width=2))))
    if ev is not None:
        color = C.COLORS["reject"] if ev.kind == "reject" else C.COLORS["assign"]
        p = sc.orders[ev.order]
        fig.add_trace(go.Scatter(x=[p[0]], y=[p[1]], mode="markers", hoverinfo="skip", marker=dict(symbol="circle-open", size=28, color=color, line=dict(width=3, color=color))))
    return _map_layout(fig, pts, curved, height)


def build_progress(online, best, k, online_name="Online", best_name="im Nachhinein bestmöglich", y_title="Paare", height=240):
    """Über die Ankünfte: Paare der Regel (Treppe) gegen die im Nachhinein optimale Paarzahl derselben ersten k Aufträge; senkrecht der aktuelle Schritt."""
    fig = go.Figure()
    xs = list(range(len(online)))
    fig.add_trace(go.Scatter(x=xs, y=best, mode="lines", name=best_name, line=dict(color=C.COLORS["opt"], dash="dot", shape="hv")))
    fig.add_trace(go.Scatter(x=xs, y=online, mode="lines", name=online_name, line=dict(color=C.COLORS["assign"], shape="hv")))
    fig.add_vline(x=k, line=dict(color="#333", width=2))
    fig.update_xaxes(title="Ankunft Nr.")
    fig.update_yaxes(title=y_title, rangemode="tozero")
    fig = _base(fig, height)
    fig.update_layout(legend=dict(orientation="h", y=-0.45))
    return fig


def build_hist(values, mean=None, median=None, current=None, x_title="Gütequotient", bin_size=0.02, height=260, color=None):
    """Verteilung über die Karten; Mittel und Median als Linien, Ihre Ziehung als Marke."""
    fig = go.Figure(go.Histogram(x=values, xbins=dict(size=bin_size), marker_color=color or C.COLORS["matched"], opacity=0.75, name="Karten"))
    if median is not None:
        fig.add_vline(x=median, line=dict(color="#333", width=2), annotation_text=f"Median {median:.2f}".replace(".", ","), annotation_position="top left")
    if mean is not None:
        fig.add_vline(x=mean, line=dict(color="#555", width=2, dash="dot"), annotation_text=f"Mittel {mean:.2f}".replace(".", ","), annotation_position="top right")
    if current is not None:
        fig.add_vline(x=current, line=dict(color="#d62728", dash="dash"))
    fig.update_xaxes(title=x_title)
    fig.update_yaxes(title="Karten")
    return _base(fig, height)


def build_window_sweep(rows, height=340):
    """Gegen das Fenster w: Gütequotient der Paare (Mittel, Linie; schlechteste Karte, gepunktet) und Prämie der Kosten [%] (rechte Achse). Das letzte Fenster ist die Offline-Lösung."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    ws = [str(r["w"]) for r in rows]
    fig.add_trace(go.Scatter(x=ws, y=[r["q_mean"] for r in rows], mode="lines+markers", name="Quotient Paare (Mittel)", line=dict(color=C.COLORS["matched"])), secondary_y=False)
    fig.add_trace(go.Scatter(x=ws, y=[r["q_worst"] for r in rows], mode="lines+markers", name="Quotient (schlechteste Karte)", line=dict(color=C.COLORS["matched"], dash="dot")), secondary_y=False)
    fig.add_trace(go.Scatter(x=ws, y=[r["premium_mean"] for r in rows], mode="lines+markers", name="Prämie Kosten (Mittel) [%]", line=dict(color=C.COLORS["opt"])), secondary_y=True)
    fig.update_xaxes(title="Fenster w (Aufträge je Stapel; das letzte = alles auf einmal)", type="category")
    fig.update_yaxes(title="Gütequotient", secondary_y=False, range=[0.5, 1.02])
    fig.update_yaxes(title="Prämie [%]", secondary_y=True, rangemode="tozero", showgrid=False)
    fig = _base(fig, height)
    fig.update_layout(legend=dict(orientation="h", y=-0.35), height=height + 50)
    return fig


def build_stair_curve(rows, height=340):
    """Auf der Treppe gegen n: Verhältnis Paare / Optimum für Greedy (-> 1/2) und Ranking (von oben -> 1 - 1/e)."""
    fig = go.Figure()
    ns = [r["n"] for r in rows]
    fig.add_trace(go.Scatter(x=ns, y=[r["greedy_ratio"] for r in rows], mode="lines+markers", name="Greedy", line=dict(color=C.COLORS["greedy"])))
    fig.add_trace(go.Scatter(x=ns, y=[r["ranking_ratio"] for r in rows], mode="lines+markers", name="Ranking (Erwartung)", line=dict(color=C.COLORS["assign"])))
    fig.add_hline(y=1 - 1 / 2.718281828459045, line=dict(color="#555", dash="dot"), annotation_text="1 − 1/e = 0,632", annotation_position="top right")
    fig.add_hline(y=0.5, line=dict(color="#555", dash="dot"), annotation_text="1/2", annotation_position="bottom right")
    fig.update_xaxes(title="Größe n der Treppe", type="log")
    fig.update_yaxes(title="Paare / Optimum", range=[0.4, 1.02])
    fig = _base(fig, height)
    fig.update_layout(legend=dict(orientation="h", y=-0.3), height=height + 40)
    return fig


def build_arrival_bars(rows, labels, height=340):
    """Gütequotient der Paare (Mittel) je Ankunftsmodell für Greedy, Ranking und Batching; die Balkenspitze zeigt die schlechteste Karte von Greedy als Marke."""
    fig = go.Figure()
    names = [labels.get(r["arr"], "Schlechteste gefundene Reihenfolge (für Greedy)") for r in rows]
    for key, label, color in (("greedy", "Greedy", C.COLORS["greedy"]), ("ranking", "Ranking", C.COLORS["assign"]), ("batch", "Batching", C.COLORS["matched"])):
        fig.add_trace(go.Bar(x=names, y=[r[key]["q_mean"] for r in rows], name=label, marker_color=color))
    fig.add_trace(go.Scatter(x=names, y=[r["greedy"]["q_worst"] for r in rows], mode="markers", name="Greedy, schlechteste Karte", marker=dict(symbol="diamond", size=10, color="#000")))
    fig.update_yaxes(title="Paare / Optimum (Mittel)", range=[0.5, 1.02])
    fig.update_xaxes(tickangle=-20)
    fig.update_layout(barmode="group")
    fig = _base(fig, height)
    fig.update_layout(legend=dict(orientation="h", y=-0.5), height=height + 80)
    return fig


def build_dynamic_sweep(rows, height=340):
    """Gegen die Belegungszeit D: Anteil der vom Optimum bedienten Aufträge, den Greedy (Mittel, schlechteste Karte) und Ranking (Mittel) bedienen."""
    fig = go.Figure()
    ds = [r["dur"] for r in rows]
    fig.add_trace(go.Scatter(x=ds, y=[r["greedy_q_mean"] for r in rows], mode="lines+markers", name="Greedy (Mittel)", line=dict(color=C.COLORS["greedy"])))
    fig.add_trace(go.Scatter(x=ds, y=[r["greedy_q_worst"] for r in rows], mode="lines+markers", name="Greedy (schlechteste Karte)", line=dict(color=C.COLORS["greedy"], dash="dot")))
    fig.add_trace(go.Scatter(x=ds, y=[r["ranking_q_mean"] for r in rows], mode="lines+markers", name="Ranking (Mittel)", line=dict(color=C.COLORS["assign"])))
    fig.update_xaxes(title="Belegungszeit D [min] (ab 121 wird kein Fahrzeug wieder frei)")
    fig.update_yaxes(title="bediente Aufträge / Optimum", range=[0.6, 1.02])
    fig = _base(fig, height)
    fig.update_layout(legend=dict(orientation="h", y=-0.3), height=height + 40)
    return fig


def build_scan_bars(rows, height=300):
    """Angesehene Kanten je Regel (Mittel über Karten)."""
    fig = go.Figure(go.Bar(x=[r["label"] for r in rows], y=[r["scanned"] for r in rows], marker_color=C.COLORS["matched"]))
    fig.update_yaxes(title="angesehene Kanten")
    fig.update_xaxes(tickangle=-25)
    fig = _base(fig, height)
    fig.update_layout(height=height + 90)
    return fig
