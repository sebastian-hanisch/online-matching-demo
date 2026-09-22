# Online-Matching – entscheiden, bevor man alles weiß – Streamlit-Demo

Siebtes Stück der **Matching-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning", ein **unabhängiger Ast neben der Kostenlinie** (Wurzel: [Greedy-Matching](https://github.com/sebastian-hanisch/greedy-matching-demo)):
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Konzept – Zuordnen **ohne Kenntnis der Zukunft** – an einem wachsenden Beispiel.
Die Ungarische Methode kennt alle Aufträge vorab. Hier **kommen die Aufträge nacheinander**, und jede Zusage ist **sofort und unwiderruflich**: ein Fahrzeug wird zugeteilt, bevor der nächste Auftrag bekannt ist. Verglichen wird mit dem **Offline-Optimum** (Ungarische Methode). Drei Regeln: **Greedy** (das billigste freie Fahrzeug), **Ranking** (Karp–Vazirani–Vazirani: eine zufällige feste Rangfolge der Fahrzeuge) und **Batching** (ein Fenster von Aufträgen sammeln und optimal lösen); vier **Ankunftsmodelle** (zufällig, flexible zuerst = gegnerisch, starre zuerst, von links nach rechts); und eine **Zeitdimension**: Fahrzeuge werden nach der Fahrt wieder frei.

**Ziel wie in den Vorgängerdemos lexikografisch:** erst möglichst viele Paare, dann geringe Kosten. Die Güte steht deshalb in **zwei Zahlen**: dem **Gütequotient der Paare** (Online-Paare geteilt durch die Paare des Optimums, je Karte) und der **Prämie** der Kosten gegenüber der billigsten Paarung mit derselben Paarzahl. Ein festes Wettbewerbsverhältnis gibt es nur für die Paare.

**Einordnung in die Reihe (die Kanten des Graphen):** dieses Stück lockert die Annahme „alles ist vorab bekannt“. Sein Preis ist die Unwiderruflichkeit; mit Umhängen (Verbesserungswege je Ankunft) wäre jede Reihenfolge wieder optimal.
```
greedy-matching-demo (Wurzel: eine gewählte Zuordnung bleibt)                     [gebaut]
  ├─ augmenting-path-demo (Verbesserungswege: Paare optimal, Kosten blind)        [gebaut]
  │    ├─ hopcroft-karp-demo (viele kürzeste Wege je Phase)                       [gebaut]
  │    ├─ hungarian-demo (Ungarische Methode: Paare zuerst, dann Kosten)           [gebaut]
  │    │    └─ auction-algorithm-demo (Auktionsalgorithmus: dezentral)             [gebaut]
  │    └─ blossom-demo (allgemeine Graphen: ungerade Kreise, Kontraktion)          [gebaut]
  │        └─ weighted-blossom-demo (Ungarisch + Blossom, Konvergenz)              [gebaut]
  ├─ gale-shapley-demo (Vorlieben statt Kosten, stabil)                            [gebaut]
  │    ├─ stabile-mitbewohner-demo (eine Gruppe statt zwei Seiten)                 [gebaut]
  │    ├─ krankenhaus-zulassung-demo (many-to-one, Kapazitäten)                    [gebaut]
  │    ├─ top-trading-cycles-demo (Tausch ohne Geld, Wohnungsmarkt)                [gebaut]
  │    └─ nierentausch-demo (Kompatibilität statt Präferenz, kurze Zyklen)         [gebaut]
  └─ online-matching-demo (Aufträge kommen nacheinander)                           [dieses Stück]
```

## Ergebnis (Zahlen aus den Tests)

Jede hier genannte Zahl ist in `tests/test_claims.py` über die 100 festen Karten (Seeds 100000–100099) belegt (20 Fahrzeuge, 20 Aufträge, Reichweite 40, zufällige Reihenfolge, wo nichts anderes steht; Ranking als Erwartung über 20 Rangfolgen je Karte). Aufwand = **angesehene Kanten**, nie Sekunden. Mittel und Median stehen zusammen.

| Frage | Ergebnis |
|---|---|
| Greedy auf Zufallskarten | ⚠️ **89,0 %** der möglichen Paare im Mittel (Median 0,900, schlechteste Karte 0,737; optimal nur auf 2 von 100 Karten), Prämie **18,8 %** (Median 19,2 %), 2,13 Paare verloren. Jede maximale Regel findet mindestens die Hälfte – geprüft auf allen 100 Karten unter allen Ankunftsmodellen. |
| Ranking | ⚠️ Paare etwas besser (**91,1 %**, schlechteste Karte 0,847; im Mittel +0,39 Paare gegen Greedy, auf 27 von 100 Karten schlechter), aber **kostenblind**: Prämie **71,0 %** (Median 70,2 %) gegen 18,8 %. Bei Reichweite 150 sind es 164,7 % gegen 22,7 %. |
| Greedy nach Index | ❌ „Kosten-Greedy = Index-Greedy“ ist falsch: dieselbe Paarung auf **0 von 100** Karten, dieselbe Paarzahl nur auf 40 von 100. 4,1 % der Greedy-Entscheidungen haben einen Kostengleichstand; ein umgekehrter Tie-Break ändert die mittlere Paarzahl nicht, aber die Paarung auf 52 von 100 Karten. |
| Batching | ✅ Fenster 1 / 2 / 5 / 10 / 20 (alle): Quotient **0,890 / 0,896 / 0,906 / 0,934 / 1,000**, Prämie 18,8 / 17,3 / 14,6 / 10,9 / 0 %, Aufwand 144 / 214 / 436 / 836 / 1 698 Kanten. Im Mittel monoton – ⚠️ **je Karte nicht**: auf 8 von 100 Karten findet Fenster 5 weniger Paare als Fenster 2. Fenster 1 ist Greedy, ein Fenster über alle Aufträge das Offline-Optimum. Im statischen Modell kostet Warten nichts: das Fenster misst, was Vorausschau wert ist. |
| Gegnerische Reihenfolge | ✅ Flexible zuerst: Greedy im Mittel **0,832** (Median 0,850, schlechteste Karte **0,650**), Prämie im Median 23,9 %. ❌ **Nicht annähernd 1/2**: auch die per Bergsteigen (300 Vertauschungen) gefundene schlechteste Reihenfolge lässt Greedy im Mittel bei 0,826 (schlechteste Karte 0,737; 40 Karten). Starre zuerst 0,947, von links nach rechts 0,896 bei 29,9 % Prämie (Ranking 0,880, Batching 0,925). |
| Die Treppe | ✅ Auftrag t erreicht genau die Fahrzeuge 0 bis n − t − 1. Greedy findet **⌈n/2⌉ von n** Paaren – ohne Tie-Break, die Kosten steigen streng – und starre zuerst wären perfekt (n). Ranking im Erwartungswert (exakt über alle Rangfolgen): 13/6, 67/24, 137/40, 4,0569, 4,6891, **5,3212** für n = 3 bis 8 (0,665 bei n = 8); Monte Carlo (2000 Rangfolgen) 0,6548 / 0,6454 / 0,6398 / 0,6358 / **0,6334** für n = 10 / 20 / 40 / 80 / 160: von oben gegen **1 − 1/e ≈ 0,632**. Batching mit Fenster 1 / 2 / 5 / 10 bleibt bei n = 20 auf 10 Paaren, erst das Fenster 20 bringt 20. |
| Kurze Reichweite | ✅ Reichweite 10: Greedy **optimal auf 85 von 100 Karten** (Mittel 0,980, Median 1,000), Prämie 5,8 % (Median 2,1 %): wo es kaum Alternativen gibt, kann eine falsche Wahl wenig anrichten. |
| Alles erreichbar | ✅ Reichweite 150: Greedy findet auf **allen** 100 Karten alle 20 Paare – der Verlust liegt allein bei den Kosten (Prämie im Median 21,2 %, Mittel 22,7 %), Ranking im Median 160 %. |
| Ungleich große Seiten | ⚠️ 10 Fahrzeuge, 20 Aufträge: Greedy findet auf 87 von 100 Karten alle Paare (Mittel 0,985), Prämie im Median 49,3 %. Bei 20 Fahrzeugen und 10 Aufträgen schlagen Ranking (0,991) und Greedy nach Index (0,997) Greedy (0,985) bei den Paaren – bei einer Prämie von im Mittel 87,0 % bzw. 85,8 % (Greedy: 7,9 %). |
| Mit Zeit | ✅ 10 Fahrzeuge, 30 Aufträge in 120 Minuten, Belegungszeit D: Greedy bedient von den im Nachhinein bedienbaren Aufträgen im Mittel **0,988 / 0,952 / 0,915 / 0,922 / 0,946 / 0,998** bei D = 1 / 10 / 20 / 30 / 60 / 121 (schlechteste Karte 0,867 / 0,767 / 0,741 / 0,793 / 0,812 / 0,900): ein **Buckel**. Der Buckel wandert mit der Last: bei 20 Fahrzeugen und 40 Aufträgen liegt das Minimum bei D = 40 (0,938). Bei Reichweite 150 wird jeder bedienbare Auftrag bedient, aber mit 36,9 statt 23,1 Minuten Fahrzeit je Auftrag (Ranking 53,3). |
| Aufwand | ⚠️ Angesehene Kanten (40 Karten): Greedy 73,6, Batching 144,5 / 215,5 / 436,3 / 834,7 (Fenster 1 / 2 / 5 / 10), Fenster 20 (alles) 1 676,0, Offline 1 685,7. |

## Was nicht funktioniert hat / widerlegte Vorab-Hypothesen

- **„Greedy bekommt 1/2.“** Das hängt an der Konstruktion, nicht am Graphen allein: auf der oberen Dreiecksmatrix mit gleichen Kosten findet Greedy mit dem kleinsten Index n/2, mit dem größten Index aber **alle n** – der Tie-Break entscheidet. Die Falle muss **geometrisch** gebaut sein (nächstes freies Fahrzeug = flexibelstes, Kosten streng steigend); dann hilft auch ein umgekehrter Tie-Break nicht. Auf Zufallskarten gibt es die 1/2 nicht.
- **„Eine gegnerische Reihenfolge drückt Greedy auf Zufallskarten Richtung 1/2.“** Nein: flexible zuerst 0,832, die per Bergsteigen gefundene schlechteste Reihenfolge 0,826 im Mittel; keine Karte fällt unter 1/2.
- **„Ranking ist ein Gewinn.“** Bei den Paaren ein kleiner (+0,39 Paare bei Reichweite 40, sonst kaum ein Unterschied, auf 27 % der Karten sogar ein Rückschritt), bei den Kosten ein großer Verlust: es ignoriert sie.
- **„Ein größeres Fenster ist nie schlechter.“** Im Mittel ja, je Karte nein (8 von 100), und auf der Treppe ändert kein Fenster unter n etwas.
- **„Ein Wettbewerbsverhältnis für die Kosten.“** Bei lexikografischem Ziel schlecht definiert: die Paarzahlen unterscheiden sich. Deshalb der Quotient nur für die Paare und die Prämie gegen die billigste Paarung derselben Paarzahl.
- **„Ankunft von links nach rechts nutzt die Lokalität.“** Für die Paare von Greedy kaum (0,896 gegen 0,890), bei den Kosten schlechter (29,9 % gegen 18,8 %); Ranking wird schlechter, nur Batching profitiert (0,925).
- **„Batching gehört auch ins Modell mit Zeit.“** Nein: eine verzögerte Zuordnung kann einen Auftrag noch bedienen, den der Sofort-Maßstab nicht kennt (im Test übertrifft sie ihn auf einzelnen Karten). Dort gibt es nur Greedy und Ranking.
- **Abgrenzung:** Umhängen erlaubt (ein Verbesserungsweg je Ankunft) erreicht auf jeder Reihenfolge die optimale Paarzahl – die Negativkontrolle zeigt, dass die Unwiderruflichkeit der Preis ist. Brute Force nur als Testorakel (kleine Karten, alle Reihenfolgen; der Offline-Fluss gegen alle (n + 1)^m Zuordnungen auf 300 Karten).

## Was die Demo zeigt

- **Regel und Ankunft:** Schritt-Slider und ▶️ über die **Ankünfte**: die Karte (künftige Aufträge hellgrau, zugeordnete orange, wartende violett, abgelehnte mit rotem Kreuz), beim letzten Ereignis die freien Fahrzeuge in Reichweite und das gewählte; daneben der Verlauf gegen die **im Nachhinein bestmögliche Paarzahl** derselben ersten Aufträge (der Abstand ist der Preis der Unwiderruflichkeit) und die Optionen mit Fahrzeit. Am Ende die **unabhängige Prüfung** des Ergebnisses (Regel neu nachgerechnet). Modell (ein Fahrzeug fährt einmal / Fahrzeuge werden wieder frei), Regel, Ankunft, Fenster, Rangfolge, Belegungszeit.
- **Wie gut?** Paare gegen das Optimum, Gütequotient, Kosten mit Prämie, angesehene Kanten; Vergleichstabelle Greedy / Ranking / Batching / Offline auf derselben Reihenfolge; Verteilung über 100 feste Karten (Mittel, Median, schlechteste Karte, Histogramme). Mit Zeit: bediente Aufträge gegen den Fluss im Nachhinein und Fahrzeit je Auftrag.
- **Wovon hängt es ab?** Fenster-Sweep (was ist Vorausschau wert), die Treppe (Greedy gegen Ranking bis n = 160), Ankunftsmodelle im Vergleich (mit der gefundenen schlechtesten Reihenfolge), Belegungszeit-Sweep, Aufwand.
- **Feste Karten:** die Treppe (8 × 8) und ein Pfad aus vier Punkten; **Wo die Annahmen enden.**

## Modell und Verfahren

- **Statisch:** Fahrzeuge vorab bekannt, ein Fahrzeug fährt höchstens einen Auftrag; beim Eintreffen sieht die Regel nur die Zeile des Auftrags. Alle Regeln bedienen, sobald ein mögliches freies Fahrzeug existiert. Greedy: billigstes, Gleichstand kleinster Index. Ranking: zufällige Rangfolge aus `SplitMix64`, Seed aus (Kartenseed, Ziehung). Batching(w): alle w Ankünfte löst die Ungarische Methode den Stapel für die noch freien Fahrzeuge, wer leer ausgeht, wird endgültig abgelehnt. Der Ankunftsstrom ist vom Kartenstrom getrennt.
- **Mit Zeit:** Auftrag j trifft in der ganzzahligen Minute a_j ein (aufsteigend, Horizont 120). Ein zugeordnetes Fahrzeug ist ab der Zuordnung genau D Minuten belegt (die Fahrzeit ist nur Kosten) und steht danach am Ort des Auftrags. **Maßstab im Nachhinein** (Zuordnung bei Ankunft, feste Belegung): Min-Cost-Flow Quelle → Fahrzeug → (Auftrag_ein → Auftrag_aus → nächster Auftrag …) → Senke, Kosten −L mit L = 1 + m · 142 je bedientem Auftrag, eine Flusseinheit je Fahrzeug, kürzeste Wege per Bellman-Ford. Getestet gegen Brute Force, networkx und die statische Ungarische Methode (D > Horizont).
- **Gütequotient und Prämie:** Quotient = Online-Paare / Optimum-Paare je Karte; Prämie = (Kosten − Σ_{i≤k} W_i) / Σ_{i≤k} W_i mit den Grenzkosten W_i der Ungarischen Methode.
- **Die Treppe:** Fahrzeuge bei x = 30 + 6 i, Auftrag t liegt links davon und erreicht genau die Fahrzeuge 0 … n − t − 1 (Reichweite 6 n, bis n = 12 auf der 100 × 100-Karte; größere n als abstrakter Graph). Ranking-Erwartung exakt über alle Rangfolgen bis n = 8, darüber Monte Carlo.

## Dateien

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `om_constants.py` | Regler-Grenzen, Presets und Hilfetexte, feste Seed-Mengen |
| `om_presets.py` | Permalink, Preset- und Zufalls-Seed-Logik (Standardmuster des Portfolios, kopiert und um Regel, Ankunft, Fenster, Rangfolge, Modell und Belegungszeit ergänzt) |
| `om_scenario.py` | Karten, eigener Zufallsgenerator (aus den Vorgängerdemos kopiert), dazu **neu:** Ankunftsströme, Ankunftsminuten, die Treppe, der Pfad aus vier Punkten |
| `om_greedy.py`, `om_augment.py`, `om_hungarian.py` | Greedy-Regeln, Verbesserungswege und Ungarische Methode aus den Vorgängerdemos (kopiert, ohne Import; die Ungarische Methode liefert Optimum, Grenzkosten und den Batch-Löser) |
| `om_online.py` | **Neu:** Greedy, Ranking, Batching, Ereignisprotokoll, Zustand nach k Ankünften, unabhängiger Prüfer |
| `om_dynamic.py` | **Neu:** das Modell mit Zeit, Min-Cost-Flow als Maßstab, Prüfer |
| `om_evaluation.py` | Einordnung, Quotient, Prämie, Verteilungen, Sweeps, Treppe, Aufwand |
| `om_visualization.py` | Plotly-Abbildungen (Achsen gesperrt für Touch-Geräte; keine Legende auf den Karten, `constrain="domain"`, da gleicher Maßstab in schmalen Spalten die Zeichenfläche verdrängen kann) |
| `tests/` | Regeln gegen unabhängige Nachbauten, Brute Force, Identitäten (Fenster 1 = Greedy, Fenster ≥ m = Offline), die Treppe (exakte Brüche, Monte Carlo), Negativkontrollen (Umhängen, feste Index-Rangfolge, verzögertes Batching), das Modell mit Zeit gegen Brute Force und networkx, belegte Zahlen, AppTest-Rauchtests (auch Abspielen auf mehrbildrigen Karten für jede Regel und jedes Modell) |

Die Kopien der Vorgänger werden durch Tests bewacht (Zufallsgenerator-Vektor, Seed-2-Karte: 17 / 232 / 20 / 316). Alle Daten sind synthetisch; die Laufzeit braucht nur numpy, pandas, plotly und streamlit (scipy und networkx sind reine Testorakel).

## Lokal starten

```bash
python -m venv venv
venv\Scripts\pip install -r requirements.txt
venv\Scripts\streamlit run app.py
```

## Tests ausführen

```bash
venv\Scripts\pip install -r requirements-dev.txt
venv\Scripts\python -m pytest tests -v
```

Die Logik rechnet ausschließlich mit ganzen Zahlen; die im Text genannten Anteile und Mittelwerte sind deshalb auf jeder Plattform identisch.
Die CI (`.github/workflows/tests.yml`) läuft auf Ubuntu mit Python 3.12, bei jedem Push und wöchentlich mit den jeweils neuesten Bibliotheksversionen.
