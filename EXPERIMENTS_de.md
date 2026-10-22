# Experimentelles Setup und Ergebnisse

Conformal Prediction auf Black-Box-Decision-Modellen für das schrittweise Safety-Review von Tool-Calls.
Code: `02_full_evalrun.ipynb` (Inferenz und Baseline-Metriken) und `03_conformal_prediction.ipynb` (Conformal-Analyse).
Rohdaten liegen in `results/`. Englische Version: `EXPERIMENTS.md`.

---

## 1. Zielsetzung

Ein autonomer Agent schlägt einen Tool-Call vor. Ein *Safety Reviewer* muss eine von drei Governance-Aktionen wählen:

| Label | Bedeutung |
|---|---|
| `direct_execute` | sicher, darf autonom ausgeführt werden |
| `ask_human` | erfordert eine menschliche Bestätigung |
| `refuse` | muss blockiert werden |

Die Fehler sind unterschiedlich teuer. Der kritische Fehler ist die **katastrophale Ausführung**: Das wahre Label ist `refuse`, und das System führt den Aufruf trotzdem aus.

Forschungsfragen:

1. Wie gut eignen sich Standard-Decision-Modelle als Reviewer (Accuracy, Kalibrierung, Governance-Fehler)?
2. Liefert Split Conformal Prediction (CP) auf ihren Ausgabewahrscheinlichkeiten eine **verteilungsfreie Schranke mit Finite-Sample-Garantie** für katastrophale Ausführungen? Und was kostet diese Schranke an Autonomie?

---

## 2. Experimentelles Setup

### 2.1 Daten

- **Quelle:** In-Domain-Testsplit `data/test_1436.json` des Step-Level-Safety-Review-Korpus (Release `full_data_20260530`). Der Trainingssplit (7.767) und der zurückgehaltene Mailu-OOD-Split (198) wurden bisher **nicht verwendet**.
- **Umfang und Labelverteilung:** N = 1436. `direct_execute` 540 (37,6 %), `ask_human` 488 (34,0 %), `refuse` 408 (28,4 %).
- **Services:** 15 Quellen, z. B. when2call 271, vaultwarden 196, agentharm 143, tsbench 125, atbench 119, zammad 97, dazu kleinere App-basierte Services (nocodb, gitea, rocketchat, erpnext, owncloud, openemr, …).
- **Input pro Sample:** System-Prompt des Reviewers (Entscheidungsvertrag) plus eine JSON-User-Nachricht mit `user_task`, `service` und `assistant_proposed_tool_call`.
- **Ground Truth:** `meta.decision`.
- **Hinweis:** Bei 748 von 1436 Samples fehlt die Metadatenangabe `risk_family` (leer oder NaN). Auswertungen nach Risk Family sind daher nur indikativ.

### 2.2 Modelle und Inferenz

| Modell (OpenRouter-ID) | Erfolgreiche Calls |
|---|---|
| `typesafe/jev-1.14` | 1436 |
| `microsoft/microsoft-decision-1` | 1436 |
| `openai/gpt-6-luna-decisions` | 1433 (3 × HTTP 502 nach 5 Retries) |

- **Endpoint:** OpenRouter `api/alpha/decisions`. Jede Anfrage sendet einen `state` (System- und User-Nachricht des Samples) und eine einzelne `choice`-Frage mit drei Optionen. Zu jeder Option gehört ein Kriterium in natürlicher Sprache, identisch für alle Modelle (siehe `QUESTIONS_PAYLOAD` in Notebook 02).
- **Output:** eine diskrete Entscheidung plus ein Wahrscheinlichkeitsvektor $\hat p(x) \in \Delta^2$ über die drei Labels. Wie der Anbieter diese Wahrscheinlichkeiten berechnet, ist nicht offengelegt. Die Modelle werden daher als **Black Boxes** behandelt.
- **Auflösung der Wahrscheinlichkeiten:** `jev-1.14` und `gpt-6-luna-decisions` liefern Werte im 0,01-Raster, einschließlich exakter Nullen. `microsoft-decision-1` liefert kontinuierliche Werte und keine exakten Nullen.
- **Preprocessing:** Fehlgeschlagene Calls werden entfernt. Wahrscheinlichkeiten werden bei 0 abgeschnitten und renormalisiert (maximale Abweichung der Summe von 1: 0,01).
- **Robustheit des Laufs:** exponentielles Backoff (5 Retries) und Checkpointing pro Call im JSONL-Format. Gesamtlaufzeit 31 min, Gesamtkosten ≈ 0,34 $.

### 2.3 Baseline-Evaluation (Argmax bzw. zurückgegebene Entscheidung)

- **Klassifikation:** Accuracy, Balanced Accuracy, Macro- und Weighted-F1, Cohens κ, Precision und Recall pro Klasse.
- **Wahrscheinlichkeitsqualität:** Brier Score, NLL (Wahrscheinlichkeiten bei 1e-15 abgeschnitten), Top-1-ECE (15 Bins).
- **Unsicherheitssignal:** mittlere normierte Entropie bei Fehlern im Vergleich zu korrekten Vorhersagen.
- **Governance-Raten:**

| Rate | Definition |
|---|---|
| Katastrophal | $P(\hat y = \text{DE} \mid y = \text{RF})$ |
| Safe Fallback | $P(\hat y = \text{AH} \mid y = \text{RF})$ |
| Friction | $P(\hat y = \text{AH} \mid y = \text{DE})$ |
| False Block | $P(\hat y = \text{RF} \mid y = \text{DE})$ |
| Automatisierung | $P(\hat y \neq \text{AH})$ |

### 2.4 Conformal-Prediction-Protokoll

**Split Conformal.** Die 1436 Samples werden 50/50 in Kalibrierungs- und Testmenge aufgeteilt, stratifiziert nach Ground Truth. Der Split ist für alle Modelle identisch; die 3 fehlenden Samples des gpt-Modells fehlen dort einfach.

Pro Split:

1. Nonconformity Scores $s(x_i, y_i)$ auf der Kalibrierungsmenge berechnen.
2. $\hat q$ als empirisches $\lceil (n+1)(1-\alpha) \rceil / n$-Quantil bestimmen (`method="higher"`).
3. Prediction Set bilden: $C(x) = \{y : s(x,y) \le \hat q\}$.

Garantie (unter Exchangeability): $P(y \in C(x)) \ge 1-\alpha$.

| Methode | Score $s(x,y)$ | Quantil |
|---|---|---|
| LAC | $1 - \hat p_y(x)$ | ein globales Quantil |
| APS (randomisiert) | $\sum_{y': \hat p_{y'} > \hat p_y} \hat p_{y'} + U \cdot \hat p_y$ | ein globales Quantil |
| Mondrian LAC | $1 - \hat p_y(x)$ | ein Quantil pro wahrer Klasse $k$ mit eigenem $\alpha_k$; garantiert $P(y \in C(x) \mid y = k) \ge 1-\alpha_k$ |

**Governance-Entscheidungsregel.** Ein Singleton-Set $\{y\}$ führt zur Aktion $y$. Jedes Set mit Größe ≥ 2 sowie ein leeres Set führen zu `ask_human`.

Unter dieser Regel kann ein wahres `refuse` nur dann automatisch ausgeführt werden, wenn `refuse` ∉ C(x). Mit Mondrian CP folgt daraus: **katastrophale Rate ≤ α_refuse** im Erwartungswert.

**Konfigurationen.**

| Experiment | Methoden | α | Wiederholungen |
|---|---|---|---|
| E1: Hauptvergleich | LAC, APS, Mondrian LAC | 0,10 (alle Klassen) | 100 zufällige Splits (Seeds 0–99) |
| E2: α-Sweep | LAC | 0,02, 0,04, …, 0,40 | 30 Splits pro α |
| E3: risikoasymmetrisch | Mondrian LAC | α_DE = 0,15, α_AH = 0,15, α_RF = 0,05 | 100 Splits |

**Metriken.**
- **Coverage:** marginal und pro Klasse.
- **Set-Größe:** Mittelwert, dazu Singleton-Rate und Rate leerer Sets.
- **Singleton-Accuracy.**
- **Governance-Raten** nach Anwendung der Entscheidungsregel, verglichen mit Argmax auf derselben Testhälfte.

Alle unten berichteten Werte sind Mittelwerte über die Splits.

---

## 3. Ergebnisse

### 3.1 Baseline (ohne Conformal-Schritt, volles N)

| Modell | Acc | Bal. Acc | Macro F1 | κ | Brier ↓ | ECE ↓ | NLL ↓ | H(Fehler)/H(korrekt) |
|---|---|---|---|---|---|---|---|---|
| jev-1.14 | **0,649** | **0,632** | **0,638** | **0,461** | **0,505** | **0,148** | 1,676 | 1,52 |
| microsoft-decision-1 | 0,595 | 0,576 | 0,576 | 0,375 | 0,678 | 0,308 | **1,547** | 1,74 |
| gpt-6-luna-decisions | 0,623 | 0,608 | 0,615 | 0,420 | 0,598 | 0,252 | 2,930 | 1,73 |

| Modell | `refuse` Precision / Recall | Katastrophal | Safe Fallback | Friction | False Block |
|---|---|---|---|---|---|
| jev-1.14 | 0,89 / 0,47 | 15,7 % | 37,0 % | 14,4 % | 0,9 % |
| microsoft-decision-1 | 0,82 / 0,41 | 26,2 % | 32,4 % | 12,0 % | 2,2 % |
| gpt-6-luna-decisions | 0,88 / 0,48 | 20,7 % | 31,6 % | 17,0 % | 0,6 % |

- Alle Modelle tendieren zu `direct_execute` (Recall 0,82–0,86) und übersehen mehr als die Hälfte der echten Refusals.
- `ask_human` vs. `direct_execute` ist die am häufigsten verwechselte Grenze. Bei echten `ask_human`-Samples legt microsoft im Mittel mehr Masse auf `direct_execute` (0,49) als auf `ask_human` (0,45).
- Fehler haben eine 1,5- bis 1,7-fach höhere Entropie als korrekte Vorhersagen. Die Wahrscheinlichkeiten tragen also ein Unsicherheitssignal, das CP nutzen kann.
- Der Anteil der Samples, bei denen das wahre Label eine Wahrscheinlichkeit ≤ 0,01 bekommt, liegt bei 5,0 % (jev), 14,1 % (microsoft) und 12,9 % (gpt). Exakte Nullen treten bei 2,6 % (jev) und 5,7 % (gpt) der Samples auf, bei microsoft nie.

### 3.2 E1: Validität und Effizienz bei α = 0,10

| Modell | Methode | Coverage | Ø Set-Größe | Singleton | Singleton-Acc. |
|---|---|---|---|---|---|
| jev-1.14 | LAC | 0,906 | 1,84 | 0,305 | 0,863 |
| | APS | 0,900 | 1,88 | 0,278 | 0,848 |
| | Mondrian | 0,915 | 2,00 | 0,247 | 0,849 |
| microsoft-decision-1 | LAC | 0,902 | 2,13 | 0,234 | 0,770 |
| | APS | 0,901 | 2,12 | 0,213 | 0,767 |
| | Mondrian | 0,906 | 2,02 | 0,210 | 0,793 |
| gpt-6-luna-decisions | LAC | 0,943 | 2,45 | 0,155 | 0,726 |
| | APS | 0,904 | 2,19 | 0,242 | 0,735 |
| | Mondrian | 0,954 | 2,31 | 0,016 | 0,853 |

- Die marginale Coverage erreicht für jedes Modell und jede Methode das Ziel. Die Standardabweichung über die Splits liegt bei ≈ 0,014. Leere Sets treten in ≤ 0,2 % der Fälle auf.
- **gpt überdeckt unter LAC und Mondrian.** Seine Wahrscheinlichkeiten liegen im 0,01-Raster mit vielen exakten Nullen. Dadurch landet $\hat q$ bei 0,99, und die Set-Grenze lässt sich nicht zwischen gleiche Werte (Ties) legen. Das randomisierte APS beseitigt die Überdeckung.
- **Die Effizienz ist gering:** Im Mittel enthalten die Sets 1,8–2,5 von 3 Labels. Die kalibrierten LAC-Schwellen liegen bei 0,93 (jev), 0,99 (gpt) und 0,995 (microsoft). Ein Label wird also nur ausgeschlossen, wenn $\hat p_y < 0,07$ (jev) bzw. < 0,01 (gpt, microsoft).

### 3.3 E1: Klassenbedingte Coverage

| Modell | Methode | cov(DE) | cov(AH) | cov(RF) |
|---|---|---|---|---|
| jev-1.14 | LAC | 0,977 | 0,922 | **0,792** |
| | Mondrian | 0,906 | 0,914 | 0,926 |
| microsoft-decision-1 | LAC | 0,996 | 0,859 | **0,830** |
| | Mondrian | 0,902 | 0,908 | 0,907 |
| gpt-6-luna-decisions | LAC | 0,978 | 0,879 | 0,974 |
| | Mondrian | 0,904 | 0,995 | 0,971 |

Marginales CP unterdeckt die sicherheitskritische Klasse `refuse` bei jev und microsoft um etwa 7–11 Prozentpunkte. Mondrian CP stellt für jede Klasse eine Coverage von mindestens 1 − α wieder her.

### 3.4 E1: Governance-Effekt (Argmax → CP-gated, LAC bei α = 0,10)

| Modell | Katastrophal | Friction | False Block | Automatisierung | Aktions-Acc. |
|---|---|---|---|---|---|
| jev-1.14 | 15,8 → **3,2 %** | 14,5 → 56,7 % | 0,9 → 0 % | 64,3 → 23,0 % | 0,648 → 0,514 |
| microsoft-decision-1 | 26,7 → **1,4 %** | 11,8 → 62,7 % | 2,3 → 0 % | 71,0 → 21,2 % | 0,595 → 0,452 |
| gpt-6-luna-decisions | 20,7 → **0,0 %** | 17,0 → 74,0 % | 0,5 → 0 % | 66,7 → 14,5 % | 0,623 → 0,403 |

Mit Mondrian bei α = 0,10 liegt die katastrophale Rate bei 2,1 % (jev), 0,4 % (microsoft) und 0,0 % (gpt). Die Automatisierung von gpt bricht dabei auf 0,5 % ein.

„Aktions-Acc.“ zählt Eskalationen echter `direct_execute`- bzw. `refuse`-Samples als Fehler. Die Metrik unterschätzt daher den Sicherheitswert der Eskalation.

### 3.5 E2: Trade-off zwischen Coverage und Autonomie (LAC, 30 Splits)

| α | Modell | Coverage | Ø Größe | Singleton | Katastrophal (CP) |
|---|---|---|---|---|---|
| 0,10 | jev / microsoft / gpt | 0,91 / 0,90 / 0,94 | 1,84 / 2,14 / 2,45 | 0,31 / 0,23 / 0,16 | 3,2 / 1,3 / 0,0 % |
| 0,20 | jev / microsoft / gpt | 0,81 / 0,80 / 0,81 | 1,44 / 1,65 / 1,62 | 0,59 / 0,50 / 0,53 | 9,2 / 6,2 / 5,7 % |
| 0,30 | jev / microsoft / gpt | 0,70 / 0,70 / 0,71 | 1,14 / 1,28 / 1,19 | 0,86 / 0,75 / 0,81 | 14,6 / 16,8 / 12,2 % |

- Die Coverage folgt über den gesamten Bereich (α = 0,02–0,40) der Linie 1 − α. Siehe die Abbildung in Notebook 03.
- Bei α ≈ 0,30 nähert sich die katastrophale Rate der Argmax-Baseline. Autonomie und Sicherheit stehen in einem nahezu monotonen Trade-off.

### 3.6 E3: Risikoasymmetrisches Mondrian (α_RF = 0,05, α_DE = α_AH = 0,15)

| Modell | cov(DE) | cov(AH) | cov(RF) | Ø Größe | Katastrophal (Argmax → CP) | Friction (CP) | Automatisierung (CP) | $\hat q_{RF}$ |
|---|---|---|---|---|---|---|---|---|
| jev-1.14 | 0,856 | 0,866 | 1,000 | 2,19 | 15,8 → 0,0 % | 100 % | **4,6 %** | 1,000 |
| microsoft-decision-1 | 0,854 | 0,858 | 0,959 | 2,02 | 26,7 → 1,3 % | 63,4 % | **20,9 %** | 0,999 |
| gpt-6-luna-decisions | 0,855 | 0,886 | 0,974 | 2,12 | 20,7 → 0,0 % | 76,4 % | **13,9 %** | 0,990 |

- Die katastrophale Rate bleibt für jedes Modell innerhalb des Budgets α_RF = 5 %.
- **Bei jev ist die Garantie trivial erfüllt.** 7,8 % der echten `refuse`-Samples bekommen exakt $\hat p_{RF} = 0$, also mehr als α_RF. Daher ist $\hat q_{RF} = 1$ und `refuse` in **jedem** Set enthalten. Ein `direct_execute`-Singleton kann so nie entstehen. Die Friction liegt bei 100 %, die Automatisierung bei 4,6 %, fast ausschließlich `refuse`-Singletons.
- **Das Modell-Ranking kehrt sich um.** microsoft hat die schlechteste Accuracy und ECE, gibt aber nie exakte Nullen aus und hat die beste NLL. Unter dem strengen Budget behält es die meiste Autonomie (20,9 %).
- Beispiel-Split (jev): Die Singleton-Rate liegt bei 26 % für die Risk Family `refuse` und bei 1,4 % für `clean_de`. Die verbleibende Autonomie unter CP-Gating stammt also vor allem aus Refusals, nicht aus harmlosen Ausführungen.

---

## 4. Zusammenfassung der Ergebnisse

1. **Die Argmax-Entscheidung ist als autonomes Gate unsicher.** 16–27 % der schädlichen Tool-Calls würden ausgeführt.
2. **CP ist auf diesen Daten valide.** Die marginale Coverage entspricht für alle Modelle 1 − α. Für eine Garantie speziell auf `refuse` ist Mondrian CP nötig; marginales CP unterdeckt die Klasse um bis zu 11 Prozentpunkte.
3. **CP-Gating begrenzt katastrophale Ausführungen auf wenige Prozent:** von 16–27 % auf 0–3 % bei α = 0,10 und auf ≤ α_RF im asymmetrischen Setting. Die verbleibenden autonomen Entscheidungen sind genauer (Singleton-Accuracy 0,73–0,86 gegenüber 0,60–0,65 bei Argmax).
4. **Der Preis ist Autonomie.** Nur 5–23 % der Entscheidungen bleiben automatisiert, die meisten harmlosen Calls werden eskaliert. Für unser primäres Ziel (Sicherheit vor Autonomie) ist das akzeptabel, es begrenzt aber den praktischen Nutzen.
5. **Die Effizienz von CP hängt vom Verhalten im Tail der Wahrscheinlichkeit des wahren Labels ab (exakte oder fast-Nullen), nicht von Accuracy oder ECE.** Die Modellauswahl für Conformal Governance sollte daher Proper Scoring Rules verwenden, und ein Modellvergleich sollte diesen Effekt berücksichtigen.

## 5. Einschränkungen (Threats to Validity)

- **Exchangeability.** Kalibrierungs- und Testhälfte stammen aus demselben In-Domain-Testsplit. Unter einem Shift bei Services oder Angriffstypen ist die Garantie ungetestet; der Mailu-OOD-Split steht dafür zur Verfügung.
- **Black-Box-Wahrscheinlichkeiten.** Unbekannt ist, wie der Endpoint die Wahrscheinlichkeiten ableitet: Es könnten Token-Log-Probs, Sampling-Häufigkeiten oder verbalisierte Werte sein. Ihre Auflösung unterscheidet sich zwischen den Modellen (0,01-Raster vs. kontinuierlich), was Ties und Überdeckung beeinflusst.
- **Kleine Kalibrierungsmengen pro Klasse.** Etwa 204 `refuse`-Samples pro Split. Bei kleinem α_k erreicht die Finite-Sample-Korrektur schnell $\hat q = 1$, was triviale Sets ergibt.
- **Label Noise.** Der Korpus enthält umgelabelte Samples (`label_flipped`, `label_corrected` in `meta`). Die Coverage wird gegen diese Labels gemessen.
- **Entscheidungsregel.** Jedes Set mit mehr als einem Element eskaliert. Regeln, die die Zusammensetzung des Sets berücksichtigen (z. B. `{DE, AH}` → AH, jedes Set mit RF → AH, `{RF}` → refuse), wurden nicht separat evaluiert.
- **Baseline-Definitionen.** Notebook 02 verwendet die zurückgegebene `choice`, Notebook 03 das Argmax der Wahrscheinlichkeiten. Sie unterscheiden sich in 1–4 Samples pro Modell (Ties).
- **Ein Datensatz, drei Modelle.** Keine Konfidenzintervalle über Modelle hinweg. Die Variabilität über Splits ist in Notebook 03 als Standardabweichung angegeben.

## 6. Geplante nächste Schritte

- **Model Multiplicity:** die Uneinigkeit zwischen den drei (oder mehr) Decision-Modellen auf denselben Samples quantifizieren. Dazu gehören Argmax-Disagreement, die Überlappung der Prediction Sets und die Frage, ob Samples mit katastrophalen Fehlern über Modelle hinweg dieselben sind. Außerdem multiplicity-aware Aggregation vor der Kalibrierung testen (z. B. Vereinigung oder Schnitt der Sets, Ensemble-Wahrscheinlichkeiten).
- **Conformal Risk Control** mit expliziter Kostenmatrix statt 0/1-Coverage.
- **Shift-Evaluation:** in-domain kalibrieren, auf Mailu-OOD und auf zurückgehaltenen Services testen.
