# Wine Quality Dataset — What, Why & How

## What Is This Dataset?

**Name:** UCI Wine Quality (Red Wine)
**Source:** UC Irvine Machine Learning Repository
**URL:** https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
**Format:** CSV (semicolon-separated)
**Size:** 1,599 wine samples × 12 columns

---

## The Story Behind It

A wine producer in Portugal wanted to know:
> "Can we predict wine quality from chemistry — without paying expert tasters every time?"

So they took 1,599 bottles of red wine, ran lab tests on each one, and also had wine experts score them (0–10). This dataset is the result — chemical measurements paired with quality scores.

---

## What's Inside (The 11 Features)

| Column | What It Measures | Why It Matters |
|--------|-----------------|----------------|
| `fixed acidity` | Non-evaporating acids (tartaric acid) | Gives wine structure and tartness |
| `volatile acidity` | Acetic acid (vinegar-like) | Too high = wine tastes bad |
| `citric acid` | Freshness acid | Adds freshness, fruity notes |
| `residual sugar` | Sugar left after fermentation | Controls sweetness |
| `chlorides` | Salt content | Too much = salty, unpleasant |
| `free sulfur dioxide` | SO₂ not yet reacted | Prevents microbial growth |
| `total sulfur dioxide` | All SO₂ (free + bound) | Preservative, regulated by law |
| `density` | Mass per volume | Reflects sugar + alcohol balance |
| `pH` | Acidity level (0–14 scale) | Low pH = more acidic = longer shelf life |
| `sulphates` | Potassium sulphate additive | Antimicrobial, affects taste |
| `alcohol` | % alcohol by volume | Biggest quality predictor |

**Target column:** `quality` — expert rating from 3 to 8 (most wines score 5 or 6)

---

## The Problem We Are Solving

### Without ML:
- Send each bottle to a certified wine taster
- Wait days for results
- Costs money per batch
- Human tasters have off-days (subjective)

### With This Pipeline:
- Run cheap lab tests (11 measurements)
- Feed numbers into trained model
- Get quality prediction in milliseconds
- **No human taster needed**

---

## How We Transformed It

Raw dataset has quality scores 3–8. We converted to **binary classification**:

```
quality >= 7  →  label = 1  (High Quality)  ← 217 wines (13.6%)
quality <  7  →  label = 0  (Low Quality)   ← 1382 wines (86.4%)
```

Why binary? Easier to act on — reject or approve a batch. No need to predict exact score.

---

## What the Model Learned

Top predictors of high quality wine (ranked by importance):

```
1. alcohol          (0.174) ← more alcohol = usually better quality
2. sulphates        (0.111) ← higher sulphates = better preservation
3. density          (0.103) ← lower density = more alcohol, less sugar
4. volatile acidity (0.102) ← lower = less vinegar taste = better
5. citric acid      (0.093) ← more = fresher taste
```

**Simple rule nature found:** High alcohol + low volatile acidity + good sulphates = high quality wine.

---

## Why This Dataset Is Good For This Pipeline

| Reason | Detail |
|--------|--------|
| Clean data | Zero null values — no messy cleaning needed |
| Real-world | Actual industry problem, not toy data |
| CSV format | Simple to ingest, no API or auth needed |
| Right size | 1,599 rows — fast to train, meaningful results |
| Open license | Freely downloadable, no restrictions |
| Tabular | Perfect for Random Forest, clear feature importance |

---

## Final Result

Model trained on this dataset achieves:
- **94.06% accuracy** on unseen test data
- **0.955 ROC-AUC** — near perfect discrimination
- Deployed as `models/rf_model.pkl` — ready to predict new wines

Feed any new wine's 11 lab measurements → model says High or Low quality instantly.
