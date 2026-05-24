# Project Talking Points

## The 20-second pitch (memorize this)
> "I wanted to understand the problems your team works on, so I built a claim-frequency model on a
> public motor-insurance dataset — the frequency half of pricing a policy. The most interesting part
> was the evaluation: since claims are rare, accuracy is misleading, so I focused on precision/recall
> and on which factors actually drive risk. I also wrapped it in a small Streamlit scorer to make it
> tangible."

## Why this project (if they ask why you built it)
- Wanted to learn the actual problem space before the interview, not just talk about it.
- Auto insurance is Farmers' founding business — felt like the right place to start.

## The framing line that signals business sense
> "Insurers price a policy as **frequency × severity**. I modeled the frequency half — the probability
> a policy files at least one claim."

## The single highest-signal point (say it unprompted)
> "Claims are rare — only about 5–10% of policies claim in a year. So **accuracy is the wrong metric**:
> a model that predicts 'no claim, ever' is ~90%+ accurate and useless. I evaluated on ROC-AUC,
> precision, and recall, and the right decision threshold depends on the **business cost** of a false
> positive versus a missed risk."

## Why two models (shows you pick the right tool, not the fanciest)
> "I used a **GLM (logistic regression)** as an interpretable baseline because insurers favor explainable
> models for **regulatory reasons**, then checked whether **XGBoost** bought enough lift to justify lower
> interpretability. For a regulated pricing use case I'd lean GLM; reserve boosting for where the extra
> accuracy is worth it."

## What drove risk (the business-communication payoff)
- Bonus-malus score, driver age (U-shaped: young + old riskier), area population density.
- All consistent with how the business already thinks about risk — a good sanity check.

## Honest limitations (name them BEFORE they do — most senior move you can make)
1. **Frequency only** — full pricing also needs claim severity (regression on amount).
2. **Exposure** — used as a feature here; production would use it as a Poisson offset.
3. **Modest AUC (~0.6s)** — realistic; claim frequency is intrinsically hard to predict.
4. **No fairness/regulatory review** — real insurance models face strict variable-permissibility scrutiny.

## If they push on technical depth
- **Imbalance handling:** `class_weight='balanced'` (GLM) and `scale_pos_weight` (XGBoost).
- **Preprocessing:** median/mode imputation, standardize numerics, one-hot encode categoricals — all in a scikit-learn `Pipeline` so there's no train/test leakage.
- **Validation:** stratified train/test split; would add cross-validation with more time.

## If asked "what would you do next?"
- Add a **severity** model (regression on claim amount) and combine into a pure-premium estimate.
- Switch to a **Poisson/Tweedie GLM** with exposure as an offset to model claim *counts* directly.
- Calibrate probabilities and tune the threshold around a real cost matrix.

## DON'T
- Don't quote a 99% accuracy number as a win — it's the rare-event trap.
- Don't lead with deep-learning jargon. This is tabular, classic modeling.
- Don't oversell — call it "a small project to learn the problem space."
