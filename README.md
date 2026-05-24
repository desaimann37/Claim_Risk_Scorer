# Auto-Insurance Claim-Frequency Modeling

A small, honest, end-to-end data-science project on public auto-insurance data —
built to understand the kind of problem an insurance data science team solves day to day.

> **Business framing:** Insurers price a policy as `Pure Premium = Claim Frequency × Claim Severity`.
> This project models the **frequency** half — *how likely is a policy to file a claim?* — which sits
> at the heart of pricing and underwriting.

## What's here

| File | What it is |
|------|------------|
| `Auto_Insurance_Claim_Frequency.ipynb` | The main notebook: EDA → GLM + XGBoost → rare-event evaluation → feature importance → stakeholder summary. **Start here.** |
| `claim_pipeline.py` | Reusable pipeline module (data loading, prep, training, evaluation). Shared by the notebook and the app — clean separation of logic. |
| `app.py` | Streamlit app: an interactive policy risk scorer (the deployment layer). |
| `INTERVIEW_CHEATSHEET.md` | One-page talking points to glance at before the interview. |
| `requirements.txt` | Dependencies. |

## How it showcases all three things the role cares about
1. **Clean modeling + honest evaluation** — interpretable GLM baseline vs. gradient boosting, judged on ROC-AUC / PR-AUC / precision-recall (not misleading accuracy).
2. **Business communication** — feature-importance plots + a plain-English summary written for a non-technical pricing partner.
3. **End-to-end deployment** — a working Streamlit scorer on top of the trained model.

## Run it

```bash
pip install -r requirements.txt

# 1) Run the notebook (Jupyter or VS Code). It trains the models and saves claim_model.joblib
jupyter notebook Auto_Insurance_Claim_Frequency.ipynb

# 2) Launch the interactive scorer
streamlit run app.py
```

**Data:** the code loads the real `freMTPL2freq` dataset from OpenML automatically. If you're
offline or OpenML is down, it transparently falls back to a synthetic dataset with the same
schema so everything still runs. The notebook prints which source it used — be honest about
this if asked.

**Deploy for free (optional):** push to GitHub, then connect the repo at
[share.streamlit.io](https://share.streamlit.io) and point it at `app.py`. You'll get a public URL
you can drop into the conversation.

## Honest scope (state this before they ask)
- **Frequency only.** A full pricing model also needs claim *severity* (regression on claim amount).
- **Exposure** is used as a feature here; a production actuarial model would use it as a Poisson offset and model the claim *count* directly (Poisson/Tweedie GLM).
- **Modest AUC is expected.** Claim frequency is intrinsically hard to predict; the value is in the framing and rigor, not a headline number.
- Built as interview prep — **not** a production pricing tool.

## The 20-second pitch
> "I wanted to understand the problems your team works on, so I built a claim-frequency model on a
> public motor-insurance dataset — the frequency half of pricing a policy. The most interesting part
> was evaluation: since claims are rare, accuracy is misleading, so I focused on precision/recall and
> on which factors actually drive risk. I also wrapped it in a small Streamlit scorer to make it
> tangible."
