# Customer Financial Health & Engagement Scoring | Python, LightGBM, WoE Scorecard

Round 2 of the **ITB Competition (Business Intelligence Season 10, UEL)**: a 30-hour hackathon on a consumer-finance lender's data for **1,000,528 customers**. The lender answered two questions in separate teams: *can this customer repay?* and *will this customer borrow again?* We built one explainable 0–100 score for each question, combined them into segments, and gave every customer one next best action with a plain-language reason.

> Competition project by team *Đường sắt Bắc – Nam*. The team went on to finish **runner-up of 200 teams** after the final round ([ITB-Shop-Growth-Potential](https://github.com/AIVIETNAM-AIO-thuhuynhtran05/ITB-Shop-Growth-Potential)). The dataset belongs to the organisers and is not included.

## The brief

| Challenge | Asked for |
|---|---|
| 1. Financial Health Score | Factors behind financial health, a 0–100 score, customer groups, positive and negative drivers, evidence the method is sound |
| 2. Engagement Score | Recency, frequency, interaction, lead and product behaviour turned into a score and segments, with drivers and proof the score tracks business outcomes |
| 3. Segmentation & Next Best Action | Combine both scores, find the opportunities, recommend one action per customer and explain why |

Rules from the brief: scores must be explainable, correlation is not causation, and demographics must not decide risk without evidence.

## Results

| | Financial Health | Engagement |
|---|---|---|
| Method | LightGBM with monotonic constraints + TreeSHAP | WoE scorecard, isotonic fit per variable |
| Inputs | 35 of 55 behavioural indicators | 6 of 23 behavioural candidates |
| Validation | Out-of-time on Dec 2024 (never used in training) | 500K held-out customers + re-test two months later |
| AUC | **0.674** (WoE scorecard baseline 0.661) | **0.705** (RFM 0.660, gradient boosting 0.730) |
| Ranking | Late rate rises from 1% in band A to 6% in band E | Top decile converts at 25.9% vs 2.3% in the bottom |

## Key findings

1. **Credit behaviour decides financial health, not income.** A late payment in the last 12 months raises risk 11.9×; lender trust (58%) and financial buffer (15%) drive most of the score, while income and debt together add about 10%.
2. **Lateness repeats and fades slowly.** One earlier late contract lifts the next contract's late rate from 1.4% to 10.2%; a late payment within 6 months means 36.9%, still 4.3% after 4 years.
3. **Engagement means signing and applying.** Loans signed and applications carry 69% of the score. Call answer rate gets no weight because it reflects who sales chose to call, not customer intent.
4. **Value is concentrated and timing matters.** Champions and Active customers are 43.8% of the base but 79.5% of new booked value; customers whose loan has just ended are 1.64× more likely to take a new one.
5. **The two scores add different information** (Spearman ρ = 0.49): 35% of the healthiest customers are barely engaged, and 43% of the riskiest are highly engaged.

## Segments and next best actions

| Segment | Share | Next best action |
|---|---|---|
| Stars (high health, high engagement) | 27% | Upsell and cross-sell; renewal offer when a loan ends |
| Sleeping Gems (high health, low engagement) | 16% | Pre-approved offer through app push or SMS |
| Growth (mid health, high engagement) | 10% | Right-sized offer: small limit, down payment |
| Cool Prospects (mid health, low engagement) | 13% | Low-cost nurture, no credit push |
| Stressed Seekers (low health, high engagement) | 7% | No new credit; counselling and repayment reminders |
| At-Risk Dormant (low health, low engagement) | 9% | Monitor only, stop marketing |
| Insufficient History | 19% | Starter limit that grows with each on-time instalment |

Risk is blocked first, then the segment action applies. We recommend an A/B test with a 10% control group (safety metric: delinquency rate) before rollout, and quarterly recalibration of the engagement cut-offs.

## Repository structure

```
├── challenge1_financial_health/
│   ├── Challenge1_Financial_Health.ipynb   # EDA, features, model, bands, SHAP, validation
│   └── code/                               # pipeline modules used by the notebook
│       ├── load_team_clean.py              # cleaned Excel tables -> parquet
│       ├── features.py, feature_doc.py     # snapshot features before each cut-off
│       ├── health_pipeline.py              # training, scoring, monitoring
│       ├── scorecard.py                    # WoE scorecard baseline
│       └── eda_cells.py, eda_reasoning.py, build_pipeline_notebook.py
├── challenge2_engagement/
│   ├── Challenge2_Customer_Engagement.ipynb
│   ├── eng_core.py, build_nb.py
│   └── output/                             # IV selection, scorecard, validation tables, figures
├── challenge3_segmentation_nba/
│   ├── Challenge3_Segmentation_NBA.ipynb
│   ├── build_nb.py
│   └── output/                             # segment summary, opportunity, A/B plan, figures
└── slides/ITB_Round2_Health_Engagement_Slides.pdf
```

Notebook comments are in Vietnamese; the slides are in English.

## How to run

1. Place the competition tables (CUSTOMER, CONTRACT, LEADS, PAST_BEHAVIOR, CALL) locally; they are not included.
2. Convert the cleaned tables to parquet: `python challenge1_financial_health/code/load_team_clean.py "<folder with *_clean.xlsx>"`.
3. Run the three notebooks in order. Challenge 3 reads the customer-level scores written by Challenges 1 and 2. Update the data path at the top of each notebook if needed.

## Tools

Python (pandas, DuckDB, PyArrow), LightGBM, SHAP, scikit-learn, matplotlib.
