# Dataset

Place the reviews file here as **`data/reviews.csv`** (or upload it in the dashboard sidebar).

Research dataset: `balanced_1-5_Arabic_dataset.csv` — 5,434 Arabic reviews of Saudi government mobile apps, balanced
between 1-star (2,710) and 5-star (2,724) ratings, 87 application identifiers (`Abbreviation`; 93 distinct `Source`
names). After the loader removes 557 duplicate texts within the same app, 4,877 reviews remain.
The earlier full file `m-government_apps_Arabic_reviews_.csv` (6,513 reviews, ratings 1-5) also loads.
Columns used by the prototype:

| Column | Used as |
|---|---|
| `Content` | original review text (main input to the agents) |
| `Abbreviation` | application identifier (`app_id`) |
| `Source` | application name (`app_name`) |
| `Rating` | star rating (1-5) |

The other columns (cleaned, stemmed, language and segmented text; unnamed in the balanced file) are not used. The agents
work on the original text so that evidence quotes are verbatim.

The dataset contains **no** ground-truth labels for severity, service-quality dimensions or escalation.

`*.csv` files in this folder are git-ignored so the research data is not committed by accident.
