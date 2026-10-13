# Dataset

Place the reviews file here as **`data/reviews.csv`** (or upload it in the dashboard sidebar).

Expected file: `m-government_apps_Arabic_reviews_.csv` (6,513 Arabic reviews of Saudi government mobile apps).
Columns used by the prototype:

| Column | Used as |
|---|---|
| `Content` | original review text (main input to the agents) |
| `Abbreviation` | application identifier (`app_id`) |
| `Source` | application name (`app_name`) |
| `Rating` | star rating (1-5) |

The other columns (`Cleaned Content`, `Stemmed Content`, `Language`, `Segmented Content`) are not used. The agents
work on the original text so that evidence quotes are verbatim.

The dataset contains **no** ground-truth labels for severity, service-quality dimensions or escalation.

`*.csv` files in this folder are git-ignored so the research data is not committed by accident.
