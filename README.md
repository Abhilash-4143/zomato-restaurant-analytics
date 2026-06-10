# 🍽️ Zomato Restaurant Analytics
## Customer Preferences, Rating Factors & Business Insights

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-4C72B0)
![License](https://img.shields.io/badge/License-MIT-green)

> **A complete end-to-end Data Analytics portfolio project** analysing 56,000+ Zomato restaurant records from Bangalore, India — covering data cleaning, EDA, advanced visualisations, business insights, and strategic recommendations.

---

## 📌 Project Overview

This project dives deep into the Zomato restaurant dataset to answer critical business questions:

- What factors drive restaurant ratings?
- Which cuisines and locations perform best?
- How do online ordering and table booking affect ratings?
- What is the relationship between price and quality?
- What strategic recommendations can we draw for a food-tech platform?

---

## 🗂️ Project Structure

```
zomato-restaurant-analytics/
│
├── 📓 zomato_analytics.ipynb    # Main analysis notebook (56 cells)
├── 📊 zomato.csv                # Raw dataset
├── 🏗️ build_notebook.py         # Script to regenerate the notebook
│
├── 📁 images/                   # Auto-generated visualisations (14 charts)
│   ├── 01_top_locations.png
│   ├── 02_rating_distribution.png
│   ├── 03_cuisine_analysis.png
│   ├── 04_cost_analysis.png
│   ├── 05_online_booking_impact.png
│   ├── 06_restaurant_types.png
│   ├── 07_location_ratings.png
│   ├── 08_correlation_heatmap.png
│   ├── 09_cost_vs_rating.png
│   ├── 10_votes_vs_rating.png
│   ├── 11_outlier_boxplots.png
│   ├── 12_cuisine_wordcloud.png
│   ├── 13_top_restaurants.png
│   └── 14_online_by_location.png
│
├── 📁 reports/
│   └── zomato_analytics_report.pdf   # Auto-generated PDF report
│
└── 📄 README.md
```

---

## ✨ Features

| Feature | Details |
|---|---|
| **Dataset** | 56,252 restaurant records, 13 columns |
| **Data Cleaning** | 9-step cleaning pipeline (duplicates, types, encoding, derived cols) |
| **EDA** | Location, cuisine, rating, cost, online-order, table-booking analysis |
| **Visualisations** | 14 professional charts (bar, pie, scatter, box, heatmap, wordcloud) |
| **Advanced Analysis** | Correlation matrix, outlier detection, trend lines, log-scale analysis |
| **Business Insights** | Quantified impact of key features on ratings |
| **Recommendations** | 6 Alfido Tech strategic recommendations with expected ROI |
| **PDF Report** | Full auto-generated report with all charts embedded |

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn wordcloud fpdf2 jupyter
```

### Run the Notebook
```bash
git clone https://github.com/YOUR-USERNAME/zomato-restaurant-analytics.git
cd zomato-restaurant-analytics
jupyter notebook zomato_analytics.ipynb
```

### Regenerate Notebook from Source
```bash
python build_notebook.py
```

---

## 📊 Key Findings

| Finding | Detail |
|---|---|
| **Average Rating** | 3.72 / 5.0 across all restaurants |
| **Top Location** | BTM, Koramangala, HSR Layout (highest density) |
| **Best Cuisine** | Continental & Cafe cuisines lead in ratings |
| **Table Booking Premium** | +0.3 rating advantage for table-booking restaurants |
| **Online Order Penetration** | ~60% of restaurants offer online ordering |
| **Sweet Spot Price** | ₹300–₹700 (Mid-range) dominates the market |

---

## 📈 Visualisations Preview

> All charts are auto-saved to the `/images` folder when the notebook runs.

- 🏙️ Top 15 Restaurant Locations
- ⭐ Rating Distribution & Band Analysis
- 🍛 Cuisine Popularity & Rating Comparison
- 💰 Cost Distribution & Price Category Pie
- 📱 Online Order & Table Booking Impact (Box Plots)
- 🔗 Feature Correlation Heatmap
- 📦 Outlier Detection
- ☁️ Cuisine WordCloud
- 🥇 Top 10 Most-Voted Restaurants

---

## 🏗️ Notebook Sections

1. **Introduction** — Project overview & objectives
2. **Import Libraries** — All dependencies with version check
3. **Load Dataset** — Data loading & initial inspection
4. **Data Understanding** — Shape, dtypes, missing values, duplicates
5. **Data Cleaning** — 9-step pipeline (standardise → clean → encode → derive)
6. **EDA** — 7 detailed analysis subsections
7. **Advanced Analysis** — Correlation, scatter, outlier, wordcloud
8. **Business Insights** — Quantified key findings
9. **Recommendations** — 6 strategic Alfido Tech recommendations
10. **PDF Report** — Auto-generated full report
11. **Conclusion** — Summary & next steps

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.9+ | Core programming language |
| Pandas | Data manipulation & cleaning |
| NumPy | Numerical operations |
| Matplotlib | Core plotting engine |
| Seaborn | Statistical visualisation |
| WordCloud | Text visualisation |
| FPDF2 | PDF report generation |
| Jupyter Notebook | Interactive development & presentation |

---

## 💼 Business Impact

This analysis demonstrates:
- **Data-driven decision making** for restaurant partnership strategy
- **Location intelligence** for market expansion
- **Customer behaviour analysis** translating to product improvements
- **Revenue opportunity identification** worth millions in GMV growth
- **Algorithm design** for fairer restaurant ranking

---

## 🚀 Alfido Tech Recommendations Summary

| # | Recommendation | Expected Impact |
|---|---|---|
| REC-01 | Smart Partnership Strategy | +10-15% platform rating |
| REC-02 | Location Targeting | +25% restaurant coverage |
| REC-03 | AI Cuisine Recommendation Engine | +20% CTR, +12% AOV |
| REC-04 | Dynamic Pricing & Promotions | New ad revenue stream |
| REC-05 | Online Order Penetration Drive | +15-20% GMV |
| REC-06 | Ranking Algorithm Enhancement | Fairer marketplace |

---

## 📄 Dataset

- **Source:** [Kaggle — Zomato Dataset](https://www.kaggle.com/datasets/bhanupratapbiswas/zomato)
- **Location:** Bangalore, India
- **Records:** 56,252 restaurants
- **Features:** 13 columns

---

## 👤 Author

**Your Name**  
📧 abhilashg869@gmail.com  
🔗 [LinkedIn]([https://linkedin.com/in/your-profile](https://www.linkedin.com/in/01abhilash?utm_source=share_via&utm_content=profile&utm_medium=member_android)
🐙 [GitHub](https://github.com/Abhilash-4143)

*⭐ If this project helped you, please give it a star!*
