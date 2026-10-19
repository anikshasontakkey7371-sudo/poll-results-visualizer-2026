"""
Poll Statistical Analysis and Insights Engine
=============================================
Calculates vote distributions, demographic cross-tabulations, satisfaction ratings,
time series patterns, feedback word frequencies, and generates narrative analytical insights.
"""

import os
import re
import json
from collections import Counter
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


# Common English stop words for feedback keyword analysis
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves", "comment", "provided", "comments"
}


class PollAnalyzer:
    """
    Computes statistical analyses and generates narrative insights for poll datasets.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        if "timestamp" in self.df.columns and not pd.api.types.is_datetime64_any_dtype(self.df["timestamp"]):
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"], errors="coerce")

    def compute_summary_kpis(self) -> Dict[str, Any]:
        """Compute top-level Key Performance Indicators for the survey."""
        total_respondents = len(self.df)
        if total_respondents == 0:
            return {}

        # Option preferences
        option_counts = self.df["preferred_option"].value_counts()
        leading_option = option_counts.index[0] if len(option_counts) > 0 else "N/A"
        leading_votes = int(option_counts.iloc[0]) if len(option_counts) > 0 else 0
        leading_pct = round((leading_votes / total_respondents) * 100, 2)

        runner_up = option_counts.index[1] if len(option_counts) > 1 else "N/A"
        runner_up_votes = int(option_counts.iloc[1]) if len(option_counts) > 1 else 0
        runner_up_pct = round((runner_up_votes / total_respondents) * 100, 2)

        # Satisfaction metrics
        sat_series = pd.to_numeric(self.df["satisfaction_rating"], errors="coerce").dropna()
        mean_sat = round(float(sat_series.mean()), 2) if len(sat_series) > 0 else 0.0
        median_sat = round(float(sat_series.median()), 2) if len(sat_series) > 0 else 0.0
        sat_std = round(float(sat_series.std()), 2) if len(sat_series) > 0 else 0.0

        # Net Satisfaction Score (% High [4,5] - % Low [1,2])
        high_sat_count = int((sat_series >= 4).sum())
        low_sat_count = int((sat_series <= 2).sum())
        pct_high_sat = round((high_sat_count / total_respondents) * 100, 2)
        pct_low_sat = round((low_sat_count / total_respondents) * 100, 2)
        net_satisfaction_score = round(pct_high_sat - pct_low_sat, 2)

        # Geographic reach
        unique_regions = int(self.df["region"].nunique()) if "region" in self.df.columns else 0
        top_region = self.df["region"].value_counts().index[0] if unique_regions > 0 else "N/A"

        return {
            "total_respondents": total_respondents,
            "leading_option": leading_option,
            "leading_votes": leading_votes,
            "leading_percentage": leading_pct,
            "runner_up_option": runner_up,
            "runner_up_votes": runner_up_votes,
            "runner_up_percentage": runner_up_pct,
            "mean_satisfaction": mean_sat,
            "median_satisfaction": median_sat,
            "satisfaction_std": sat_std,
            "pct_high_satisfaction": pct_high_sat,
            "pct_low_satisfaction": pct_low_sat,
            "net_satisfaction_score": net_satisfaction_score,
            "unique_regions_represented": unique_regions,
            "top_participating_region": top_region,
        }

    def compute_option_breakdown(self) -> pd.DataFrame:
        """Calculate vote counts, percentages, and satisfaction stats per option."""
        total = len(self.df)
        grouped = self.df.groupby("preferred_option").agg(
            votes=("respondent_id", "count"),
            mean_satisfaction=("satisfaction_rating", "mean"),
            median_satisfaction=("satisfaction_rating", "median"),
            std_satisfaction=("satisfaction_rating", "std"),
        ).reset_index()

        grouped["percentage"] = (grouped["votes"] / total * 100).round(2)
        grouped["mean_satisfaction"] = grouped["mean_satisfaction"].round(2)
        grouped["std_satisfaction"] = grouped["std_satisfaction"].fillna(0).round(2)
        grouped = grouped.sort_values(by="votes", ascending=False).reset_index(drop=True)
        return grouped

    def compute_cross_tabulation(
        self, index_col: str, normalize: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Cross-tabulate preferred_option against another categorical column.

        Parameters
        ----------
        index_col : str
            Demographic dimension (e.g. 'region', 'age_group', 'ai_adoption_level')
        normalize : Optional[str]
            'index' for row percentages, 'columns' for column percentages, or None for raw counts.
        """
        if index_col not in self.df.columns or "preferred_option" not in self.df.columns:
            return pd.DataFrame()

        ct = pd.crosstab(
            self.df[index_col],
            self.df["preferred_option"],
            normalize=normalize,
        )
        if normalize:
            ct = (ct * 100).round(2)
        return ct

    def compute_satisfaction_by_dimension(self, dim_col: str) -> pd.DataFrame:
        """Compute satisfaction statistics grouped by a categorical dimension."""
        if dim_col not in self.df.columns:
            return pd.DataFrame()

        res = self.df.groupby(dim_col)["satisfaction_rating"].agg(
            respondents="count",
            mean_satisfaction="mean",
            median_satisfaction="median",
            std_satisfaction="std",
        ).reset_index()

        res["mean_satisfaction"] = res["mean_satisfaction"].round(2)
        res["std_satisfaction"] = res["std_satisfaction"].fillna(0).round(2)
        return res.sort_values(by="mean_satisfaction", ascending=False).reset_index(drop=True)

    def compute_time_trends(self) -> pd.DataFrame:
        """Aggregate daily response counts and cumulative voting curve."""
        if "timestamp" not in self.df.columns or self.df["timestamp"].isnull().all():
            return pd.DataFrame()

        temp = self.df.dropna(subset=["timestamp"]).copy()
        temp["date"] = temp["timestamp"].dt.date
        daily = temp.groupby("date").agg(daily_votes=("respondent_id", "count")).reset_index()
        daily = daily.sort_values("date").reset_index(drop=True)
        daily["cumulative_votes"] = daily["daily_votes"].cumsum()
        daily["cumulative_percentage"] = (daily["cumulative_votes"] / len(self.df) * 100).round(2)
        return daily

    def compute_feedback_keywords(self, top_n: int = 25) -> List[Dict[str, Any]]:
        """Extract frequent keywords from the feedback text column."""
        if "feedback" not in self.df.columns:
            return []

        all_text = " ".join(self.df["feedback"].dropna().astype(str).tolist()).lower()
        # Tokenize alphanumeric words with length >= 4
        words = re.findall(r"\b[a-z]{4,}\b", all_text)
        filtered = [w for w in words if w not in STOP_WORDS]
        counts = Counter(filtered).most_common(top_n)

        return [{"word": word, "count": count} for word, count in counts]

    def generate_narrative_insights(self) -> List[str]:
        """
        Synthesize algorithmic, dynamic insights from the computed statistics.
        Avoids hard-coding results so any uploaded poll data receives custom analysis.
        """
        kpis = self.compute_summary_kpis()
        options = self.compute_option_breakdown()
        insights = []

        if not kpis or options.empty:
            return ["Insufficient data to produce narrative insights."]

        # 1. Primary Winner insight
        lead = kpis["leading_option"]
        lead_pct = kpis["leading_percentage"]
        lead_votes = kpis["leading_votes"]
        runner = kpis["runner_up_option"]
        runner_pct = kpis["runner_up_percentage"]
        margin = round(lead_pct - runner_pct, 2)

        insights.append(
            f"**Plurality Winner**: **'{lead}'** emerged as the primary choice with **{lead_pct}%** "
            f"({lead_votes:,} votes), leading runner-up **'{runner}'** ({runner_pct}%) by a margin of {margin} percentage points."
        )

        # 2. Satisfaction Contrast insight
        opt_sat = options.set_index("preferred_option")["mean_satisfaction"].to_dict()
        if len(opt_sat) >= 2:
            highest_sat_opt = max(opt_sat, key=opt_sat.get)
            lowest_sat_opt = min(opt_sat, key=opt_sat.get)
            insights.append(
                f"**Satisfaction Disparity**: Respondents favoring **'{highest_sat_opt}'** reported the highest satisfaction "
                f"(mean **{opt_sat[highest_sat_opt]:.2f} / 5.0**), whereas **'{lowest_sat_opt}'** recorded the lowest satisfaction "
                f"(mean **{opt_sat[lowest_sat_opt]:.2f} / 5.0**)."
            )

        # 3. Net Satisfaction Sentiment
        net_score = kpis["net_satisfaction_score"]
        pct_high = kpis["pct_high_satisfaction"]
        pct_low = kpis["pct_low_satisfaction"]
        sentiment_label = "overwhelmingly positive" if net_score > 40 else "moderately positive" if net_score > 0 else "predominantly critical"
        insights.append(
            f"**Overall Sentiment Health**: Net Satisfaction stands at **+{net_score}%** ({sentiment_label}), "
            f"with **{pct_high}%** of respondents rating their experience a 4 or 5 compared to only **{pct_low}%** rating 1 or 2."
        )

        # 4. Demographic Variance (Age / Region)
        if "age_group" in self.df.columns and "preferred_option" in self.df.columns:
            age_ct = self.compute_cross_tabulation("age_group", normalize="index")
            if not age_ct.empty and lead in age_ct.columns:
                top_age_for_lead = age_ct[lead].idxmax()
                top_age_lead_pct = age_ct.loc[top_age_for_lead, lead]
                insights.append(
                    f"**Generational Alignment**: Support for **'{lead}'** peaked among the **{top_age_for_lead}** demographic cohort "
                    f"at **{top_age_lead_pct:.1f}%**, demonstrating strong generational affinity."
                )

        if "region" in self.df.columns:
            region_sat = self.compute_satisfaction_by_dimension("region")
            if not region_sat.empty:
                top_reg = region_sat.iloc[0]["region"]
                top_reg_score = region_sat.iloc[0]["mean_satisfaction"]
                insights.append(
                    f"**Geographic Hotspot**: **{top_reg}** registered the highest regional satisfaction "
                    f"rating at **{top_reg_score:.2f} / 5.0**."
                )

        return insights

    def export_full_report(self, output_dir: str = "outputs") -> Dict[str, str]:
        """Export comprehensive analysis summary as both JSON and Markdown reports."""
        os.makedirs(output_dir, exist_ok=True)
        kpis = self.compute_summary_kpis()
        options = self.compute_option_breakdown().to_dict(orient="records")
        insights = self.generate_narrative_insights()
        keywords = self.compute_feedback_keywords(top_n=20)

        # 1. JSON Report
        report_data = {
            "kpi_summary": kpis,
            "option_breakdown": options,
            "narrative_insights": insights,
            "top_feedback_keywords": keywords,
        }
        json_path = os.path.join(output_dir, "summary_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=4)

        # 2. Markdown Report
        md_lines = [
            "# Poll Results Visualizer: Statistical Analysis Report",
            "",
            "## 1. Executive Summary & KPIs",
            f"- **Total Valid Respondents**: `{kpis.get('total_respondents', 0):,}`",
            f"- **Plurality Preference**: **{kpis.get('leading_option', 'N/A')}** ({kpis.get('leading_percentage', 0)}% | {kpis.get('leading_votes', 0):,} votes)",
            f"- **Runner-Up Preference**: **{kpis.get('runner_up_option', 'N/A')}** ({kpis.get('runner_up_percentage', 0)}% | {kpis.get('runner_up_votes', 0):,} votes)",
            f"- **Average Satisfaction Rating**: `{kpis.get('mean_satisfaction', 0):.2f} / 5.0` (Std Dev: `{kpis.get('satisfaction_std', 0):.2f}`)",
            f"- **Net Satisfaction Score**: `+{kpis.get('net_satisfaction_score', 0)}%`",
            f"- **Top Participating Region**: `{kpis.get('top_participating_region', 'N/A')}`",
            "",
            "## 2. Option Vote Breakdown",
            "| Preferred Option | Votes | Vote % | Mean Satisfaction | Std Dev |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
        for opt in options:
            md_lines.append(
                f"| **{opt['preferred_option']}** | {opt['votes']:,} | {opt['percentage']}% | "
                f"{opt['mean_satisfaction']:.2f} / 5.0 | {opt['std_satisfaction']:.2f} |"
            )

        md_lines.extend([
            "",
            "## 3. Automated Narrative Insights",
        ])
        for idx, insight in enumerate(insights, start=1):
            md_lines.append(f"{idx}. {insight}")

        md_lines.extend([
            "",
            "## 4. Top Qualitative Feedback Keywords",
            "| Keyword | Frequency Mentioned |",
            "| :--- | :--- |",
        ])
        for kw in keywords[:10]:
            md_lines.append(f"| `{kw['word']}` | {kw['count']} |")

        md_path = os.path.join(output_dir, "summary_report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        print(f"[Poll Analyzer] Successfully saved JSON report: {json_path}")
        print(f"[Poll Analyzer] Successfully saved Markdown report: {md_path}")
        return {"json": json_path, "markdown": md_path}


if __name__ == "__main__":
    cleaned_file = os.path.join("data", "cleaned_poll_results.csv")
    print(f"Loading cleaned data from: {cleaned_file}")
    df_clean = pd.read_csv(cleaned_file)

    analyzer = PollAnalyzer(df_clean)
    kpis = analyzer.compute_summary_kpis()
    print("\nKPI Summary:")
    for k, v in kpis.items():
        print(f"  {k}: {v}")

    print("\nNarrative Insights:")
    for ins in analyzer.generate_narrative_insights():
        print(f"  * {ins}")

    analyzer.export_full_report()
