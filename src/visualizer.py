"""
Poll Results Visualization Suite
================================
Generates publication-grade static figures (Matplotlib / Seaborn) and interactive
figures (Plotly) for poll analysis reports and Streamlit dashboards.
Includes WordCloud generation and automated export to outputs/ and images/.
"""

import os
from typing import Optional, List, Dict, Any
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend suitable for headless scripts
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud

# Clean, professional styling configuration
PALETTE = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#06B6D4"]
BG_COLOR = "#FAFAFB"
CARD_BG = "#FFFFFF"
TEXT_COLOR = "#1F2937"
GRID_COLOR = "#E5E7EB"

plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "font.family": "sans-serif",
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": CARD_BG,
    "text.color": TEXT_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "grid.color": GRID_COLOR,
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
})


class PollVisualizer:
    """
    Visualization engine producing static and interactive graphics for poll data.
    """

    def __init__(self, df: pd.DataFrame, output_dir: str = "outputs", images_dir: str = "images"):
        self.df = df.copy()
        if "timestamp" in self.df.columns and not pd.api.types.is_datetime64_any_dtype(self.df["timestamp"]):
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"], errors="coerce")
        self.output_dir = output_dir
        self.images_dir = images_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

    def _save_figure(self, fig: plt.Figure, filename: str) -> str:
        """Helper to save a matplotlib figure to both output_dir and images_dir."""
        path1 = os.path.join(self.output_dir, filename)
        path2 = os.path.join(self.images_dir, filename)
        fig.savefig(path1, dpi=300, bbox_inches="tight")
        fig.savefig(path2, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path1

    # ==========================================
    # STATIC MATPLOTLIB / SEABORN VISUALIZATIONS
    # ==========================================

    def plot_vote_counts_bar(self) -> str:
        """Generate polished bar chart showing vote totals and percentages."""
        fig, ax = plt.subplots(figsize=(10, 6))

        counts = self.df["preferred_option"].value_counts()
        total = len(self.df)
        pcts = (counts / total * 100).round(1)

        y_positions = np.arange(len(counts))
        colors = PALETTE[:len(counts)]

        bars = ax.barh(y_positions, counts.values, color=colors, height=0.6, edgecolor="none", alpha=0.9)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(counts.index, fontsize=11, fontweight="normal")
        ax.invert_yaxis()  # Top-down order

        # Annotate bars with vote count and percentage
        for bar, count, pct in zip(bars, counts.values, pcts.values):
            width = bar.get_width()
            ax.text(
                width + (total * 0.012),
                bar.get_y() + bar.get_height() / 2,
                f"{count:,} ({pct}%)",
                va="center",
                ha="left",
                fontsize=11,
                fontweight="bold",
                color="#1E3A8A"
            )

        ax.set_xlabel("Number of Responses", fontsize=11, fontweight="normal", labelpad=8)
        ax.set_title("Primary Work Model Preference (Vote Counts & Share)", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlim(0, max(counts.values) * 1.18)
        ax.grid(axis="x", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        return self._save_figure(fig, "01_vote_distribution_bar.png")

    def plot_preference_donut(self) -> str:
        """Generate sleek donut chart illustrating market/preference share."""
        fig, ax = plt.subplots(figsize=(8, 8))

        counts = self.df["preferred_option"].value_counts()
        colors = PALETTE[:len(counts)]

        wedges, texts, autotexts = ax.pie(
            counts.values,
            labels=counts.index,
            autopct="%1.1f%%",
            pctdistance=0.75,
            startangle=140,
            colors=colors,
            wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2.5),
        )

        for text in texts:
            text.set_fontsize(11)
            text.set_fontweight("normal")
        for autotext in autotexts:
            autotext.set_fontsize(11)
            autotext.set_fontweight("bold")
            autotext.set_color("white")

        # Center summary badge
        ax.text(0, 0, f"Total\n{len(self.df):,}\nVotes", ha="center", va="center", fontsize=14, fontweight="bold", color="#374151")
        ax.set_title("Poll Preference Distribution (Donut Chart)", fontsize=14, fontweight="bold", pad=20)

        return self._save_figure(fig, "02_preference_share_donut.png")

    def plot_satisfaction_boxplots(self) -> str:
        """Generate box plot showing satisfaction ratings across options with mean markers."""
        fig, ax = plt.subplots(figsize=(10, 6))

        order = self.df.groupby("preferred_option")["satisfaction_rating"].mean().sort_values(ascending=False).index

        sns.boxplot(
            data=self.df,
            x="preferred_option",
            y="satisfaction_rating",
            hue="preferred_option",
            legend=False,
            order=order,
            palette=PALETTE[:len(order)],
            showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 7},
            boxprops=dict(alpha=0.85),
            ax=ax
        )

        ax.set_title("Satisfaction Rating Distribution by Work Model (1 = Low, 5 = High)", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Preferred Work Model", fontsize=11, fontweight="bold", labelpad=8)
        ax.set_ylabel("Satisfaction Rating", fontsize=11, fontweight="bold", labelpad=8)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels(["1 (Very Dissatisfied)", "2 (Dissatisfied)", "3 (Neutral)", "4 (Satisfied)", "5 (Very Satisfied)"])
        ax.grid(axis="y", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        return self._save_figure(fig, "03_satisfaction_distribution.png")

    def plot_demographic_heatmap(self) -> str:
        """Generate annotated heatmap of Age Group vs Preferred Option percentages."""
        if "age_group" not in self.df.columns:
            return ""

        fig, ax = plt.subplots(figsize=(10, 6))
        ct = pd.crosstab(self.df["age_group"], self.df["preferred_option"], normalize="index") * 100

        # Natural age ordering if present
        desired_order = ["18-24", "25-34", "35-44", "45-54", "55+", "Unspecified"]
        current_rows = [r for r in desired_order if r in ct.index]
        ct = ct.reindex(current_rows)

        sns.heatmap(
            ct,
            annot=True,
            fmt=".1f",
            cmap="YlGnBu",
            cbar_kws={"label": "Percentage within Age Cohort (%)"},
            linewidths=1.5,
            linecolor="white",
            ax=ax,
        )

        ax.set_title("Demographic Preference Breakdown: Age Group vs Work Model", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Preferred Work Model", fontsize=11, fontweight="normal", labelpad=8)
        ax.set_ylabel("Age Cohort", fontsize=11, fontweight="normal", labelpad=8)

        return self._save_figure(fig, "04_demographic_heatmap.png")

    def plot_time_trend(self) -> str:
        """Generate response trend line chart over the survey timeline."""
        if "timestamp" not in self.df.columns or self.df["timestamp"].isnull().all():
            return ""

        temp = self.df.dropna(subset=["timestamp"]).copy()
        temp["date"] = temp["timestamp"].dt.date
        daily = temp.groupby("date").size().reset_index(name="count")
        daily = daily.sort_values("date").reset_index(drop=True)
        daily["cumulative"] = daily["count"].cumsum()

        fig, ax1 = plt.subplots(figsize=(11, 5.5))

        # Bar for daily count
        color1 = "#3B82F6"
        bars = ax1.bar(range(len(daily)), daily["count"], color=color1, alpha=0.45, label="Daily Responses")
        ax1.set_xlabel("Survey Timeline (Days)", fontsize=11, fontweight="normal", labelpad=8)
        ax1.set_ylabel("Daily Responses", color=color1, fontsize=11, fontweight="bold", labelpad=8)
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.grid(axis="x", alpha=0.3)

        # X-tick formatting (sample points)
        step = max(1, len(daily) // 7)
        xticks = list(range(0, len(daily), step))
        ax1.set_xticks(xticks)
        ax1.set_xticklabels([str(daily["date"].iloc[i]) for i in xticks], rotation=25, ha="right")

        # Line on secondary axis for cumulative
        ax2 = ax1.twinx()
        color2 = "#10B981"
        ax2.plot(range(len(daily)), daily["cumulative"], color=color2, linewidth=3, marker="o", markersize=4, label="Cumulative Responses")
        ax2.set_ylabel("Cumulative Total Responses", color=color2, fontsize=11, fontweight="bold", labelpad=8)
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.grid(False)

        ax1.set_title("Response Velocity & Cumulative Growth Curve", fontsize=14, fontweight="bold", pad=15)
        ax1.spines["top"].set_visible(False)
        ax2.spines["top"].set_visible(False)

        return self._save_figure(fig, "05_time_trend_trajectory.png")

    def plot_wordcloud(self) -> str:
        """Generate word cloud from respondent qualitative feedback comments."""
        if "feedback" not in self.df.columns:
            return ""

        text = " ".join(self.df["feedback"].dropna().astype(str).tolist())
        if not text.strip():
            return ""

        wc = WordCloud(
            width=1000,
            height=500,
            background_color="#FFFFFF",
            colormap="crest",
            max_words=80,
            random_state=42,
            contour_width=1,
            contour_color="#E5E7EB",
        ).generate(text)

        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title("Qualitative Feedback Keywords Word Cloud", fontsize=14, fontweight="bold", pad=15)

        return self._save_figure(fig, "06_feedback_wordcloud.png")

    def render_all_static_plots(self) -> Dict[str, str]:
        """Generate and save all static figures."""
        paths = {
            "bar": self.plot_vote_counts_bar(),
            "donut": self.plot_preference_donut(),
            "satisfaction": self.plot_satisfaction_boxplots(),
            "heatmap": self.plot_demographic_heatmap(),
            "trend": self.plot_time_trend(),
            "wordcloud": self.plot_wordcloud(),
        }
        print("[Poll Visualizer] All 6 static figures successfully rendered and exported to outputs/ and images/.")
        return paths

    # ==========================================
    # INTERACTIVE PLOTLY FIGURES FOR STREAMLIT
    # ==========================================

    def create_interactive_bar(self) -> go.Figure:
        """Interactive Plotly bar chart."""
        counts = self.df["preferred_option"].value_counts().reset_index()
        counts.columns = ["Preferred Option", "Votes"]
        counts["Percentage"] = (counts["Votes"] / len(self.df) * 100).round(2)

        fig = px.bar(
            counts,
            x="Votes",
            y="Preferred Option",
            orientation="h",
            text="Percentage",
            color="Preferred Option",
            color_discrete_sequence=PALETTE,
            title="Interactive Preference Distribution",
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=40, t=40, b=20),
            xaxis=dict(showgrid=True, gridcolor="#E5E7EB"),
        )
        return fig

    def create_interactive_pie(self) -> go.Figure:
        """Interactive Plotly donut chart."""
        counts = self.df["preferred_option"].value_counts().reset_index()
        counts.columns = ["Preferred Option", "Votes"]
        fig = px.pie(
            counts,
            names="Preferred Option",
            values="Votes",
            hole=0.45,
            color_discrete_sequence=PALETTE,
            title="Interactive Preference Share",
        )
        fig.update_traces(textinfo="percent+label", hoverinfo="label+value+percent")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=40, b=20),
        )
        return fig

    def create_interactive_satisfaction(self) -> go.Figure:
        """Interactive Plotly box plot for satisfaction ratings."""
        fig = px.box(
            self.df,
            x="preferred_option",
            y="satisfaction_rating",
            color="preferred_option",
            color_discrete_sequence=PALETTE,
            points="all",
            title="Satisfaction Ratings Spread by Preference",
            labels={"preferred_option": "Work Model", "satisfaction_rating": "Satisfaction Rating (1-5)"},
        )
        fig.update_layout(
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=40, b=20),
            yaxis=dict(tickmode="linear", tick0=1, dtick=1, range=[0.5, 5.5], gridcolor="#E5E7EB"),
        )
        return fig

    def create_interactive_trend(self) -> go.Figure:
        """Interactive Plotly line and bar chart for timeline."""
        if "timestamp" not in self.df.columns or self.df["timestamp"].isnull().all():
            return go.Figure()

        temp = self.df.dropna(subset=["timestamp"]).copy()
        temp["date"] = temp["timestamp"].dt.date
        daily = temp.groupby("date").size().reset_index(name="Daily Votes")
        daily = daily.sort_values("date").reset_index(drop=True)
        daily["Cumulative Votes"] = daily["Daily Votes"].cumsum()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=daily["date"],
            y=daily["Daily Votes"],
            name="Daily Votes",
            marker_color="#93C5FD",
            opacity=0.8,
        ))
        fig.add_trace(go.Scatter(
            x=daily["date"],
            y=daily["Cumulative Votes"],
            name="Cumulative Growth",
            mode="lines+markers",
            line=dict(color="#2563EB", width=3),
            yaxis="y2",
        ))

        fig.update_layout(
            title="Daily & Cumulative Response Trajectory",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title="Daily Votes", showgrid=True, gridcolor="#E5E7EB"),
            yaxis2=dict(title="Cumulative Votes", overlaying="y", side="right"),
            legend=dict(x=0.01, y=0.99),
            margin=dict(l=20, r=40, t=40, b=20),
        )
        return fig


if __name__ == "__main__":
    cleaned_file = os.path.join("data", "cleaned_poll_results.csv")
    print(f"Loading cleaned data for visualization from: {cleaned_file}")
    df_clean = pd.read_csv(cleaned_file)

    viz = PollVisualizer(df_clean)
    generated = viz.render_all_static_plots()
    for name, path in generated.items():
        print(f"  * {name}: {path}")
