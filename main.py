"""
Poll Results Visualizer - Command-Line Pipeline Runner
======================================================
End-to-end pipeline runner for:
1. Synthetic survey data generation (or loading user CSV)
2. Data cleaning, validation, and deduplication
3. Statistical analysis and automated narrative insights
4. Publication-quality figure generation (saved to outputs/ & images/)
5. Summary report generation (Markdown & JSON)

Usage:
------
    python main.py
    python main.py --samples 2000 --seed 123
    python main.py --input-file data/my_custom_survey.csv
"""

import os
import sys
import argparse
from typing import Optional

from src.generator import generate_poll_data, save_dataset
from src.cleaner import DataCleaner
from src.analyzer import PollAnalyzer
from src.visualizer import PollVisualizer


def run_pipeline(
    input_file: Optional[str] = None,
    samples: int = 1500,
    seed: int = 42,
    raw_output_path: str = os.path.join("data", "raw_poll_responses.csv"),
    clean_output_path: str = os.path.join("data", "cleaned_poll_results.csv"),
    output_dir: str = "outputs",
    images_dir: str = "images",
) -> None:
    """
    Executes the full poll data processing and visualization pipeline.
    """
    print("=" * 75)
    print("        POLL RESULTS VISUALIZER - COMPREHENSIVE PIPELINE")
    print("=" * 75)

    # ---------------------------------------------------------
    # Step 1: Data Acquisition / Generation
    # ---------------------------------------------------------
    if input_file and os.path.exists(input_file):
        print(f"\n[Phase 1/4] Loading provided survey dataset from: {input_file}")
        source_file = input_file
    else:
        print(f"\n[Phase 1/4] Generating reproducible synthetic dataset ({samples:,} respondents, seed={seed})...")
        raw_df = generate_poll_data(n_samples=samples, random_seed=seed, inject_noise=True)
        save_dataset(raw_df, raw_output_path)
        source_file = raw_output_path
        print(f" -> Raw dataset saved: {raw_output_path} ({len(raw_df):,} rows with intentional anomalies)")

    # ---------------------------------------------------------
    # Step 2: Data Cleaning & Validation
    # ---------------------------------------------------------
    print("\n[Phase 2/4] Executing Data Cleaning & Validation Pipeline...")
    cleaner = DataCleaner.from_csv(source_file)
    cleaned_df, audit = cleaner.clean()
    cleaner.save_cleaned(clean_output_path)

    print(f" -> Deduplication: Removed {audit['duplicates_removed']} duplicate submissions.")
    print(f" -> Bounds Validation: Corrected {audit['out_of_bounds_ratings_corrected']} invalid ratings.")
    print(f" -> Imputation: Imputed {audit['missing_ratings_imputed']} missing satisfaction ratings.")
    print(f" -> Clean Verified Records: {audit['final_rows']:,}")

    # ---------------------------------------------------------
    # Step 3: Statistical Analysis & Narrative Insights
    # ---------------------------------------------------------
    print("\n[Phase 3/4] Performing Statistical Analysis & Extracting Insights...")
    analyzer = PollAnalyzer(cleaned_df)
    kpis = analyzer.compute_summary_kpis()
    insights = analyzer.generate_narrative_insights()
    report_paths = analyzer.export_full_report(output_dir=output_dir)

    print(f" -> Leading Choice: {kpis['leading_option']} ({kpis['leading_percentage']}% | {kpis['leading_votes']:,} votes)")
    print(f" -> Runner-Up:      {kpis['runner_up_option']} ({kpis['runner_up_percentage']}% | {kpis['runner_up_votes']:,} votes)")
    print(f" -> Mean Satisfaction: {kpis['mean_satisfaction']:.2f} / 5.0 (Std Dev: {kpis['satisfaction_std']:.2f})")
    print(f" -> Net Satisfaction Score: +{kpis['net_satisfaction_score']}%")
    print(f" -> Analysis Reports exported: {report_paths['markdown']} & {report_paths['json']}")

    # ---------------------------------------------------------
    # Step 4: Visualizations
    # ---------------------------------------------------------
    print("\n[Phase 4/4] Rendering High-Resolution Charts (Matplotlib, Seaborn, WordCloud)...")
    visualizer = PollVisualizer(cleaned_df, output_dir=output_dir, images_dir=images_dir)
    rendered_plots = visualizer.render_all_static_plots()

    for name, path in rendered_plots.items():
        if path:
            print(f" -> [{name.upper()}] {path}")

    # ---------------------------------------------------------
    # Final Pipeline Summary
    # ---------------------------------------------------------
    print("\n" + "=" * 75)
    print("                    AUTOMATED NARRATIVE INSIGHTS")
    print("=" * 75)
    for idx, item in enumerate(insights, start=1):
        clean_item = item.replace("**", "")
        print(f"  {idx}. {clean_item}")

    print("\n" + "=" * 75)
    print("                  PIPELINE EXECUTION COMPLETED")
    print("=" * 75)
    print("To launch the interactive dashboard, run:")
    print("    streamlit run app.py")
    print("=" * 75 + "\n")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Poll Results Visualizer: Automated analysis, cleaning, and reporting pipeline."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Path to custom survey CSV file. If omitted, synthetic data is generated.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1500,
        help="Number of synthetic respondents to generate (default: 1500).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible synthetic generation (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save generated charts and summary reports.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(
        input_file=args.input_file,
        samples=args.samples,
        seed=args.seed,
        output_dir=args.output_dir,
    )
