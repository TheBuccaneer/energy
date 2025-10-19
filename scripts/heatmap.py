#!/usr/bin/env python3
"""
Step 1: Time-of-Day Patterns - Heatmaps & by-hour metrics
Creates hourly aggregations and visualizations for electricity prices and CO2 intensity.

Data sources:
- Prices: EUR/kWh (originally EUR/MWh from ENTSO-E, converted via ÷1000)
- CI: Lifecycle carbon intensity in g/kWh (Scope 2 location-based, GHG Protocol)
- September 2024 is hourly (PT60M); 15-min MTU starts 30.09.2025 per EPEX SPOT
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple


def load_and_merge_data(price_file: str, ci_file: str) -> pd.DataFrame:
    """
    Load and merge price and CI data.
    
    Args:
        price_file: Path to day-ahead prices CSV
        ci_file: Path to carbon intensity CSV
    
    Returns:
        Merged DataFrame with harmonized timestamps
    """
    # Load price data
    df_price = pd.read_csv(price_file)
    if 'timestamp' not in df_price.columns:
        raise ValueError(f"Missing 'timestamp' column in {price_file}")
    
    # Load CI data
    df_ci = pd.read_csv(ci_file)
    if 'ts' not in df_ci.columns:
        raise ValueError(f"Missing 'ts' column in {ci_file}")
    
    # Harmonize timestamps
    # Prices: parse as timezone-naive
    df_price['timestamp'] = pd.to_datetime(df_price['timestamp'])
    
    # CI: parse and remove timezone (keep local time)
    # Using tz_localize(None) to strip timezone without conversion
    df_ci['ts'] = pd.to_datetime(df_ci['ts']).dt.tz_localize(None)
    
    print(f"Loaded {len(df_price)} price rows, {len(df_ci)} CI rows")
    
    # Merge on timestamp
    df = pd.merge(df_price, df_ci, left_on='timestamp', right_on='ts', how='inner')
    
    if len(df) == 0:
        raise ValueError("Merge resulted in 0 rows - timestamps don't match!")
    
    print(f"Merged: {len(df)} rows")
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour, weekday, and weekday_name features."""
    df['hour'] = df['timestamp'].dt.hour  # 0-23
    df['weekday'] = df['timestamp'].dt.weekday  # 0=Mon, 6=Sun
    
    weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    df['weekday_name'] = df['weekday'].map(lambda x: weekday_names[x])
    
    return df


def aggregate_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate metrics by area and hour.
    
    Returns DataFrame with columns:
        area, hour, mean_price_eur_kwh, mean_ci_g_per_kwh, neg_price_share
    """
    areas = {
        'DE': ('price_eur_kwh_de', 'de_ci_lifecycle_g_per_kwh'),
        'FR': ('price_eur_kwh_fr', 'fr_ci_lifecycle_g_per_kwh'),
        'PL': ('price_eur_kwh_pl', 'pl_ci_lifecycle_g_per_kwh'),
    }
    
    results = []
    
    for area, (price_col, ci_col) in areas.items():
        if price_col not in df.columns or ci_col not in df.columns:
            print(f"⚠️ Warning: Missing columns for {area}, skipping")
            continue
        
        # Aggregate by hour
        agg = df.groupby('hour').agg({
            price_col: 'mean',
            ci_col: 'mean',
        }).reset_index()
        
        # Calculate negative price share
        neg_share = df.groupby('hour').apply(
            lambda x: (x[price_col] < 0).mean()
        ).reset_index(name='neg_price_share')
        
        # Merge aggregations
        agg = agg.merge(neg_share, on='hour')
        
        # Rename columns
        agg.columns = ['hour', 'mean_price_eur_kwh', 'mean_ci_g_per_kwh', 'neg_price_share']
        agg['area'] = area
        
        results.append(agg)
    
    # Combine all areas
    final = pd.concat(results, ignore_index=True)
    
    # Reorder columns
    final = final[['area', 'hour', 'mean_price_eur_kwh', 'mean_ci_g_per_kwh', 'neg_price_share']]
    
    return final


def create_heatmap_data(df: pd.DataFrame, area: str, value_col: str) -> Tuple[np.ndarray, list, list]:
    """
    Create heatmap data matrix (weekday × hour).
    
    Returns:
        - 2D numpy array (7 weekdays × 24 hours)
        - list of weekday labels
        - list of hour labels
    """
    price_cols = {
        'DE': 'price_eur_kwh_de',
        'FR': 'price_eur_kwh_fr',
        'PL': 'price_eur_kwh_pl',
    }
    ci_cols = {
        'DE': 'de_ci_lifecycle_g_per_kwh',
        'FR': 'fr_ci_lifecycle_g_per_kwh',
        'PL': 'pl_ci_lifecycle_g_per_kwh',
    }
    
    if value_col == 'price':
        col = price_cols[area]
    else:
        col = ci_cols[area]
    
    # Create pivot table
    pivot = df.pivot_table(
        values=col,
        index='weekday',
        columns='hour',
        aggfunc='mean'
    )
    
    # Ensure we have all 7 weekdays and 24 hours
    weekdays = list(range(7))
    hours = list(range(24))
    pivot = pivot.reindex(index=weekdays, columns=hours)
    
    weekday_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    hour_labels = [str(h) for h in hours]
    
    return pivot.values, weekday_labels, hour_labels


def plot_heatmap(data: np.ndarray, weekday_labels: list, hour_labels: list,
                 title: str, cbar_label: str, outfile: Path):
    """
    Create heatmap using matplotlib.imshow.
    See: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Create heatmap with imshow
    im = ax.imshow(data, aspect='auto', cmap='YlOrRd', origin='upper')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(hour_labels)))
    ax.set_yticks(np.arange(len(weekday_labels)))
    ax.set_xticklabels(hour_labels)
    ax.set_yticklabels(weekday_labels)
    
    # Labels
    ax.set_xlabel('Hour of Day', fontsize=11)
    ax.set_ylabel('Weekday', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, fontsize=10)
    
    # Grid
    ax.set_xticks(np.arange(len(hour_labels))-.5, minor=True)
    ax.set_yticks(np.arange(len(weekday_labels))-.5, minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")


def plot_negative_price_share(by_hour: pd.DataFrame, area: str, outfile: Path):
    """Create bar chart of negative price share by hour."""
    data = by_hour[by_hour['area'] == area].sort_values('hour')
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(data['hour'], data['neg_price_share'], color='steelblue', alpha=0.7)
    
    ax.set_xlabel('Hour of Day', fontsize=11)
    ax.set_ylabel('Share of Hours with Negative Prices', fontsize=11)
    ax.set_title(f'Negative Price Frequency by Hour — {area} (Sep 2024)', 
                 fontsize=12, fontweight='bold')
    ax.set_xticks(range(24))
    ax.set_ylim(0, max(data['neg_price_share'].max() * 1.1, 0.1))
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate time-of-day heatmaps and hourly aggregations'
    )
    parser.add_argument('--prices', default='./day_ahead_sept2024_final.csv',
                        help='Path to price CSV')
    parser.add_argument('--ci', default='./ci_DE_FR_PL_hourly.csv',
                        help='Path to CI CSV')
    parser.add_argument('--outdir', default='.',
                        help='Base output directory')
    
    args = parser.parse_args()
    
    # Setup paths
    outdir = Path(args.outdir)
    tables_dir = outdir / 'tables'
    figs_dir = outdir / 'figs'
    tables_dir.mkdir(exist_ok=True)
    figs_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("STEP 1: Time-of-Day Patterns")
    print("=" * 70)
    
    # Load and merge data
    print("\n[1/4] Loading and merging data...")
    df = load_and_merge_data(args.prices, args.ci)
    
    # Add time features
    print("\n[2/4] Adding time features...")
    df = add_time_features(df)
    
    # Aggregate by hour
    print("\n[3/4] Aggregating by hour and area...")
    by_hour = aggregate_by_hour(df)
    
    # Save table
    out_table = tables_dir / 'by_hour_area.csv'
    by_hour.to_csv(out_table, index=False)
    print(f"\n✅ Saved: {out_table}")
    print(f"   {len(by_hour)} rows (3 areas × 24 hours)")
    
    # Create plots
    print("\n[4/4] Creating visualizations...")
    
    areas = ['DE', 'FR', 'PL']
    
    for area in areas:
        print(f"\n{area}:")
        
        # Price heatmap
        data, wd_labels, hr_labels = create_heatmap_data(df, area, 'price')
        plot_heatmap(
            data, wd_labels, hr_labels,
            f'Electricity Price by Day/Hour — {area} (Sep 2024)',
            'EUR/kWh',
            figs_dir / f'heatmap_price_{area.lower()}.png'
        )
        
        # CI heatmap
        data, wd_labels, hr_labels = create_heatmap_data(df, area, 'ci')
        plot_heatmap(
            data, wd_labels, hr_labels,
            f'CO₂ Intensity by Day/Hour — {area} (Sep 2024)',
            'g/kWh (Lifecycle)',
            figs_dir / f'heatmap_ci_{area.lower()}.png'
        )
        
        # Negative price bar chart
        plot_negative_price_share(
            by_hour, area,
            figs_dir / f'neg_price_share_by_hour_{area.lower()}.png'
        )
    
    print("\n" + "=" * 70)
    print("✅ DONE! Check tables/ and figs/ directories.")
    print("=" * 70)


if __name__ == '__main__':
    main()