"""
Quantitative Momentum Portfolio Construction for the Tokyo Stock Exchange (TSE).

This script implements a cross‑sectional momentum strategy adapted from
Wes Gray and Jack Vogel's **Quantitative Momentum** framework.  The goal is
to rank high‑liquidity Japanese equities by their medium‑term price
performance and by the *quality* or smoothness of that performance, then
construct an equally weighted portfolio of the top names.  The strategy
uses the standard 12‑month lookback with a one‑month skip, an approach
recommended by academic research.  By excluding the most recent month,
momentum portfolios avoid the short‑term reversal effect often observed in
equity returns【676164831756555†L138-L146】【676164831756555†L234-L246】.

Key design choices:

* **Universe:** A curated list of large and liquid stocks from the TSE
  Prime Market.  The list below contains nearly one hundred securities
  drawn from the Nikkei 225 constituents.  Each ticker is expressed in
  the four‑digit format used by the TSE with the ``.T`` suffix required by
  Yahoo Finance (e.g. Toyota Motor is ``7203.T``).  Users can amend this
  list to reflect their own investable universe.
* **Data:** The script pulls three years of daily adjusted close prices
  using the `yfinance` package.  Adjusted prices account for dividends
  and corporate actions, ensuring that returns are comparable across
  securities.
* **Momentum filter (12‑1):** For each stock we compute the total return
  over the past 252 trading days (approximately one calendar year),
  but we start the measurement 21 trading days before the most recent
  trading day.  In other words, the return is ``(price[t‑21] / price[t‑252]) – 1``.
  Academic studies show that skipping the most recent month helps
  eliminate micro‑structure noise and mean‑reversion【676164831756555†L138-L146】.
* **Smoothness filter:** For the subset of stocks that rank in the top
  30 % by momentum, we evaluate the *quality* of the momentum.  A stock
  that drifts steadily higher is preferred over one that rallies on a
  handful of large gaps.  To capture this behaviour we compute the
  proportion of positive daily returns between ``t‑252`` and ``t‑21``.  A
  higher fraction indicates smoother, more persistent momentum.  Other
  definitions—such as the information ratio (mean divided by standard
  deviation)—can easily be substituted.
* **Portfolio construction:** Among the candidates that pass the
  momentum filter we rank names by both their momentum score and their
  smoothness score.  The combined rank is the average of the two
  component ranks.  We select the top 15 stocks and allocate an equal
  weight of ``1/15`` to each.  The resulting portfolio is reported in a
  clean table showing ticker, company name, 12‑1 return (in percent),
  smoothness score (0–1 scale) and target weight.

Usage:

    python quant_momentum_tse.py

The script will fetch data via the internet.  Ensure that the `yfinance`
package is installed and that you have network access.  If you wish to
customise the stock universe or the ranking methodology, edit the
``TICKERS`` list or the functions defined below.
"""

from __future__ import annotations

import datetime as _dt
from typing import List

import numpy as np
import pandas as pd

try:
    import yfinance as yf  # type: ignore
except ImportError as exc:
    raise ImportError(
        "The yfinance package is required to run this script. Please install it via 'pip install yfinance'."
    ) from exc


# -----------------------------------------------------------------------------
# Configuration
#
# The following list contains numeric stock codes for many of the Nikkei 225
# constituents.  Each code is suffixed with ".T" to form the ticker as
# understood by Yahoo Finance.  While the Nikkei 225 includes 225 names, the
# universe below is limited to the larger, more liquid issues to keep the
# demonstration manageable.  Users can freely expand or shrink this list.
# -----------------------------------------------------------------------------

NUMERIC_CODES: List[str] = [
    # Automotive
    "7267", "7202", "7261", "7211", "7201", "7270", "7269", "7203", "7272",
    # Banking
    "8304", "8331", "8354", "8306", "8411", "8308", "5831", "8316", "8309", "7186",
    # Chemicals
    "3407", "4061", "4901", "4452", "3405", "4188", "4183", "4021",
    "6988", "4004", "4063", "4911", "4005", "4043", "4042", "4208",
    # Communications
    "9433", "9432", "9434", "9984",
    # Construction
    "1721", "1925", "1808", "1963", "1812", "1802", "1928", "1803", "1801",
    # Electric machinery
    "6857", "6770", "7751", "6902", "6954", "6504", "6702", "6501", "6861",
    "6971", "6920", "6479", "6503", "6981", "6701", "6594", "6645", "6752",
    "6723", "7752", "6963", "7735", "6724", "6753", "6758", "6526", "6976",
    "6762", "8035", "6506", "6841",
    # Electric power
    "9502", "9503", "9501",
    # Fishery
    "1332",
    # Foods
    "2802", "2502", "2914", "2801", "2503", "2269", "2282", "2871", "2002", "2501",
    # Gas
    "9532", "9531",
    # Glass & ceramics
    "5201", "5333", "5214", "5233",
]

# Convert codes to Yahoo Finance tickers with the '.T' suffix
TICKERS: List[str] = [f"{code}.T" for code in NUMERIC_CODES]


def fetch_price_data(tickers: List[str], lookback_years: int = 3) -> pd.DataFrame:
    """Fetch daily adjusted close price data for a list of tickers.

    Parameters
    ----------
    tickers : list of str
        List of ticker symbols (including '.T' suffix) to download.
    lookback_years : int, optional
        Number of years of historical data to retrieve.  Defaults to 3.

    Returns
    -------
    DataFrame
        Adjusted close prices indexed by date with tickers as columns.
    """
    end_date = _dt.datetime.today().date()
    start_date = end_date - _dt.timedelta(days=lookback_years * 365)

    # yfinance accepts a space‑separated string of tickers for batch download
    data = yf.download(
        tickers=" ".join(tickers),
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=True,
        group_by="ticker",
    )

    # When multiple tickers are requested, yfinance returns a dictionary-like
    # structure keyed by ticker.  Each value is a DataFrame of OHLCV columns.
    # We extract the 'Adj Close' for each ticker and align them on the index.
    adj_closes = {}
    for tkr in tickers:
        try:
            ticker_data = data[tkr]
            # Some tickers may not have an 'Adj Close' column (e.g. newly listed
            # names).  Skip these gracefully.
            if 'Adj Close' in ticker_data.columns:
                adj_series = ticker_data['Adj Close'].dropna()
                if len(adj_series) > 0:
                    adj_closes[tkr] = adj_series
        except Exception:
            # If the ticker cannot be retrieved, ignore it
            continue
    if not adj_closes:
        raise ValueError("No price data could be fetched for the specified tickers.")
    # Combine into a single DataFrame
    prices_df = pd.DataFrame(adj_closes)
    return prices_df


def compute_momentum_and_smoothness(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute 12‑1 momentum and smoothness scores for each stock.

    Parameters
    ----------
    prices : DataFrame
        Adjusted closing prices indexed by date with columns corresponding to
        tickers.

    Returns
    -------
    DataFrame
        Table with ticker, momentum return and smoothness score.
    """
    results = []
    # Compute daily returns once for smoothness calculation
    daily_returns = prices.pct_change()

    for tkr in prices.columns:
        series = prices[tkr].dropna()
        # Ensure there is enough history to compute a 12‑1 momentum signal
        if len(series) < 252 + 21:
            continue
        # Index alignment: use the most recent available date common to all series
        latest_idx = series.index[-1]
        # Compute lagged and past prices relative to the latest date
        try:
            price_t_minus_21 = series.shift(21).loc[latest_idx]
            price_t_minus_252 = series.shift(252).loc[latest_idx]
        except KeyError:
            # If the index is not aligned exactly (shouldn't happen), skip
            continue
        # Skip if past price is zero or missing
        if pd.isna(price_t_minus_21) or pd.isna(price_t_minus_252) or price_t_minus_252 <= 0:
            continue
        momentum_return = (price_t_minus_21 / price_t_minus_252) - 1.0

        # Extract the daily returns between t‑252 and t‑21
        returns_slice = daily_returns[tkr].dropna()
        # Determine the positions in the index corresponding to the lookback window
        slice_start = -252  # inclusive
        slice_end = -21     # exclusive: up to but not including t‑20
        window_returns = returns_slice.iloc[slice_start:slice_end]
        if len(window_returns) == 0:
            continue
        positive_fraction = float((window_returns > 0).sum()) / len(window_returns)

        results.append({
            'Ticker': tkr,
            'MomentumReturn': momentum_return,
            'Smoothness': positive_fraction,
        })

    return pd.DataFrame(results)


def rank_and_select(df: pd.DataFrame, top_fraction: float = 0.30, top_n: int = 15) -> pd.DataFrame:
    """Rank stocks by momentum and smoothness and select the top names.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing 'Ticker', 'MomentumReturn' and 'Smoothness'.
    top_fraction : float, optional
        Fraction of stocks to keep based on momentum before applying the
        smoothness filter.  Defaults to 0.30 (top 30 %).
    top_n : int, optional
        Number of final stocks to select.  Defaults to 15.

    Returns
    -------
    DataFrame
        Selected stocks with combined scores and portfolio weights.
    """
    if df.empty:
        raise ValueError("No momentum/smoothness data available for ranking.")
    # Rank by momentum (descending)
    df_sorted = df.sort_values('MomentumReturn', ascending=False).reset_index(drop=True)
    # Keep only the top fraction of stocks by momentum
    keep_count = max(int(len(df_sorted) * top_fraction), 1)
    df_top = df_sorted.iloc[:keep_count].copy()
    # Compute ranks within the filtered set (1 = best)
    df_top['MomentumRank'] = df_top['MomentumReturn'].rank(ascending=False, method='first')
    df_top['SmoothnessRank'] = df_top['Smoothness'].rank(ascending=False, method='first')
    # Combine ranks: lower sum is better; we use negative to sort descending
    df_top['CombinedScore'] = -0.5 * (df_top['MomentumRank'] + df_top['SmoothnessRank'])
    # Select top N by combined score
    df_selected = df_top.sort_values('CombinedScore', ascending=False).head(top_n).reset_index(drop=True)
    # Assign equal weights
    n_selected = len(df_selected)
    if n_selected > 0:
        df_selected['Weight'] = 1.0 / n_selected
    else:
        df_selected['Weight'] = np.nan
    return df_selected


def fetch_company_names(tickers: List[str]) -> dict[str, str]:
    """Retrieve company names for a list of tickers using yfinance.

    Parameters
    ----------
    tickers : list of str
        List of ticker symbols.

    Returns
    -------
    dict
        Mapping from ticker to company short name.  Missing names are filled with
        the ticker itself.
    """
    names = {}
    # yfinance allows batch retrieval via the Tickers class for efficiency
    batch = yf.Tickers(" ".join(tickers))
    for tkr, obj in batch.tickers.items():
        try:
            info = obj.info
            name = info.get('shortName') or info.get('longName')
        except Exception:
            name = None
        if not name:
            name = tkr
        names[tkr] = name
    return names


def main() -> None:
    """Main entry point: fetch data, compute signals, rank and report."""
    print("Fetching price data...")
    prices = fetch_price_data(TICKERS, lookback_years=3)
    print(f"Fetched data for {len(prices.columns)} tickers.")

    print("Computing momentum and smoothness metrics...")
    metrics_df = compute_momentum_and_smoothness(prices)
    if metrics_df.empty:
        print("No metrics computed. Exiting.")
        return
    print(f"Computed metrics for {len(metrics_df)} tickers.")

    print("Ranking and selecting stocks...")
    selection_df = rank_and_select(metrics_df, top_fraction=0.30, top_n=15)
    if selection_df.empty:
        print("No stocks selected. Exiting.")
        return

    # Fetch company names for selected tickers
    selected_tickers = selection_df['Ticker'].tolist()
    name_map = fetch_company_names(selected_tickers)
    selection_df['Company'] = selection_df['Ticker'].map(name_map)

    # Convert return and weight to percentages for display
    selection_df['MomentumReturnPct'] = selection_df['MomentumReturn'] * 100.0
    selection_df['WeightPct'] = selection_df['Weight'] * 100.0

    # Sort final table by combined score descending
    selection_df = selection_df.sort_values('CombinedScore', ascending=False)

    # Select columns and format nicely
    report_cols = ['Ticker', 'Company', 'MomentumReturnPct', 'Smoothness', 'WeightPct']
    report_df = selection_df[report_cols].copy()
    report_df.rename(columns={
        'MomentumReturnPct': '12‑1 Return (%)',
        'Smoothness': 'Smoothness Score',
        'WeightPct': 'Portfolio Weight (%)'
    }, inplace=True)

    # Print the final report table
    print("\nQuantitative Momentum Portfolio (Top 15 TSE Stocks)")
    print(report_df.to_string(index=False, formatters={
        '12‑1 Return (%)': lambda x: f"{x:6.2f}",
        'Smoothness Score': lambda x: f"{x:6.3f}",
        'Portfolio Weight (%)': lambda x: f"{x:5.2f}"
    }))


if __name__ == "__main__":
    main()
