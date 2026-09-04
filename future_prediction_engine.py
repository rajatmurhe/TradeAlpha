import os
import numpy as np
import pandas as pd


LIVE_FILE = "data/processed/latest_forecast.csv"

# ------------------------------------------------------------
# MODEL ASSUMPTION
# ------------------------------------------------------------
# XGBoost produces a next-day return signal.
# We do NOT assume that this signal remains constant for
# an entire month or year.
#
# Instead, the signal gradually decays as the horizon increases.
#
# Half-life = 10 trading days:
# after 10 trading days, the original daily signal has
# approximately 50% of its influence.
# ------------------------------------------------------------

SIGNAL_HALF_LIFE = 10.0


def load_latest_forecast():

    if not os.path.exists(LIVE_FILE):
        raise FileNotFoundError(
            f"Forecast file not found: {LIVE_FILE}"
        )

    df = pd.read_csv(LIVE_FILE)

    required = [
        "Ticker",
        "Close",
        "predicted_return",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    df["Close"] = pd.to_numeric(
        df["Close"],
        errors="coerce"
    )

    df["predicted_return"] = pd.to_numeric(
        df["predicted_return"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "Ticker",
            "Close",
            "predicted_return"
        ]
    ).copy()

    # Protect the calculation from impossible returns.
    df["predicted_return"] = (
        df["predicted_return"]
        .clip(-0.99, 0.99)
    )

    # One-day model direction
    df["predicted_direction"] = np.where(
        df["predicted_return"] > 0,
        "BULLISH",
        np.where(
            df["predicted_return"] < 0,
            "BEARISH",
            "NEUTRAL"
        )
    )

    df["predicted_price_1d"] = (
        df["Close"]
        *
        (
            1.0
            +
            df["predicted_return"]
        )
    )

    return df


def generate_projection(
    df,
    horizon
):

    horizon_days = {
        "1 Day": 1,
        "1 Week": 5,
        "1 Month": 21,
        "1 Year": 252,
    }

    if horizon not in horizon_days:
        raise ValueError(
            f"Unsupported horizon: {horizon}"
        )

    days = horizon_days[horizon]

    result = df.copy()

    daily_signal = (
        result["predicted_return"]
        .astype(float)
        .to_numpy()
    )

    # --------------------------------------------------------
    # SIGNAL DECAY
    # --------------------------------------------------------
    #
    # Day 0 = full model signal
    # Day 10 = approximately 50%
    # Day 20 = approximately 25%
    #
    # This prevents a one-day prediction from being blindly
    # compounded for 252 trading days.
    # --------------------------------------------------------

    decay_factors = (
        0.5
        **
        (
            np.arange(days)
            / SIGNAL_HALF_LIFE
        )
    )

    # --------------------------------------------------------
    # Calculate horizon-specific projected return
    # for every stock.
    # --------------------------------------------------------

    projected_returns = []

    for stock_return in daily_signal:

        # Generate the decaying daily signal
        daily_path = (
            stock_return
            *
            decay_factors
        )

        # Prevent impossible daily losses
        daily_path = np.clip(
            daily_path,
            -0.99,
            0.99
        )

        # Compound the decaying signal
        projected_return = (
            np.prod(
                1.0 + daily_path
            )
            - 1.0
        )

        projected_returns.append(
            projected_return
        )

    result["projected_return"] = (
        projected_returns
    )

    result["projected_price"] = (
        result["Close"]
        *
        (
            1.0
            +
            result["projected_return"]
        )
    )

    # --------------------------------------------------------
    # CONSISTENT INVESTOR OUTLOOK
    # --------------------------------------------------------

    result["direction"] = np.where(
        result["projected_return"] > 0,
        "BULLISH",
        np.where(
            result["projected_return"] < 0,
            "BEARISH",
            "NEUTRAL"
        )
    )

    return result


def portfolio_projection(
    df,
    investment
):

    # --------------------------------------------------------
    # Allocate more capital to assets with stronger positive
    # model signals.
    #
    # Negative signals do not receive additional allocation.
    # If everything is negative, use equal allocation rather
    # than creating an empty portfolio.
    # --------------------------------------------------------

    signals = np.maximum(
        df["predicted_return"]
        .to_numpy(dtype=float),
        0.0
    )

    total = signals.sum()

    if total > 0:

        weights = (
            signals
            /
            total
        )

    else:

        weights = (
            np.ones(len(df))
            /
            len(df)
        )

    result = df.copy()

    result["recommended_weight"] = (
        weights
    )

    result["recommended_amount"] = (
        result["recommended_weight"]
        *
        float(investment)
    )

    return result


def generate_prediction(
    horizon="1 Day",
    investment=100000
):

    df = load_latest_forecast()

    prediction = generate_projection(
        df,
        horizon
    )

    prediction = portfolio_projection(
        prediction,
        investment
    )

    # --------------------------------------------------------
    # PORTFOLIO EXPECTED RETURN
    # --------------------------------------------------------

    portfolio_return = float(
        np.sum(
            prediction[
                "recommended_weight"
            ]
            *
            prediction[
                "projected_return"
            ]
        )
    )

    projected_value = (
        float(investment)
        *
        (
            1.0
            +
            portfolio_return
        )
    )

    expected_profit = (
        projected_value
        -
        float(investment)
    )

    return {
        "horizon": horizon,
        "investment": float(investment),
        "portfolio_return": portfolio_return,
        "projected_value": projected_value,
        "expected_profit": expected_profit,
        "stocks": prediction,
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("FUTURE PREDICTION ENGINE")
    print("=" * 70)

    for horizon in [
        "1 Day",
        "1 Week",
        "1 Month",
        "1 Year"
    ]:

        result = generate_prediction(
            horizon=horizon,
            investment=100000
        )

        print()
        print(
            f"{horizon}:"
        )

        print(
            f"Projected Return: "
            f"{result['portfolio_return'] * 100:.2f}%"
        )

        print(
            f"Projected Value: "
            f"₹{result['projected_value']:,.2f}"
        )

        print(
            f"Expected P/L: "
            f"₹{result['expected_profit']:,.2f}"
        )

    print()
    print("=" * 70)
    print("FUTURE PREDICTION ENGINE READY")
    print("=" * 70)