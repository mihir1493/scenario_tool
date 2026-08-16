"""Generate the weekly panel: 4 categories x 241 weeks, week ending Sunday.

Volume is built as a log-log response surface with elasticities that are known
per category, so everything downstream can be scored against the truth. Two
things are deliberately not recoverable -- a persistent AR(1) factor and weekly
noise -- which is what keeps in-sample R^2 in a believable range instead of 0.999.

Each category also gets its own seasonal shape and its own media carryover rate,
because those are the hyperparameters `wk_model.py` validates per category. If
every category shared them, the validation step would be theatre.

    python wk_data.py     # writes data/weekly_panel.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import wk_config as cfg
import wk_features as feat


def weeks() -> pd.DatetimeIndex:
    return pd.date_range(cfg.HISTORY_START, cfg.HISTORY_END, freq="W-SUN")


def external_drivers(dates: pd.DatetimeIndex, rng) -> pd.DataFrame:
    """Weather and macro. One real-world path, observed by all four categories.

    Shared *inputs* are not pooling: each category still fits its own response to
    them, and three of the four will conclude temperature does nothing.
    """
    n = len(dates)
    t = np.arange(n, dtype=float)
    doy = dates.dayofyear.to_numpy()

    # Northern-hemisphere temperature: annual sinusoid peaking mid-July.
    # Floored at 3C rather than at zero: this is a market-average temperature, and
    # log(0.5) vs log(25) would hand ice cream a 4-unit swing in log space and a
    # frankly silly seasonal ratio. The floor keeps the log transform honest.
    temp = 14.0 + 10.0 * np.sin(2 * np.pi * (doy - 105) / 365.25) + rng.normal(0, 1.2, n)

    # Food CPI: inflation running hot into 2023, then easing.
    weekly_infl = np.where(t < 78, 0.0022, np.where(t < 130, 0.0009, 0.0005))
    cpi = 100.0 * np.exp(np.cumsum(weekly_infl)) * np.exp(rng.normal(0, 0.0012, n))

    # Consumer confidence: dips with the inflation peak, recovers slowly.
    cc = 95 - 18 * np.exp(-(((t - 80) / 35.0) ** 2)) + 0.045 * t
    cc = cc * np.exp(rng.normal(0, 0.010, n))

    return pd.DataFrame(
        {
            "date": dates,
            "avg_temp_c": np.clip(temp, 3.0, None),
            "cpi_food": cpi,
            "consumer_confidence": cc,
        }
    )


def category_drivers(cat: str, dates: pd.DatetimeIndex, ext: pd.DataFrame,
                     rng) -> pd.DataFrame:
    """The commercial levers a category team actually controls."""
    n = len(dates)
    t = np.arange(n, dtype=float)
    wk = dates.isocalendar().week.to_numpy(dtype=int)

    # Promo calendar: retail promotes into the holidays and around key seasons.
    promo_season = (
        0.55 * np.exp(-(((wk - 48) / 3.0) ** 2))
        + 0.40 * np.exp(-(((wk - 26) / 4.0) ** 2))
        + 0.25 * np.exp(-(((wk - 13) / 3.0) ** 2))
    )
    depth_base = {"Ice Cream": 22, "Ground Coffee": 18,
                  "Laundry Detergent": 25, "Baby Formula": 8}[cat]
    promo_depth = np.clip(
        depth_base * (1 + 0.5 * promo_season) * np.exp(rng.normal(0, 0.16, n)), 2, 45)

    share_base = {"Ice Cream": 38, "Ground Coffee": 31,
                  "Laundry Detergent": 45, "Baby Formula": 14}[cat]
    promo_share = np.clip(
        share_base * (1 + 0.6 * promo_season) * np.exp(rng.normal(0, 0.20, n)), 3, 85)

    # Feature + display runs with promo but is a separate merchandising decision.
    feat_base = {"Ice Cream": 20, "Ground Coffee": 16,
                 "Laundry Detergent": 24, "Baby Formula": 6}[cat]
    feature_display = np.clip(
        feat_base * (1 + 0.7 * promo_season) * np.exp(rng.normal(0, 0.28, n)), 0, 70)

    # Price: list price tracks food CPI, shelf price is net of the promo running.
    p0 = {"Ice Cream": 4.80, "Ground Coffee": 8.90,
          "Laundry Detergent": 11.50, "Baby Formula": 27.00}[cat]
    cpi_rel = ext["cpi_food"].to_numpy() / 100.0
    list_price = p0 * cpi_rel ** 0.80 * np.exp(rng.normal(0, 0.010, n))
    avg_price = list_price * (1 - (promo_share / 100.0) * (promo_depth / 100.0))

    # Competitor price: correlated with ours but with its own promo rhythm.
    competitor_price = (
        p0 * 0.97 * cpi_rel ** 0.78
        * (1 - 0.6 * (promo_share / 100.0) * (promo_depth / 100.0))
        * np.exp(rng.normal(0, 0.030, n))
    )

    # Distribution: a logistic build toward a category ceiling.
    d0, d1 = {"Ice Cream": (68, 87), "Ground Coffee": (74, 91),
              "Laundry Detergent": (82, 94), "Baby Formula": (61, 83)}[cat]
    dist = d0 + (d1 - d0) / (1 + np.exp(-(t - 110) / 32.0))
    dist = np.clip(dist * np.exp(rng.normal(0, 0.012, n)), 30, 98)

    # Media: flighted, with genuine dark weeks. Not a smooth spend line.
    tv_base = {"Ice Cream": 95, "Ground Coffee": 70,
               "Laundry Detergent": 55, "Baby Formula": 110}[cat]
    on_air = rng.random(n) < {"Ice Cream": 0.45, "Ground Coffee": 0.40,
                              "Laundry Detergent": 0.35, "Baby Formula": 0.55}[cat]
    tv_grps = np.where(on_air, tv_base * np.exp(rng.normal(0, 0.45, n)), 0.0)

    dig_base = {"Ice Cream": 42, "Ground Coffee": 55,
                "Laundry Detergent": 30, "Baby Formula": 75}[cat]
    digital_spend = np.clip(
        dig_base * (1 + 0.35 * promo_season) * np.exp(rng.normal(0, 0.35, n)), 0, None)
    digital_spend[rng.random(n) < 0.05] = 0.0

    return pd.DataFrame(
        {
            "date": dates, "category": cat,
            "avg_price": avg_price, "competitor_price": competitor_price,
            "promo_depth": promo_depth, "promo_share": promo_share,
            "feature_display": feature_display, "distribution_acv": dist,
            "tv_grps": tv_grps, "digital_spend": digital_spend,
        }
    )


def seasonal_curve(dates: pd.DatetimeIndex, shape: dict) -> np.ndarray:
    """The category's own yearly cycle, built from the Fourier amplitudes in config."""
    doy = dates.dayofyear.to_numpy(dtype=float)
    peak_doy = (shape["peak_week"] - 1) * 7 + 4
    out = np.zeros(len(doy))
    for j, amp in enumerate(shape["seasonal_amp"], start=1):
        if amp:
            out += amp * np.cos(2 * np.pi * j * (doy - peak_doy) / 365.25)
    return out


def holiday_curve(dates: pd.DatetimeIndex, cat: str) -> np.ndarray:
    """Holiday lifts that have nothing to do with any driver."""
    flags = feat.holiday_flags(pd.Series(dates))
    lift = {
        "Ice Cream": {"hol_yearend": 0.04, "hol_thanksgiving": 0.05,
                      "hol_july4": 0.14, "hol_easter": 0.03},
        "Ground Coffee": {"hol_yearend": 0.10, "hol_thanksgiving": 0.09,
                          "hol_july4": -0.03, "hol_easter": 0.02},
        "Laundry Detergent": {"hol_yearend": -0.05, "hol_thanksgiving": 0.03,
                              "hol_july4": -0.02, "hol_easter": 0.02},
        "Baby Formula": {"hol_yearend": 0.02, "hol_thanksgiving": 0.01,
                         "hol_july4": 0.0, "hol_easter": 0.0},
    }[cat]
    return sum(lift[c] * flags[c].to_numpy() for c in flags.columns)


def generate() -> pd.DataFrame:
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    dates = weeks()
    ext = external_drivers(dates, rng)

    frames = []
    for cat in cfg.CATEGORIES:
        shape = cfg.CATEGORY_SHAPE[cat]
        betas = cfg.TRUE_ELASTICITIES[cat]
        n = len(dates)

        df = category_drivers(cat, dates, ext, rng).merge(ext, on="date")
        # Apply this category's own carryover before reading the elasticities off,
        # exactly as the model will when it fits.
        adstocked = feat.apply_adstock(df, shape["adstock_decay"])

        contrib = np.zeros(n)
        for name in cfg.DRIVER_NAMES:
            if not betas[name]:
                continue  # a true zero: this driver does nothing here
            lx = feat.log_driver(adstocked[name].to_numpy(), name)
            contrib += betas[name] * (lx - lx.mean())

        trend = shape["trend"] * np.arange(n) / 52.0
        seasonal = seasonal_curve(dates, shape)
        holiday = holiday_curve(dates, cat)

        latent = np.zeros(n)
        for i in range(1, n):
            latent[i] = cfg.LATENT_AR * latent[i - 1] + rng.normal(0, cfg.LATENT_SD)
        noise = rng.normal(0, cfg.NOISE_SD, n)

        log_v = (np.log(shape["base_volume"]) + trend + seasonal + holiday
                 + contrib + latent + noise)
        df["volume"] = np.round(np.exp(log_v))
        frames.append(df)

    panel = pd.concat(frames, ignore_index=True)
    cols = ["date", "category", "volume"] + cfg.DRIVER_NAMES
    return panel[cols].sort_values(["category", "date"]).reset_index(drop=True)


def true_elasticity_table() -> pd.DataFrame:
    """Ground truth as a category x driver frame, for scoring."""
    return pd.DataFrame(cfg.TRUE_ELASTICITIES).T.loc[cfg.CATEGORIES, cfg.DRIVER_NAMES]


if __name__ == "__main__":
    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    panel = generate()
    panel.to_csv(cfg.PANEL_CSV, index=False)
    print(panel.head().to_string(index=False))
    print(f"\n{len(panel)} rows ({len(cfg.CATEGORIES)} categories x "
          f"{len(weeks())} weeks) -> {cfg.PANEL_CSV}")
    print("\nWeekly volume by category:")
    print(panel.groupby("category")["volume"].agg(["mean", "min", "max"]).round(0).to_string())
    print("\nTrue elasticities (0 means the driver genuinely does nothing there):")
    print(true_elasticity_table().to_string())
