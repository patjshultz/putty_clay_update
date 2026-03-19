# Putty-Clay Friction Test

## What this project does
Tests whether putty-clay technology holds in US electricity generation
using EIA plant-level data. The core test is whether plants with older
capital vintages respond more to relative energy price changes than
plants with newer capital — the cross-sectional vintage interaction γ_h.

## Main script
putty_clay_test.py — full pipeline: download, build, estimate, figures.

Usage:
  python putty_clay_test.py --demo        # test on simulated data
  python putty_clay_test.py --download    # fetch EIA-923 and EIA-860
  python putty_clay_test.py --build       # build analysis dataset
  python putty_clay_test.py --estimate    # run regressions
  python putty_clay_test.py --all         # everything

## Data locations
  putty_clay_data/eia923_raw/    — raw EIA-923 Excel files by year
  putty_clay_data/eia860_raw/    — raw EIA-860 ZIP files by year
  putty_clay_data/prices_raw/    — price data (EIA API + LBL manual)
  putty_clay_data/clean/         — merged analysis parquet files
  putty_clay_data/results/       — regression outputs and figures

## Key variables
  re_share = renewable share of plant generation (outcome)
  log_rel_price = log(p_RE / p_FF) by NERC region and quarter
  vintage_share_past = share of plant capacity past expected economic life
  vintage_avg_age = capacity-weighted average age of generators
  vintage_pre1990 = share of capacity installed before 1990 (IV)

## Putty-clay test
  β_h: average price elasticity at horizon h (should be ~0 for h < 3yr)
  γ_h: vintage interaction (should be > 0 at ALL horizons if putty-clay)
  Null under putty-clay:   β_h ≈ 0 short-run, γ_h > 0 always
  Null under putty-putty:  β_h < 0 always,    γ_h = 0 always

## Known parsing issues
EIA file column names change across years. If a year fails with
"Could not identify key columns", open the file manually, find the
column names, and add them to the col_map dict in parse_eia923_year()
or parse_eia860_year().

## Environment
EIA_API_KEY must be set in environment for price download.
Python 3.10+, dependencies in requirements.txt.
