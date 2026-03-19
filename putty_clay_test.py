"""
Putty-Clay Friction Test: Cross-Sectional Vintage Interaction
=============================================================

Tests whether the putty-clay assumption holds in electricity generation
using EIA-923 (generation/fuel consumption) and EIA-860 (generator
capital vintage) plant-level panel data.

CORE TEST
---------
The estimating equation at horizon h is:

  Δ_h log(RE_it / E_it) = α_i + α_rt
                         + β_h · Δ_h log(p^RE_rt / p^FF_rt)
                         + γ_h · [Δ_h log(p^RE_rt / p^FF_rt) × V̄_it]
                         + ε_it

where:
  RE_it / E_it  = renewable share of plant i's generation in quarter t
  p^RE / p^FF   = relative price of renewable vs fossil fuel energy
  V̄_it          = capacity-weighted share of plant i's capital past
                  economic life (the "vintage ripeness" measure)
  α_i           = plant fixed effects
  α_rt          = region × time fixed effects

NULL HYPOTHESES
---------------
Putty-clay:   β_h ≈ 0 for h < 36m, β_h < 0 for h > 36m  AND  γ_h > 0 all h
Putty-putty:  β_h < 0 for all h                           AND  γ_h = 0 all h

DATA SOURCES (downloaded automatically)
----------------------------------------
1. EIA-923: monthly fuel consumption and generation by plant
   URL: https://www.eia.gov/electricity/data/eia923/
   
2. EIA-860: annual generator inventory with installation years
   URL: https://www.eia.gov/electricity/data/eia860/
   
3. EIA regional electricity prices (Form EIA-861 / EIA API)
   URL: https://api.eia.gov/v2/electricity/retail-sales/
   
4. LBL Wind/Solar PPA prices by region
   URL: https://emp.lbl.gov/publications/  (manual download required)

REQUIREMENTS
------------
pip install pandas numpy scipy matplotlib requests tqdm openpyxl xlrd

USAGE
-----
python putty_clay_test.py --download    # download raw data (needs internet)
python putty_clay_test.py --build       # build analysis dataset
python putty_clay_test.py --estimate    # run regressions
python putty_clay_test.py --all         # do everything
python putty_clay_test.py --demo        # run on simulated data (no internet needed)
"""

import os
import sys
import argparse
import warnings
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import linalg, stats

warnings.filterwarnings('ignore')

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150,
})

# ── Directory setup ──────────────────────────────────────────────────────────
ROOT     = Path("putty_clay_data")
RAW_923  = ROOT / "eia923_raw"
RAW_860  = ROOT / "eia860_raw"
RAW_PRICE= ROOT / "prices_raw"
CLEAN    = ROOT / "clean"
RESULTS  = ROOT / "results"

for d in [RAW_923, RAW_860, RAW_PRICE, CLEAN, RESULTS]:
    d.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1: DATA DOWNLOAD
# ════════════════════════════════════════════════════════════════════════════

def _is_valid_zip(path):
    """Return True if path is a real ZIP file (not an HTML error page)."""
    import zipfile
    try:
        with zipfile.ZipFile(path):
            return True
    except Exception:
        return False


def download_eia923(years=range(2008, 2024)):
    """
    Download EIA Form 923 annual ZIP files and extract the Schedules 2-5 XLSX.

    EIA distributes all years as f923_{year}.zip from the archive URL.
    Each ZIP contains EIA923_Schedules_2_3_4_5_M_12_{year}_*.xlsx which we
    extract and save as eia923_{year}.xlsx.
    """
    import urllib.request, zipfile, io

    base = "https://www.eia.gov/electricity/data/eia923/archive/xls/"

    for year in years:
        out = RAW_923 / f"eia923_{year}.xlsx"
        if out.exists() and out.stat().st_size > 1_000_000:
            print(f"  EIA-923 {year}: already downloaded")
            continue
        if out.exists():
            out.unlink()   # remove stale / corrupted file

        zip_url = f"{base}f923_{year}.zip"
        print(f"  Downloading EIA-923 {year}…")
        try:
            with urllib.request.urlopen(zip_url, timeout=120) as resp:
                zip_bytes = resp.read()
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
                names = z.namelist()
                # Find the Schedules 2-5 workbook (monthly generation/fuel)
                # Matches both .xlsx and .xls; name contains "2_3_4_5"
                def _score(n):
                    nl = n.lower()
                    if '2_3_4_5' in nl: return 2
                    if 'schedule' in nl and nl.endswith(('.xlsx','.xls','.xlsm')): return 1
                    return 0
                sched_file = max(names, key=_score, default=None)
                if sched_file is None or _score(sched_file) == 0:
                    print(f"    ✗ No suitable XLSX found in ZIP for {year}: {names}")
                    continue
                # Preserve original extension (.xlsx or .xls)
                ext = Path(sched_file).suffix.lower()
                out = out.with_suffix(ext)
                with z.open(sched_file) as xf:
                    out.write_bytes(xf.read())
            print(f"    → extracted {sched_file!r} → {out} "
                  f"({out.stat().st_size/1e6:.1f} MB)")
        except Exception as e:
            print(f"    ✗ Failed for {year}: {e}")


def download_eia860(years=range(2008, 2024)):
    """
    Download EIA Form 860 annual ZIP files.
    Each ZIP contains the generator-level inventory with installation years.
    """
    import urllib.request

    base = "https://www.eia.gov/electricity/data/eia860/archive/xls/"

    for year in years:
        out = RAW_860 / f"eia860_{year}.zip"
        if out.exists() and _is_valid_zip(out):
            print(f"  EIA-860 {year}: already downloaded")
            continue
        if out.exists():
            out.unlink()   # remove stale / corrupted file

        # EIA-860 archive naming: eia860{4-digit-year}.zip  (e.g. eia8602023.zip)
        fname = f"eia860{year}.zip"
        url = base + fname
        print(f"  Downloading EIA-860 {year}…")
        try:
            urllib.request.urlretrieve(url, out)
            if _is_valid_zip(out):
                print(f"    → saved to {out} ({out.stat().st_size/1e6:.1f} MB)")
            else:
                out.unlink()
                print(f"    ✗ Downloaded file is not a valid ZIP")
        except Exception as e:
            print(f"    ✗ Failed: {e}")


def download_prices():
    """
    Download EIA regional electricity and natural gas prices via EIA API v2.

    Electricity : retail industrial prices by state (cents/kWh)
                  endpoint: /v2/electricity/retail-sales/data/
    Natural gas : citygate prices by state ($/MCF)
                  endpoint: /v2/natural-gas/pri/sum/data/ (process=PG057)

    Requires EIA_API_KEY environment variable.
    """
    import urllib.request

    api_key = os.environ.get("EIA_API_KEY", "")
    if not api_key:
        print("  ⚠ No EIA API key found. Set EIA_API_KEY environment variable.")
        print("  Get a free key at: https://www.eia.gov/opendata/register.php")
        print("  Skipping price download — will use approximate prices.")
        return

    def fetch_all_pages(url_base, label=""):
        """Fetch all records from EIA API v2 with pagination (api_key in URL)."""
        records = []
        offset = 0
        page_size = 5000
        while True:
            url = f"{url_base}&length={page_size}&offset={offset}"
            try:
                with urllib.request.urlopen(url, timeout=60) as resp:
                    data = json.loads(resp.read())
            except Exception as e:
                print(f"    Request failed at offset {offset}: {e}")
                break
            page   = data.get('response', {}).get('data', [])
            total  = int(data.get('response', {}).get('total', len(page)))
            records.extend(page)
            offset += len(page)
            if len(page) == 0 or offset >= total:
                break
            if label:
                print(f"    {label}: {offset:,}/{total:,} records…")
            time.sleep(0.25)   # respect rate limit
        return records

    # ── 1. Monthly retail electricity prices, industrial sector, by state ──
    # Facet is sectorid=IND (not sectorName); price in cents/kWh
    elec_path = RAW_PRICE / "elec_retail_industrial.json"
    if not elec_path.exists():
        print("  Downloading EIA retail electricity prices (industrial, monthly)…")
        url = (f"https://api.eia.gov/v2/electricity/retail-sales/data/"
               f"?api_key={api_key}"
               f"&frequency=monthly&data[0]=price"
               f"&facets[sectorid][]=IND"
               f"&start=2008-01&end=2023-12"
               f"&sort[0][column]=period&sort[0][direction]=asc")
        records = fetch_all_pages(url, "electricity")
        with open(elec_path, 'w') as f:
            json.dump(records, f)
        print(f"    → {len(records):,} records saved to {elec_path}")
    else:
        print(f"  Electricity prices: already downloaded ({elec_path.name})")

    # ── 2. Monthly natural gas citygate prices by state ─────────────────────
    # EIA v2: /v2/natural-gas/pri/sum/data/ process=PG1 ("City Gate Price")
    # duoarea format: "SXX" where XX is 2-letter state code; value in $/MCF
    gas_path = RAW_PRICE / "ng_citygate_monthly.json"
    if not gas_path.exists():
        print("  Downloading EIA natural gas citygate prices (monthly)…")
        url = (f"https://api.eia.gov/v2/natural-gas/pri/sum/data/"
               f"?api_key={api_key}"
               f"&frequency=monthly&data[0]=value"
               f"&facets[process][]=PG1"
               f"&start=2008-01&end=2023-12"
               f"&sort[0][column]=period&sort[0][direction]=asc")
        records = fetch_all_pages(url, "natural gas")
        if records:
            with open(gas_path, 'w') as f:
                json.dump(records, f)
            print(f"    → {len(records):,} records saved to {gas_path}")
        else:
            print("    No gas records returned — will use approximate prices.")
    else:
        print(f"  Natural gas prices: already downloaded ({gas_path.name})")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2: DATA PARSING — EIA-923
# ════════════════════════════════════════════════════════════════════════════

# Fuel type mappings
RENEWABLE_FUELS = {
    'SUN', 'WND', 'WAT', 'GEO',  # Solar, Wind, Hydro, Geothermal
    'WH',                          # Waste heat (clean)
    'WDS', 'WDL', 'OBL', 'OBS',   # Wood and biomass (debatable; include for now)
}
FOSSIL_FUELS = {
    'COL', 'NG', 'DFO', 'RFO', 'KER', 'PET',  # Coal, gas, oil products
    'BFG', 'OOG', 'SGC', 'PC',                  # Blast furnace gas, etc.
    'LFG', 'OBG',                                 # Landfill gas (fossil proxy)
}
CLEAN_RENEWABLE = {'SUN', 'WND', 'WAT', 'GEO'}  # Strictly clean renewables

# NERC region codes to names
NERC_REGIONS = {
    'SERC': 'Southeast', 'RFC':  'ReliabilityFirst',
    'WECC': 'West',       'MRO':  'Midwest Reliability',
    'SPP':  'Southwest Power Pool', 'TRE': 'Texas RE',
    'NPCC': 'Northeast',  'FRCC': 'Florida'
}


def parse_eia923_year(year):
    """
    Parse one year of EIA-923 into a clean plant-month DataFrame.
    Returns DataFrame with columns:
      plant_id, year, month, fuel_type, mmbtu, mwh
    """
    fpath = next(
        (RAW_923 / f"eia923_{year}{ext}" for ext in ('.xlsx', '.xls')
         if (RAW_923 / f"eia923_{year}{ext}").exists()),
        None
    )
    if fpath is None:
        return None

    # EIA-923 has a complex multi-sheet structure.
    # Sheet "Page 1 Generation and Fuel" contains what we need.
    # Row layout: header rows vary by year (usually 5 header rows).
    try:
        # Try header=5 with standard sheet name (2011+)
        df = pd.read_excel(fpath,
                           sheet_name='Page 1 Generation and Fuel',
                           header=5, dtype=str)
    except Exception:
        try:
            # 2011+ variant with 'Data' suffix in sheet name, still header=5
            df = pd.read_excel(fpath,
                               sheet_name='Page 1 Generation and Fuel Data',
                               header=5, dtype=str)
        except Exception:
            try:
                # First sheet, header=5 (generic fallback)
                df = pd.read_excel(fpath, sheet_name=0, header=5, dtype=str)
            except Exception:
                try:
                    # 2009-2010: sheet has 'Data' suffix and uses header=7
                    df = pd.read_excel(fpath,
                                       sheet_name='Page 1 Generation and Fuel Data',
                                       header=7, dtype=str)
                except Exception as e:
                    print(f"    Could not parse EIA-923 {year}: {e}")
                    return None

    # Standardise column names (EIA changes them slightly across years)
    df.columns = [str(c).strip().upper() for c in df.columns]

    # Find plant ID column
    plant_col = next((c for c in df.columns
                      if 'PLANT' in c and ('ID' in c or 'CODE' in c)), None)
    if plant_col is None:
        plant_col = next((c for c in df.columns if 'PLANT' in c), None)

    # Find fuel type column (tries ENERGY SOURCE first, then FUEL TYPE CODE)
    fuel_col = next((c for c in df.columns
                     if 'ENERGY' in c and 'SOURCE' in c), None)
    if fuel_col is None:
        fuel_col = next((c for c in df.columns
                         if 'FUEL' in c and 'TYPE' in c), None)
    if fuel_col is None:
        fuel_col = next((c for c in df.columns
                         if 'REPORTED FUEL' in c or 'AER FUEL' in c), None)

    if plant_col is None or fuel_col is None:
        print(f"    Could not identify key columns in EIA-923 {year}")
        print(f"    Available columns: {list(df.columns[:20])}")
        return None

    # Month columns: labelled JAN, FEB, ... DEC (for MMBtu or MWh)
    month_names = ['JAN','FEB','MAR','APR','MAY','JUN',
                   'JUL','AUG','SEP','OCT','NOV','DEC']

    # We need both MMBtu consumed and MWh generated.
    # EIA-923 page 1 has separate rows for consumption and generation.
    # Column naming varies by year:
    #   2011+: JAN (MMBtu), JAN (MWh)  — matched by month name + keyword
    #   2009-10: TOT_MMBTU_JAN, NETGEN_JAN  — matched by MMBTU/NETGEN keyword
    # Prefer MMBTU columns over raw QUANTITY (physical units) columns.
    def _find_month_col(cols, month, *keywords):
        # Priority: prefer exact MMBTU/NETGEN columns over generic QUANTITY
        for kw in keywords:
            hit = next((c for c in cols if month in c and kw in c), None)
            if hit:
                return hit
        return None

    mmbtu_cols = {m: _find_month_col(df.columns, m,
                                      'TOT_MMBTU', 'ELEC_MMBTUS',
                                      'MMBTU', 'CONSUMPTION', 'QUANTITY')
                  for m in month_names}
    mwh_cols   = {m: _find_month_col(df.columns, m,
                                      'NETGEN', 'MWH', 'GENERATION', 'NET GEN')
                  for m in month_names}

    # If column identification fails, use positional approach
    # EIA-923 typically has months in columns 9-20 (MMBtu) and 21-32 (MWh)
    numeric_cols = [c for c in df.columns
                    if df[c].str.replace(',','',regex=False)
                             .str.replace('.','',regex=False)
                             .str.replace('-','',regex=False)
                             .str.strip().str.isnumeric().mean() > 0.5]

    rows = []
    for _, row in df.iterrows():
        plant_id = str(row.get(plant_col, '')).strip()
        fuel     = str(row.get(fuel_col,  '')).strip().upper()

        if not plant_id.isdigit() or len(plant_id) < 1:
            continue
        if fuel not in RENEWABLE_FUELS and fuel not in FOSSIL_FUELS:
            continue

        for m_idx, month_name in enumerate(month_names, 1):
            mmbtu_col = mmbtu_cols.get(month_name)
            mwh_col   = mwh_cols.get(month_name)

            mmbtu = 0.0
            if mmbtu_col and mmbtu_col in row.index:
                try:
                    mmbtu = float(str(row[mmbtu_col]).replace(',','').strip())
                except:
                    mmbtu = 0.0

            mwh = 0.0
            if mwh_col and mwh_col in row.index:
                try:
                    mwh = float(str(row[mwh_col]).replace(',','').strip())
                except:
                    mwh = 0.0

            if mmbtu > 0 or mwh > 0:
                rows.append({
                    'plant_id': int(plant_id),
                    'year': year,
                    'month': m_idx,
                    'fuel_type': fuel,
                    'mmbtu': max(0, mmbtu),
                    'mwh':   max(0, mwh),
                })

    if not rows:
        return None

    return pd.DataFrame(rows)


def build_923_panel(years=range(2008, 2024)):
    """
    Build quarterly plant-level renewable share panel from EIA-923.
    Output: plant_id, year, quarter, re_share, total_mmbtu, total_mwh,
            re_mmbtu, ff_mmbtu, nerc_region
    """
    out_path = CLEAN / "eia923_quarterly.parquet"
    if out_path.exists():
        print("  Loading cached EIA-923 panel...")
        return pd.read_parquet(out_path)

    print("  Parsing EIA-923 annual files...")
    all_frames = []
    for y in years:
        df = parse_eia923_year(y)
        if df is not None:
            all_frames.append(df)
            print(f"    {y}: {len(df):,} fuel-plant-month observations")

    if not all_frames:
        print("  No EIA-923 data found. Run with --download first.")
        return None

    df = pd.concat(all_frames, ignore_index=True)

    # Create quarter
    df['quarter'] = ((df['month'] - 1) // 3) + 1
    df['is_re'] = df['fuel_type'].isin(RENEWABLE_FUELS).astype(float)
    df['is_ff'] = df['fuel_type'].isin(FOSSIL_FUELS).astype(float)

    # Aggregate to plant-quarter
    agg = (df.groupby(['plant_id','year','quarter'])
             .apply(lambda x: pd.Series({
                 'total_mmbtu': x['mmbtu'].sum(),
                 're_mmbtu':    x.loc[x['is_re']==1,'mmbtu'].sum(),
                 'ff_mmbtu':    x.loc[x['is_ff']==1,'mmbtu'].sum(),
                 'total_mwh':   x['mwh'].sum(),
                 're_mwh':      x.loc[x['is_re']==1,'mwh'].sum(),
             }))
             .reset_index())

    # Renewable share (use MWh as primary, MMBtu as fallback)
    agg['re_share_mwh'] = np.where(
        agg['total_mwh'] > 0,
        agg['re_mwh'] / agg['total_mwh'],
        np.nan
    )
    agg['re_share_mmbtu'] = np.where(
        agg['total_mmbtu'] > 0,
        agg['re_mmbtu'] / agg['total_mmbtu'],
        np.nan
    )
    # Use MWh where available, MMBtu otherwise
    agg['re_share'] = agg['re_share_mwh'].fillna(agg['re_share_mmbtu'])

    # Drop plants with no meaningful energy
    agg = agg[agg['total_mmbtu'] > 0].copy()

    # Add time index for panel
    agg['time'] = (agg['year'] - agg['year'].min()) * 4 + agg['quarter'] - 1

    agg.to_parquet(out_path, index=False)
    print(f"  EIA-923 panel: {len(agg):,} plant-quarter observations, "
          f"{agg['plant_id'].nunique():,} plants")
    return agg


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3: DATA PARSING — EIA-860 (VINTAGE)
# ════════════════════════════════════════════════════════════════════════════

# Expected economic life by fuel type (years)
# Sources: BEA Fixed Assets Tables, EIA technology reports
ECONOMIC_LIFE = {
    'COL':  40,   # Coal steam
    'NG':   30,   # Natural gas combined cycle
    'DFO':  30,   # Distillate fuel oil
    'RFO':  30,   # Residual fuel oil
    'NUC':  40,   # Nuclear (relicensed often to 60)
    'WAT':  50,   # Conventional hydro
    'GEO':  30,   # Geothermal
    'WND':  25,   # Wind
    'SUN':  30,   # Utility solar
    'WDS':  20,   # Wood/biomass
    'OBL':  20,   # Other biomass liquid
    'OBS':  20,   # Other biomass solid
    'default': 30,
}


def parse_eia860_year(year):
    """
    Parse one year of EIA-860 to get generator-level vintage data.
    Returns DataFrame: plant_id, generator_id, fuel_type, install_year,
                       capacity_mw, nerc_region, state
    """
    import zipfile

    zpath = RAW_860 / f"eia860_{year}.zip"
    if not zpath.exists():
        return None

    # Inside each ZIP there are multiple Excel files.
    # The generator file is typically named "3_1_Generator_Y{year}.xlsx"
    # or "GeneratorY{year}.xlsx" depending on the year.
    try:
        with zipfile.ZipFile(zpath, 'r') as z:
            names = z.namelist()
            # Find the generator schedule file
            gen_file = next(
                (n for n in names
                 if 'generator' in n.lower() and n.endswith('.xlsx')
                 and '3_1' in n or 'generator' in n.lower()),
                None
            )
            if gen_file is None:
                gen_file = next(
                    (n for n in names if n.endswith('.xlsx')), None
                )
            if gen_file is None:
                return None

            with z.open(gen_file) as f:
                # 2012+: header=1;  2009-2011: header=0 (no title row)
                # Detect by checking whether header=1 gives real column names
                import io
                raw_bytes = f.read()
            # Header tokens present in real EIA-860 column rows, never in data
            _HDR_TOKENS = {
                'UTILITY_ID', 'UTILITY ID', 'PLANT CODE', 'PLANT_CODE',
                'PLANT ID', 'PLANT_ID', 'GENERATOR_ID', 'GENERATOR ID',
                'NAMEPLATE', 'OPERATING_YEAR', 'OPERATING YEAR',
                'ENERGY SOURCE 1', 'ENERGY_SOURCE_1',
            }
            for hdr in [1, 0]:
                df_try = pd.read_excel(io.BytesIO(raw_bytes), header=hdr,
                                       dtype=str, nrows=2)
                cols_try = {str(c).strip().upper() for c in df_try.columns}
                if cols_try & _HDR_TOKENS:   # intersection → real header
                    df = pd.read_excel(io.BytesIO(raw_bytes), header=hdr, dtype=str)
                    break
            else:
                df = pd.read_excel(io.BytesIO(raw_bytes), header=1, dtype=str)

    except Exception as e:
        print(f"    EIA-860 {year}: {e}")
        return None

    df.columns = [str(c).strip().upper() for c in df.columns]

    # Map column names (EIA-860 naming varies across years)
    # Underscored versions cover 2009-2011; spaced versions cover 2012+
    col_map = {
        'plant_id':      ['PLANT ID', 'PLANT CODE', 'PLANTID', 'PLANT_ID',
                          'PLANT_CODE'],
        'generator_id':  ['GENERATOR ID', 'GENERATOR CODE', 'GENID',
                          'GENERATOR_ID'],
        'fuel_type':     ['ENERGY SOURCE 1', 'ENERGY SOURCE', 'FUEL TYPE 1',
                          'ENERGY_SOURCE_1', 'ENERGY SOURCE CODE 1'],
        'install_year':  ['OPERATING YEAR', 'YEAR INSTALLED', 'INSTYR',
                          'NAMEPLATE CAPACITY YEAR', 'OPERATING_YEAR',
                          'OPERATING MONTH'],   # fallback; month parsed below
        'capacity_mw':   ['NAMEPLATE CAPACITY (MW)', 'NAMEPLATE MW',
                          'SUMMER CAPACITY (MW)', 'NAMEPLATE_CAPACITY_MW',
                          'NAMEPLATE', 'SUMMER_CAPABILITY'],
        'nerc_region':   ['NERC REGION', 'NERC', 'BALANCING AUTHORITY CODE'],
        'state':         ['STATE', 'PLANT STATE', 'PLANT_STATE'],
        'status':        ['STATUS', 'GENERATOR STATUS', 'GENERATOR_STATUS'],
        'retire_year':   ['RETIREMENT YEAR', 'PLANNED RETIREMENT YEAR',
                          'RETIREMENT_YEAR'],
    }

    resolved = {}
    for key, candidates in col_map.items():
        for c in candidates:
            if c in df.columns:
                resolved[key] = c
                break

    required = ['plant_id', 'fuel_type', 'install_year', 'capacity_mw']
    if not all(k in resolved for k in required):
        missing = [k for k in required if k not in resolved]
        print(f"    EIA-860 {year}: missing columns {missing}")
        print(f"    Available: {list(df.columns[:15])}")
        return None

    rows = []
    for _, row in df.iterrows():
        try:
            pid  = str(row[resolved['plant_id']]).strip()
            fuel = str(row[resolved['fuel_type']]).strip().upper()
            yr   = str(row[resolved['install_year']]).strip()
            cap  = str(row[resolved['capacity_mw']]).strip()

            if not pid.isdigit(): continue
            try:
                install_year = int(float(yr))
                capacity_mw  = float(cap.replace(',',''))
            except:
                continue

            if install_year < 1900 or install_year > year + 1: continue
            if capacity_mw <= 0: continue

            row_dict = {
                'plant_id':    int(pid),
                'fuel_type':   fuel,
                'install_year':install_year,
                'capacity_mw': capacity_mw,
                'eia860_year': year,
            }

            if 'nerc_region' in resolved:
                row_dict['nerc_region'] = str(row[resolved['nerc_region']]).strip().upper()
            if 'state' in resolved:
                row_dict['state'] = str(row[resolved['state']]).strip().upper()
            if 'status' in resolved:
                row_dict['status'] = str(row[resolved['status']]).strip().upper()
            if 'retire_year' in resolved:
                try:
                    row_dict['retire_year'] = int(float(
                        str(row[resolved['retire_year']]).strip()
                    ))
                except:
                    row_dict['retire_year'] = np.nan
            if 'generator_id' in resolved:
                row_dict['generator_id'] = str(row[resolved['generator_id']]).strip()

            rows.append(row_dict)
        except:
            continue

    return pd.DataFrame(rows) if rows else None


def build_vintage_panel(years=range(2008, 2024)):
    """
    Build the quarterly plant-level vintage measure V̄_it.
    
    V̄_it = capacity-weighted share of plant i's capital stock
            that has been in service longer than its expected economic life
            as of year t.
    
    Three measures are computed:
      vintage_share_past_life : share of capacity past economic life
      vintage_avg_age         : capacity-weighted average age
      vintage_pre1990_share   : share of capacity installed before 1990
                                (time-invariant, used as IV)
    """
    out_path = CLEAN / "vintage_panel.parquet"
    if out_path.exists():
        print("  Loading cached vintage panel...")
        return pd.read_parquet(out_path)

    print("  Parsing EIA-860 annual files for vintage data...")
    all_frames = []
    for y in years:
        df = parse_eia860_year(y)
        if df is not None:
            all_frames.append(df)

    if not all_frames:
        print("  No EIA-860 data found.")
        return None

    gen_panel = pd.concat(all_frames, ignore_index=True)

    # Keep only operating units (status = OP or missing)
    if 'status' in gen_panel.columns:
        gen_panel = gen_panel[
            gen_panel['status'].isin(['OP', 'SB', 'OA', ''])
            | gen_panel['status'].isna()
        ].copy()

    # Compute vintage measures for each plant-year
    records = []
    for (plant_id, eia_year), grp in gen_panel.groupby(['plant_id','eia860_year']):
        total_cap = grp['capacity_mw'].sum()
        if total_cap <= 0:
            continue

        # Capacity-weighted average age
        ages = eia_year - grp['install_year']
        avg_age = (ages * grp['capacity_mw']).sum() / total_cap

        # Share past economic life
        def past_life(row):
            life = ECONOMIC_LIFE.get(row['fuel_type'],
                                      ECONOMIC_LIFE['default'])
            return (eia_year - row['install_year']) > life

        past_mask = grp.apply(past_life, axis=1)
        share_past = grp.loc[past_mask, 'capacity_mw'].sum() / total_cap

        # Share installed before 1990 (time-invariant IV)
        pre1990 = grp.loc[grp['install_year'] < 1990, 'capacity_mw'].sum()
        share_pre1990 = pre1990 / total_cap

        # NERC region: use EIA-860 column if present, else map from state
        nerc = None
        if 'nerc_region' in grp.columns:
            nerc_counts = grp['nerc_region'].dropna().value_counts()
            if len(nerc_counts) > 0:
                nerc = nerc_counts.index[0]
        if nerc is None and 'state' in grp.columns:
            state_mode = grp['state'].dropna().mode()
            if len(state_mode) > 0:
                nerc = get_state_nerc_map().get(state_mode.iloc[0])

        # Renewable capacity share (is this plant predominantly renewable?)
        re_cap = grp.loc[grp['fuel_type'].isin(RENEWABLE_FUELS),
                         'capacity_mw'].sum()
        re_cap_share = re_cap / total_cap

        records.append({
            'plant_id':             int(plant_id),
            'year':                 int(eia_year),
            'total_capacity_mw':    total_cap,
            'vintage_avg_age':      avg_age,
            'vintage_share_past':   share_past,
            'vintage_pre1990':      share_pre1990,
            're_capacity_share':    re_cap_share,
            'nerc_region':          nerc,
        })

    vdf = pd.DataFrame(records)

    # Expand to quarterly by repeating annual vintage measure
    # (vintage changes at annual frequency, EIA-860 is annual)
    quarterly_rows = []
    for _, row in vdf.iterrows():
        for q in [1, 2, 3, 4]:
            r = row.to_dict()
            r['quarter'] = q
            quarterly_rows.append(r)

    vdf_q = pd.DataFrame(quarterly_rows)
    vdf_q['time'] = (vdf_q['year'] - vdf_q['year'].min()) * 4 + vdf_q['quarter'] - 1

    vdf_q.to_parquet(out_path, index=False)
    print(f"  Vintage panel: {len(vdf_q):,} plant-quarter obs, "
          f"{vdf_q['plant_id'].nunique():,} plants")
    return vdf_q


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4: PRICE DATA
# ════════════════════════════════════════════════════════════════════════════

def build_price_panel():
    """
    Construct log(electricity_price / gas_price_mwh) by NERC region and quarter.

    price_re  = EIA retail industrial electricity price (cents/kWh → $/MWh × 10)
    price_ff  = EIA natural gas citygate price converted to $/MWh electricity
                equivalent: ($/MCF ÷ MCF_TO_MMBTU) × GAS_HEAT_RATE

    log_rel_price = log(price_re / price_ff)

    Falls back to build_approximate_prices() if EIA data is unavailable.
    """
    # Conversion constants
    MCF_TO_MMBTU = 1.02          # 1 MCF ≈ 1.02 MMBtu
    GAS_HEAT_RATE = 7.0          # MMBtu/MWh (CCGT heat rate)

    out_path = CLEAN / "price_panel.parquet"
    if out_path.exists():
        print("  Loading cached price panel...")
        return pd.read_parquet(out_path)

    state_nerc = get_state_nerc_map()

    # ── 1. Electricity prices ─────────────────────────────────────────────
    elec_path = RAW_PRICE / "elec_retail_industrial.json"
    elec_q = None
    if elec_path.exists():
        print("  Loading EIA retail electricity prices…")
        with open(elec_path) as f:
            records = json.load(f)
        if records:
            edf = pd.DataFrame(records)
            # Columns returned by /v2/electricity/retail-sales/data/:
            #   period (YYYY-MM), location (state abbrev), price (cents/kWh)
            # EIA retail-sales columns: period(YYYY-MM), stateid, price(cents/kWh)
            edf['price_mwh'] = pd.to_numeric(edf.get('price', edf.get('value')),
                                              errors='coerce') * 10   # → $/MWh
            # Parse period: accept YYYY-MM or YYYY
            edf['year'] = edf['period'].astype(str).str[:4].astype(int)
            if edf['period'].astype(str).str.len().max() >= 7:
                edf['month'] = edf['period'].astype(str).str[5:7].astype(int)
            else:
                edf['month'] = 6   # annual → mid-year
            edf['quarter'] = ((edf['month'] - 1) // 3) + 1
            # state column: try 'stateid' first, then 'location', 'state'
            for col in ['stateid', 'location', 'state']:
                if col in edf.columns:
                    edf['state'] = edf[col].astype(str).str.upper().str.strip()
                    break
            edf['nerc_region'] = edf['state'].map(state_nerc)
            edf = edf.dropna(subset=['price_mwh', 'nerc_region'])
            edf = edf[edf['price_mwh'] > 0]
            elec_q = (edf.groupby(['year', 'quarter', 'nerc_region'])
                          ['price_mwh'].median()
                          .reset_index()
                          .rename(columns={'price_mwh': 'price_re'}))
            print(f"    {len(elec_q):,} region-quarter electricity obs "
                  f"({elec_q['nerc_region'].nunique()} regions)")

    # ── 2. Natural gas citygate prices ────────────────────────────────────
    gas_path = RAW_PRICE / "ng_citygate_monthly.json"
    gas_q = None
    if gas_path.exists():
        print("  Loading EIA natural gas citygate prices…")
        with open(gas_path) as f:
            records = json.load(f)
        if records:
            gdf = pd.DataFrame(records)
            # EIA /v2/natural-gas/pri/sum/data/ returns columns like:
            #   period (YYYY-MM), duoarea (geographic code), value ($/MCF)
            # duoarea format: "SCA" (S + 2-letter state), "NUS" (national)
            val_col = next((c for c in ['value', 'price'] if c in gdf.columns), None)
            if val_col:
                gdf['price_mcf'] = pd.to_numeric(gdf[val_col], errors='coerce')
                # Convert $/MCF → $/MWh electricity equivalent
                gdf['gas_price_mwh'] = (gdf['price_mcf'] / MCF_TO_MMBTU) * GAS_HEAT_RATE

                gdf['year'] = gdf['period'].astype(str).str[:4].astype(int)
                if gdf['period'].astype(str).str.len().max() >= 7:
                    gdf['month'] = gdf['period'].astype(str).str[5:7].astype(int)
                else:
                    gdf['month'] = 6
                gdf['quarter'] = ((gdf['month'] - 1) // 3) + 1

                # Map duoarea → state → NERC region
                # EIA v2 gas duoarea format: "SXX" where XX = 2-letter state
                # e.g. "SAR"→AR, "SCA"→CA, "SFL"→FL; national is "NUS-Z00"
                area_col = next((c for c in ['duoarea', 'area', 'location', 'state']
                                 if c in gdf.columns), None)
                if area_col:
                    raw_area = gdf[area_col].astype(str).str.upper().str.strip()
                    # Drop leading 'S' from "SXX" format to get 2-letter state
                    gdf['state'] = raw_area.str[1:]   # e.g. "SAR" → "AR"
                    gdf['nerc_region'] = gdf['state'].map(state_nerc)
                    # Fallback: duoarea is already a 2-letter state code
                    mask = gdf['nerc_region'].isna()
                    gdf.loc[mask, 'nerc_region'] = raw_area[mask].map(state_nerc)
                    # Drop national aggregates (NUS) and unmapped areas
                    gdf = gdf.dropna(subset=['gas_price_mwh', 'nerc_region'])
                    gdf = gdf[gdf['gas_price_mwh'] > 0]
                    gas_q = (gdf.groupby(['year', 'quarter', 'nerc_region'])
                                 ['gas_price_mwh'].median()
                                 .reset_index()
                                 .rename(columns={'gas_price_mwh': 'price_ff'}))
                    print(f"    {len(gas_q):,} region-quarter gas obs "
                          f"({gas_q['nerc_region'].nunique()} regions)")

    # ── 3. Merge or fall back ─────────────────────────────────────────────
    if elec_q is None or len(elec_q) == 0:
        print("  No electricity price data — using approximate panel.")
        pp = build_approximate_prices()
        pp.to_parquet(out_path, index=False)
        return pp

    if gas_q is None or len(gas_q) == 0:
        print("  No gas price data — falling back to approximate panel.")
        pp = build_approximate_prices()
        pp.to_parquet(out_path, index=False)
        return pp

    price_panel = pd.merge(elec_q, gas_q,
                           on=['year', 'quarter', 'nerc_region'],
                           how='inner')

    if len(price_panel) == 0:
        print("  Merge of electricity and gas prices is empty — using approximate panel.")
        pp = build_approximate_prices()
        pp.to_parquet(out_path, index=False)
        return pp

    price_panel['log_rel_price'] = (
        np.log(price_panel['price_re']) - np.log(price_panel['price_ff'])
    )

    # Ensure all NERC regions present for every year-quarter
    # (fill gaps with national-average log_rel_price)
    all_regions = list(NERC_REGIONS.keys())
    all_yq = price_panel[['year', 'quarter']].drop_duplicates()
    full_idx = pd.MultiIndex.from_product(
        [all_yq['year'].unique(), [1,2,3,4], all_regions],
        names=['year', 'quarter', 'nerc_region']
    ).to_frame(index=False)
    price_panel = pd.merge(full_idx, price_panel,
                           on=['year', 'quarter', 'nerc_region'],
                           how='left')
    # Fill missing with year-quarter national median
    for col in ['price_re', 'price_ff', 'log_rel_price']:
        nat_avg = price_panel.groupby(['year', 'quarter'])[col].transform('median')
        price_panel[col] = price_panel[col].fillna(nat_avg)
    price_panel = price_panel.dropna(subset=['log_rel_price'])

    price_panel.to_parquet(out_path, index=False)
    print(f"  Price panel: {len(price_panel):,} region-quarter observations")
    print(f"    Regions: {sorted(price_panel['nerc_region'].dropna().unique())}")
    print(f"    Mean log_rel_price: {price_panel['log_rel_price'].mean():.3f}  "
          f"(negative → electricity < gas on MWh basis = expected)")
    return price_panel


def build_approximate_prices():
    """
    Build an approximate price panel using national time trends when
    region-level data is not available.
    
    Uses:
    - Solar/wind LCOE trends from NREL Annual Technology Baseline
    - Natural gas prices from EIA Henry Hub monthly data
    
    This approximation is less precise than the regional PPA data
    but gives the right sign and order of magnitude for the price signal.
    """
    # NREL ATB approximate utility-scale solar LCOE ($/MWh, nominal)
    # Source: NREL Annual Technology Baseline 2023
    solar_lcoe = {
        2008: 350, 2009: 280, 2010: 210, 2011: 160, 2012: 110,
        2013: 90,  2014: 75,  2015: 60,  2016: 50,  2017: 42,
        2018: 38,  2019: 35,  2020: 32,  2021: 34,  2022: 36,
        2023: 33,
    }
    # Wind LCOE ($/MWh)
    wind_lcoe = {
        2008: 105, 2009: 95,  2010: 88,  2011: 80,  2012: 68,
        2013: 62,  2014: 58,  2015: 52,  2016: 48,  2017: 45,
        2018: 42,  2019: 40,  2020: 38,  2021: 39,  2022: 41,
        2023: 38,
    }
    # Average renewable LCOE
    re_lcoe = {y: (solar_lcoe[y] + wind_lcoe[y]) / 2 for y in solar_lcoe}

    # EIA Henry Hub natural gas price ($/MMBtu), converted to $/MWh
    # (1 MMBtu ≈ 293 kWh for gas generation at ~33% efficiency → ×0.293/0.33)
    # Approximate delivered electricity cost from gas $/MWh
    gas_price_elec = {  # $/MWh equivalent at average heat rate
        2008: 65, 2009: 38, 2010: 40, 2011: 42, 2012: 30,
        2013: 40, 2014: 48, 2015: 35, 2016: 30, 2017: 35,
        2018: 42, 2019: 38, 2020: 30, 2021: 55, 2022: 80,
        2023: 50,
    }

    rows = []
    nerc_regions = list(NERC_REGIONS.keys())
    # Add regional variation (±20% random but consistent by region)
    np.random.seed(42)
    region_price_adj = {r: np.random.uniform(0.85, 1.15) for r in nerc_regions}

    for year in range(2008, 2024):
        for q in [1, 2, 3, 4]:
            for region in nerc_regions:
                adj = region_price_adj[region]
                p_re = re_lcoe.get(year, 60) * adj
                p_ff = gas_price_elec.get(year, 45) * adj
                rows.append({
                    'year':          year,
                    'quarter':       q,
                    'nerc_region':   region,
                    'price_re':      p_re,
                    'price_ff':      p_ff,
                    'log_rel_price': np.log(p_re) - np.log(p_ff),
                    'approx':        True,
                })

    df = pd.DataFrame(rows)
    print(f"  Approximate price panel: {len(df):,} region-quarter obs")
    return df


def get_state_nerc_map():
    """Map US state abbreviations to NERC regions."""
    return {
        'CT':'NPCC','ME':'NPCC','MA':'NPCC','NH':'NPCC',
        'NY':'NPCC','RI':'NPCC','VT':'NPCC',
        'DE':'RFC','IL':'RFC','IN':'RFC','KY':'RFC',
        'MD':'RFC','MI':'RFC','NJ':'RFC','OH':'RFC',
        'PA':'RFC','TN':'RFC','VA':'RFC','WV':'RFC','WI':'RFC',
        'AR':'SERC','AL':'SERC','FL':'FRCC','GA':'SERC',
        'LA':'SERC','MS':'SERC','MO':'SERC','NC':'SERC',
        'SC':'SERC','TX':'TRE',
        'AZ':'WECC','CA':'WECC','CO':'WECC','ID':'WECC',
        'MT':'WECC','NV':'WECC','NM':'WECC','OR':'WECC',
        'SD':'WECC','UT':'WECC','WA':'WECC','WY':'WECC',
        'IA':'MRO','MN':'MRO','ND':'MRO','NE':'MRO',
        'KS':'SPP','OK':'SPP',
    }


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5: MERGE AND CONSTRUCT ANALYSIS DATASET
# ════════════════════════════════════════════════════════════════════════════

def build_analysis_dataset():
    """
    Merge EIA-923, EIA-860 vintage, and price panels into a single
    analysis dataset at the plant-quarter level.
    """
    out_path = CLEAN / "analysis_dataset.parquet"
    if out_path.exists():
        print("  Loading cached analysis dataset...")
        return pd.read_parquet(out_path)

    gen_panel = build_923_panel()
    vin_panel = build_vintage_panel()
    prc_panel = build_price_panel()

    if gen_panel is None or vin_panel is None or prc_panel is None:
        print("  Cannot build analysis dataset — missing data.")
        return None

    print("  Merging panels...")

    # Merge generation + vintage on plant_id, year, quarter
    df = pd.merge(gen_panel, vin_panel,
                  on=['plant_id','year','quarter'],
                  how='inner',
                  suffixes=('','_v'))

    # Merge prices on nerc_region, year, quarter
    if 'nerc_region' not in df.columns:
        print("  Warning: nerc_region missing from plant data, "
              "using approximate national prices")
        df['nerc_region'] = 'RFC'  # default fallback

    df = pd.merge(df, prc_panel,
                  on=['nerc_region','year','quarter'],
                  how='left')

    # Fill missing prices with national average
    for col in ['log_rel_price','price_re','price_ff']:
        if col in df.columns:
            national_avg = prc_panel.groupby(['year','quarter'])[col].mean()
            df[col] = df[col].fillna(
                df[['year','quarter']].apply(
                    lambda r: national_avg.get((r['year'], r['quarter']), np.nan),
                    axis=1
                )
            )

    # Create log renewable share (bounded away from 0 and 1)
    eps = 1e-4
    df['re_share_clipped'] = df['re_share'].clip(eps, 1-eps)
    df['log_re_share'] = np.log(df['re_share_clipped'])
    df['logit_re_share'] = np.log(df['re_share_clipped'] /
                                   (1 - df['re_share_clipped']))

    # Drop extreme outliers
    df = df[df['total_mmbtu'] > 0].copy()
    df = df[df['re_share'].between(0, 1)].copy()
    df = df.dropna(subset=['log_rel_price','vintage_share_past',
                            'vintage_avg_age','log_re_share'])

    # Sort for panel structure
    df = df.sort_values(['plant_id','year','quarter']).reset_index(drop=True)

    # Create unique plant and region-time identifiers for FE
    df['plant_fe'] = pd.Categorical(df['plant_id']).codes
    df['region_time_fe'] = pd.Categorical(
        df['nerc_region'].astype(str) + '_' +
        df['year'].astype(str) + '_' +
        df['quarter'].astype(str)
    ).codes

    df.to_parquet(out_path, index=False)
    print(f"  Analysis dataset: {len(df):,} obs, "
          f"{df['plant_id'].nunique():,} plants, "
          f"{df['year'].nunique()} years")
    return df


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6: PANEL FIXED-EFFECTS ESTIMATOR WITH CLUSTERED SE
# ════════════════════════════════════════════════════════════════════════════

class PanelFE:
    """
    Within-estimator (Mundlak/Demeaning) panel fixed effects regression
    with two-way fixed effects (plant + region-time) and clustering.
    
    Handles:
      - Plant FE (demean within plant)
      - Region × time FE (demean within region-quarter)
      - Two-way FE via iterative demeaning (Gaure 2013)
      - Clustered standard errors at the plant level
    
    Implements the Frisch-Waugh-Lovell theorem for efficient computation.
    """

    def __init__(self, df, y_col, x_cols, cluster_col='plant_id',
                 fe_cols=('plant_fe', 'region_time_fe')):
        self.df         = df.copy()
        self.y_col      = y_col
        self.x_cols     = x_cols
        self.cluster_col= cluster_col
        self.fe_cols    = fe_cols
        self.n          = len(df)
        self.k          = len(x_cols)

    def _demean(self, arr, group_codes, n_iter=100, tol=1e-8):
        """Demean an array by group membership (within-group means)."""
        result = arr.copy().astype(float)
        for _ in range(n_iter):
            old = result.copy()
            means = np.bincount(group_codes,
                                weights=result,
                                minlength=group_codes.max()+1)
            counts= np.bincount(group_codes,
                                minlength=group_codes.max()+1).clip(1)
            result = result - means[group_codes] / counts[group_codes]
            if np.max(np.abs(result - old)) < tol:
                break
        return result

    def _two_way_demean(self, arr):
        """
        Iterative demeaning for two-way FE.
        Alternates between demeaning by FE1 and FE2 until convergence.
        """
        fe1 = self.df[self.fe_cols[0]].values.astype(int)
        fe2 = self.df[self.fe_cols[1]].values.astype(int)
        result = arr.copy().astype(float)
        for _ in range(200):
            old = result.copy()
            result = self._demean(result, fe1)
            result = self._demean(result, fe2)
            if np.max(np.abs(result - old)) < 1e-8:
                break
        return result

    def fit(self):
        """Run the within estimator and return results dict."""
        # Demean outcome
        y_raw = self.df[self.y_col].values.astype(float)
        y_dm  = self._two_way_demean(y_raw)

        # Demean regressors
        X_dm = np.column_stack([
            self._two_way_demean(self.df[c].values.astype(float))
            for c in self.x_cols
        ])

        # OLS on demeaned data
        try:
            beta, _, _, _ = linalg.lstsq(X_dm, y_dm)
        except:
            beta = np.linalg.lstsq(X_dm, y_dm, rcond=None)[0]

        # Residuals
        resid = y_dm - X_dm @ beta

        # Degrees of freedom correction
        # Two-way FE: subtract n_plants + n_region_times - 1
        n_fe1 = self.df[self.fe_cols[0]].nunique()
        n_fe2 = self.df[self.fe_cols[1]].nunique()
        df_resid = self.n - self.k - n_fe1 - n_fe2 + 1

        # Clustered standard errors (cluster by plant)
        cluster_ids = self.df[self.cluster_col].values
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        # Sandwich estimator: V = (X'X)^{-1} * B * (X'X)^{-1}
        # where B = sum_c (X_c' e_c e_c' X_c)
        XtX = X_dm.T @ X_dm
        try:
            XtX_inv = np.linalg.inv(XtX)
        except:
            XtX_inv = np.linalg.pinv(XtX)

        B = np.zeros((self.k, self.k))
        for cid in unique_clusters:
            mask = cluster_ids == cid
            Xc   = X_dm[mask]
            ec   = resid[mask]
            score = Xc.T @ ec
            B += np.outer(score, score)

        # Small-sample correction: (n-1)/(n-k) * G/(G-1)
        n, G = self.n, n_clusters
        correction = ((n - 1) / (n - self.k)) * (G / (G - 1))
        V_clust = correction * XtX_inv @ B @ XtX_inv

        se = np.sqrt(np.diag(V_clust))
        t_stat = beta / se
        p_val  = 2 * stats.t.sf(np.abs(t_stat), df=G - 1)

        # R-squared (within)
        ss_res = np.sum(resid**2)
        ss_tot = np.sum((y_dm - y_dm.mean())**2)
        r2_within = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        return {
            'beta':        beta,
            'se':          se,
            't_stat':      t_stat,
            'p_val':       p_val,
            'r2_within':   r2_within,
            'n_obs':       self.n,
            'n_clusters':  n_clusters,
            'df_resid':    df_resid,
            'x_cols':      self.x_cols,
            'resid':       resid,
        }


# ════════════════════════════════════════════════════════════════════════════
# SECTION 7: HORIZON REGRESSION — THE KEY TEST
# ════════════════════════════════════════════════════════════════════════════

def run_horizon_regressions(df, horizons=None, vintage_measure='vintage_share_past',
                             outcome='log_re_share'):
    """
    Main estimation: run the interaction regression at each horizon h.
    
    Returns a DataFrame with one row per horizon containing:
      h, beta_price, se_price, p_price,
      gamma_vintage, se_vintage, p_vintage,
      n_obs, n_clusters
    """
    if horizons is None:
        # Quarterly horizons: 0, 1, 2, ..., 20 quarters (0-5 years)
        horizons = list(range(0, 21))

    # Sort panel
    df = df.sort_values(['plant_id','year','quarter']).copy()
    df['t_idx'] = df['year'] * 4 + df['quarter']

    results = []
    print(f"  Running horizon regressions (0 to {max(horizons)} quarters)...")

    for h in horizons:
        # Construct h-period forward differences
        # For each observation (i,t), find observation (i, t+h)
        df_base   = df.copy()
        df_future = df.copy()
        df_future['t_idx'] = df_future['t_idx'] - h  # shift future back to match base

        merged = pd.merge(
            df_base[['plant_id','t_idx', outcome, 'log_rel_price',
                      vintage_measure, 'plant_fe', 'region_time_fe',
                      'nerc_region', 'year', 'quarter']],
            df_future[['plant_id','t_idx', outcome, 'log_rel_price']],
            on=['plant_id','t_idx'],
            suffixes=('_t', '_t+h'),
            how='inner'
        )

        if len(merged) < 100:
            continue

        # h-period differences
        merged['d_outcome']   = merged[f'{outcome}_t+h'] - merged[f'{outcome}_t']
        merged['d_log_price'] = (merged['log_rel_price_t+h'] -
                                  merged['log_rel_price_t'])

        # Interaction: price change × vintage measure (at base period)
        V = merged[vintage_measure].values
        # Demean the vintage measure for clean interpretation
        V_dm = V - V.mean()
        merged['price_x_vintage'] = merged['d_log_price'] * V_dm

        # Drop missing
        merged = merged.dropna(subset=['d_outcome','d_log_price',
                                        'price_x_vintage', vintage_measure])
        if len(merged) < 50:
            continue

        # Two-way FE panel regression with clustered SE
        try:
            reg = PanelFE(
                df      = merged,
                y_col   = 'd_outcome',
                x_cols  = ['d_log_price', 'price_x_vintage'],
                cluster_col = 'plant_id',
                fe_cols = ('plant_fe', 'region_time_fe'),
            )
            res = reg.fit()

            results.append({
                'h':               h,
                'h_years':         h / 4,
                'beta_price':      res['beta'][0],
                'se_price':        res['se'][0],
                'p_price':         res['p_val'][0],
                'gamma_vintage':   res['beta'][1],
                'se_vintage':      res['se'][1],
                'p_vintage':       res['p_val'][1],
                'ci95_beta_lo':    res['beta'][0] - 1.96*res['se'][0],
                'ci95_beta_hi':    res['beta'][0] + 1.96*res['se'][0],
                'ci95_gamma_lo':   res['beta'][1] - 1.96*res['se'][1],
                'ci95_gamma_hi':   res['beta'][1] + 1.96*res['se'][1],
                'n_obs':           res['n_obs'],
                'n_clusters':      res['n_clusters'],
                'r2_within':       res['r2_within'],
            })
        except Exception as e:
            print(f"    h={h}: estimation failed: {e}")
            continue

        if h % 4 == 0:
            r = results[-1]
            sig_b = '*' if r['p_price'] < 0.1 else ''
            sig_g = '*' if r['p_vintage'] < 0.1 else ''
            print(f"    h={h:2d}Q ({h/4:.1f}yr):  "
                  f"β_price={r['beta_price']:+.4f}{sig_b}  "
                  f"γ_vintage={r['gamma_vintage']:+.4f}{sig_g}  "
                  f"N={r['n_obs']:,}")

    return pd.DataFrame(results)


def run_pooled_regression(df, vintage_measure='vintage_share_past',
                          outcome='log_re_share',
                          horizons_short=[0,1,2,3],
                          horizons_long=[8,9,10,11,12]):
    """
    Pooled regression at short and long horizons to get clean coefficient table.
    Uses the same estimator but pools observations across specified horizons.
    """
    results = {}
    for label, hs in [('Short-run (0-3Q)', horizons_short),
                       ('Long-run (8-12Q)', horizons_long)]:
        df_pool = df.sort_values(['plant_id','year','quarter']).copy()
        df_pool['t_idx'] = df_pool['year'] * 4 + df_pool['quarter']
        rows = []
        for h in hs:
            df_base   = df_pool.copy()
            df_future = df_pool.copy()
            df_future['t_idx'] = df_future['t_idx'] - h
            merged = pd.merge(
                df_base[['plant_id','t_idx', outcome, 'log_rel_price',
                          vintage_measure, 'plant_fe', 'region_time_fe']],
                df_future[['plant_id','t_idx', outcome, 'log_rel_price']],
                on=['plant_id','t_idx'], suffixes=('_t','_t+h'), how='inner'
            )
            merged['d_outcome']   = merged[f'{outcome}_t+h'] - merged[f'{outcome}_t']
            merged['d_log_price'] = merged['log_rel_price_t+h'] - merged['log_rel_price_t']
            V = merged[vintage_measure].values
            merged['price_x_vintage'] = merged['d_log_price'] * (V - V.mean())
            merged['h'] = h
            rows.append(merged.dropna(subset=['d_outcome','d_log_price',
                                               'price_x_vintage']))
        if not rows: continue
        pooled = pd.concat(rows, ignore_index=True)
        try:
            reg = PanelFE(pooled, 'd_outcome',
                          ['d_log_price','price_x_vintage'],
                          fe_cols=('plant_fe','region_time_fe'))
            res = reg.fit()
            results[label] = res
        except Exception as e:
            print(f"  Pooled regression failed ({label}): {e}")

    return results


# ════════════════════════════════════════════════════════════════════════════
# SECTION 8: FIGURES
# ════════════════════════════════════════════════════════════════════════════

def make_figures(horizon_df, df_analysis, pooled_results=None):
    """
    Generate the four main figures:
    1. β_h profile (average price response by horizon)
    2. γ_h profile (vintage interaction by horizon) — THE KEY TEST
    3. Scatter: vintage age vs price responsiveness (cross-section)
    4. Summary coefficient table figure
    """

    fig_path = RESULTS / "figures"
    fig_path.mkdir(exist_ok=True)

    # ── Figure 1 & 2: Horizon profiles ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Putty-Clay Test: Horizon Regression Profiles",
                 fontsize=13, fontweight='bold', y=1.02)

    h    = horizon_df['h_years'].values
    sig5 = horizon_df['p_price'] < 0.05
    sig10= horizon_df['p_price'] < 0.10

    # Left: β_h (average price response)
    ax = axes[0]
    ax.fill_between(h,
                    horizon_df['ci95_beta_lo'],
                    horizon_df['ci95_beta_hi'],
                    alpha=0.15, color='#1565C0', label='95% CI')
    ax.plot(h, horizon_df['beta_price'],
            color='#1565C0', lw=2, marker='o', ms=4, label=r'$\hat\beta_h$')
    ax.scatter(h[sig5],  horizon_df.loc[sig5,  'beta_price'],
               color='#1565C0', s=60, zorder=5, label='Significant (5%)')
    ax.axhline(0, color='black', lw=0.8, ls='-')
    ax.axvline(3, color='gray',  lw=1.0, ls='--', alpha=0.6,
               label='3-year threshold')
    ax.set_xlabel("Horizon (years)", fontsize=11)
    ax.set_ylabel(r"$\hat\beta_h$: avg.\ price response", fontsize=11)
    ax.set_title(r"Average Price Elasticity Profile $\hat\beta_h$",
                 fontsize=10, pad=8)
    ax.legend(fontsize=8, frameon=False)

    # Annotate putty-clay prediction
    ax.text(0.5, 0.95,
            "Putty-clay: flat near 0 until ~3yr, then negative",
            transform=ax.transAxes, fontsize=8, color='gray',
            ha='center', va='top', style='italic')

    # Right: γ_h (vintage interaction — THE KEY TEST)
    ax2 = axes[1]
    sig5g  = horizon_df['p_vintage'] < 0.05
    sig10g = horizon_df['p_vintage'] < 0.10

    ax2.fill_between(h,
                     horizon_df['ci95_gamma_lo'],
                     horizon_df['ci95_gamma_hi'],
                     alpha=0.15, color='#B71C1C', label='95% CI')
    ax2.plot(h, horizon_df['gamma_vintage'],
             color='#B71C1C', lw=2, marker='s', ms=4, label=r'$\hat\gamma_h$')
    ax2.scatter(h[sig5g], horizon_df.loc[sig5g, 'gamma_vintage'],
                color='#B71C1C', s=60, zorder=5, label='Significant (5%)')
    ax2.axhline(0, color='black', lw=0.8)
    ax2.axvline(3, color='gray',  lw=1.0, ls='--', alpha=0.6)

    ax2.set_xlabel("Horizon (years)", fontsize=11)
    ax2.set_ylabel(r"$\hat\gamma_h$: vintage interaction", fontsize=11)
    ax2.set_title(r"Vintage Interaction Profile $\hat\gamma_h$" +
                  "\n(Key test: positive → putty-clay)",
                  fontsize=10, pad=4)
    ax2.legend(fontsize=8, frameon=False)

    # Annotation boxes
    verdict = "PUTTY-CLAY" if (
        horizon_df.loc[horizon_df['h_years'] < 1.5, 'gamma_vintage'].mean() > 0
        and
        horizon_df.loc[horizon_df['h_years'] < 1.5, 'p_vintage'].min() < 0.10
    ) else "INCONCLUSIVE"
    color = '#2E7D32' if verdict == "PUTTY-CLAY" else '#F57F17'
    ax2.text(0.97, 0.05,
             f"Verdict: {verdict}",
             transform=ax2.transAxes, fontsize=10, fontweight='bold',
             color=color, ha='right', va='bottom',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor=color, alpha=0.8))

    fig.tight_layout()
    out1 = fig_path / "fig1_horizon_profiles.png"
    fig.savefig(out1, bbox_inches='tight', dpi=150)
    print(f"  Saved {out1}")
    plt.close(fig)

    # ── Figure 2: Cross-sectional scatter ─────────────────────────────────
    # For each plant, compute average responsiveness at h=2 quarters
    # and plot against vintage age
    fig2, ax3 = plt.subplots(figsize=(8, 6))

    if 'vintage_avg_age' in df_analysis.columns:
        plant_stats = (df_analysis.groupby('plant_id')
                       .agg(avg_age=('vintage_avg_age','mean'),
                            avg_re_share=('re_share','mean'),
                            n_obs=('re_share','count'))
                       .reset_index())
        plant_stats = plant_stats[plant_stats['n_obs'] > 8].copy()

        # Bin by vintage age
        plant_stats['age_bin'] = pd.cut(plant_stats['avg_age'],
                                         bins=[0,10,20,30,40,60,200],
                                         labels=['0-10','10-20','20-30',
                                                 '30-40','40-60','60+'])
        bin_stats = (plant_stats.groupby('age_bin')
                     .agg(mean_re=('avg_re_share','mean'),
                          se_re=('avg_re_share', lambda x: x.std()/np.sqrt(len(x))),
                          n=('avg_re_share','count'))
                     .reset_index())

        xs = np.arange(len(bin_stats))
        ax3.bar(xs, bin_stats['mean_re'],
                yerr=1.96*bin_stats['se_re'],
                color='#1565C0', alpha=0.75, capsize=4,
                label='Mean renewable share (±95% CI)')
        ax3.set_xticks(xs)
        ax3.set_xticklabels(bin_stats['age_bin'], fontsize=9)
        ax3.set_xlabel("Capacity-weighted average vintage age (years)", fontsize=11)
        ax3.set_ylabel("Mean renewable share of generation", fontsize=11)
        ax3.set_title("Renewable Share by Plant Vintage Age\n"
                       "(Older plants have more 'ripe' capacity for replacement)",
                       fontsize=10)

        # Add N labels
        for i, (_, row) in enumerate(bin_stats.iterrows()):
            ax3.text(i, row['mean_re']+1.96*row['se_re']+0.005,
                     f"N={int(row['n'])}",
                     ha='center', fontsize=8, color='gray')

    fig2.tight_layout()
    out2 = fig_path / "fig2_vintage_re_share.png"
    fig2.savefig(out2, bbox_inches='tight', dpi=150)
    print(f"  Saved {out2}")
    plt.close(fig2)

    # ── Figure 3: Coefficient table ────────────────────────────────────────
    if pooled_results:
        fig3, ax4 = plt.subplots(figsize=(9, 4))
        ax4.axis('off')

        col_labels = ['Specification', 'β (price)',   'SE',
                      'γ (vintage)',   'SE',   'N', 'Clusters']
        rows_data  = []
        for label, res in pooled_results.items():
            b = res['beta']; se = res['se']; pv = res['p_val']
            def star(p):
                return ('***' if p<0.01 else '**' if p<0.05
                        else '*' if p<0.10 else '')
            rows_data.append([
                label,
                f"{b[0]:.4f}{star(pv[0])}",
                f"({se[0]:.4f})",
                f"{b[1]:.4f}{star(pv[1])}",
                f"({se[1]:.4f})",
                f"{res['n_obs']:,}",
                f"{res['n_clusters']:,}",
            ])

        tbl = ax4.table(cellText=rows_data,
                        colLabels=col_labels,
                        cellLoc='center', loc='center',
                        bbox=[0,0,1,1])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        for j in range(len(col_labels)):
            tbl[0,j].set_facecolor('#E3F2FD')
            tbl[0,j].set_text_props(fontweight='bold')

        ax4.set_title("Pooled Regression Results\n"
                       "Plant + region×time FE; clustered SE (plant level)\n"
                       "*p<0.10, **p<0.05, ***p<0.01",
                       fontsize=10, pad=10, loc='left')
        fig3.tight_layout()
        out3 = fig_path / "fig3_coef_table.png"
        fig3.savefig(out3, bbox_inches='tight', dpi=150)
        print(f"  Saved {out3}")
        plt.close(fig3)

    return fig_path


# ════════════════════════════════════════════════════════════════════════════
# SECTION 9: DEMO MODE (SIMULATED DATA)
# ════════════════════════════════════════════════════════════════════════════

def run_demo():
    """
    Run the full pipeline on simulated data that is constructed to exhibit
    putty-clay behaviour. Lets you validate the estimator and figures
    without downloading any data.
    
    DGP:
    - 500 plants, 60 quarters (2008-2022), 8 NERC regions
    - Each plant has a vintage structure drawn from realistic distribution
    - Price shocks are common within region-quarter (Bartik structure)
    - Energy mix responds to prices only through replacement investment:
        Δ(re_share)_it = γ * V̄_it * Δ(log_price)_rt + ε_it
      where V̄_it = share of capacity past economic life
    - Putty-clay: β_h = 0 for h<8, V̄-response positive for all h
    - Putty-putty counterfactual also generated for comparison
    """
    print("\n" + "="*60)
    print("DEMO MODE: Running on simulated putty-clay DGP")
    print("="*60)

    np.random.seed(42)
    N_PLANTS   = 500
    N_QUARTERS = 60    # 2008Q1 to 2022Q4
    N_REGIONS  = 8
    DELTA      = 1/30  # economic life = 30 years => depreciation = 1/30 per year
    ALPHA_TRUE = -0.15  # True γ (vintage × price interaction)
    SIGMA_EPS  = 0.03   # idiosyncratic noise

    # Assign plants to regions
    plant_region = np.random.randint(0, N_REGIONS, N_PLANTS)

    # Generate realistic vintage structures
    # Each plant has a distribution of capital ages
    # Mix of old coal (mean age 35yr) and newer gas (mean age 15yr)
    plant_type = np.random.choice(['coal','gas','mixed'],
                                   N_PLANTS, p=[0.3, 0.4, 0.3])
    vintage_age_plant = np.where(
        plant_type == 'coal',    np.random.uniform(25, 50, N_PLANTS),
        np.where(plant_type == 'gas', np.random.uniform(5, 25, N_PLANTS),
                 np.random.uniform(15, 40, N_PLANTS))
    )

    # Baseline renewable share (plants with old coal have low RE share)
    re_share_0 = np.clip(
        0.10 + 0.002*(vintage_age_plant - 20) + 0.3*(plant_type=='mixed')
        + np.random.normal(0, 0.05, N_PLANTS),
        0.02, 0.90
    )

    # Regional price processes: AR(1) with common trend
    # log(p_RE/p_FF) declines over time (RE becomes cheaper)
    log_price_region = np.zeros((N_REGIONS, N_QUARTERS))
    trend = np.linspace(0, -1.2, N_QUARTERS)  # RE gets 70% cheaper over sample
    for r in range(N_REGIONS):
        innov = np.random.normal(0, 0.08, N_QUARTERS)
        log_price_region[r] = trend + np.cumsum(innov - innov.mean()) * 0.5

    # Build panel
    rows = []
    for i in range(N_PLANTS):
        r_idx = plant_region[i]
        # Vintage measure: share of capacity past economic life
        # Increases over time as plant ages, resets when new investment occurs
        v_base = np.clip((vintage_age_plant[i] - 20) / 30, 0.0, 0.9)
        re_share_t = re_share_0[i]

        for t in range(N_QUARTERS):
            year    = 2008 + t // 4
            quarter = t % 4 + 1

            # Vintage share: increases slowly, with small jumps at new investment
            v_t = np.clip(v_base + t * 0.002 + np.random.normal(0, 0.02), 0.01, 0.95)

            # Price signal
            log_price_t = log_price_region[r_idx, t]

            # PUTTY-CLAY DGP: RE share changes only through replacement investment
            # Replacement intensity = V̄_it (more ripe capacity = more replacement)
            # Each quarter: δ fraction of capital turns over
            # RE share of new investment driven by price
            # => Δ(re_share) ≈ DELTA * V̄_it * f(price) + noise
            if t > 0:
                d_log_price = log_price_t - log_price_region[r_idx, t-1]
                # Only plants with high vintage respond (putty-clay mechanism)
                d_re = (DELTA * v_t * ALPHA_TRUE * d_log_price
                        + np.random.normal(0, SIGMA_EPS))
                re_share_t = np.clip(re_share_t + d_re, 0.01, 0.99)

            rows.append({
                'plant_id':          i,
                'year':              year,
                'quarter':           quarter,
                'time':              t,
                'nerc_region':       f"R{r_idx}",
                'log_rel_price':     log_price_t,
                're_share':          re_share_t,
                'log_re_share':      np.log(re_share_t / (1 - re_share_t)),
                'vintage_share_past':v_t,
                'vintage_avg_age':   vintage_age_plant[i] + t / 4,
                'vintage_pre1990':   float(vintage_age_plant[i] > 18),
                'total_mmbtu':       np.random.exponential(1000) + 100,
                'plant_fe':          i,
                'region_time_fe':    r_idx * N_QUARTERS + t,
            })

    df = pd.DataFrame(rows)
    print(f"  Simulated panel: {len(df):,} obs, {N_PLANTS} plants, "
          f"{N_QUARTERS} quarters")
    print(f"  True γ (vintage interaction) = {ALPHA_TRUE}")
    print(f"  True β (avg price response)  = ~0 short-run "
          f"(putty-clay)")

    # Run horizon regressions
    print("\n  Running horizon regressions on simulated data...")
    horizon_df = run_horizon_regressions(
        df, horizons=list(range(0, 17)),
        vintage_measure='vintage_share_past',
        outcome='log_re_share'
    )

    # Pooled regressions
    print("\n  Running pooled regressions...")
    pooled = run_pooled_regression(df, vintage_measure='vintage_share_past',
                                    outcome='log_re_share')

    # Save results
    if len(horizon_df) > 0:
        horizon_df.to_csv(RESULTS / "demo_horizon_results.csv", index=False)
        pooled_table(pooled)
        make_figures(horizon_df, df, pooled)

        # Print interpretation
        print("\n" + "─"*55)
        print("INTERPRETATION")
        print("─"*55)
        short_gamma = horizon_df.loc[horizon_df['h'] <= 3, 'gamma_vintage'].mean()
        short_beta  = horizon_df.loc[horizon_df['h'] <= 3, 'beta_price'].mean()
        long_beta   = horizon_df.loc[horizon_df['h'] >= 8, 'beta_price'].mean()

        print(f"  Average γ̂ at h≤3Q (short-run): {short_gamma:+.4f}")
        print(f"  Average β̂ at h≤3Q (short-run): {short_beta:+.4f}")
        print(f"  Average β̂ at h≥8Q (long-run):  {long_beta:+.4f}")
        print()
        if short_gamma > 0:
            print("  ✓ γ̂ > 0 at short horizons: CONSISTENT WITH PUTTY-CLAY")
            print("    Interpretation: older-vintage plants respond to prices")
            print("    even when average plant does not.")
        if abs(short_beta) < abs(long_beta):
            print("  ✓ |β̂| rises with horizon: CONSISTENT WITH PUTTY-CLAY")
        print()
        print("  Figures saved to:", RESULTS / "figures")


def pooled_table(pooled_results):
    """Print a formatted coefficient table to stdout."""
    if not pooled_results:
        return
    print("\n  ─" * 28)
    print(f"  {'Specification':<28} {'β_price':>10} {'γ_vintage':>12}  N")
    print("  ─" * 28)
    for label, res in pooled_results.items():
        b = res['beta']; se = res['se']; pv = res['p_val']
        def star(p):
            return ('***' if p<0.01 else '**' if p<0.05
                    else '*' if p<0.10 else '  ')
        print(f"  {label:<28} "
              f"{b[0]:>+8.4f}{star(pv[0])}  "
              f"{b[1]:>+8.4f}{star(pv[1])}  "
              f"{res['n_obs']:>7,}")
        print(f"  {'':28} ({se[0]:8.4f})    ({se[1]:8.4f})")
    print("  ─" * 28)
    print("  * p<0.10  ** p<0.05  *** p<0.01")
    print("  Plant + region×time FE; SE clustered at plant level")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 10: FULL PIPELINE ON REAL DATA
# ════════════════════════════════════════════════════════════════════════════

def run_full_pipeline():
    """Run download → build → estimate → figure pipeline on real EIA data."""

    print("\n" + "="*60)
    print("FULL PIPELINE: Real EIA Data")
    print("="*60)

    # Build analysis dataset
    df = build_analysis_dataset()
    if df is None:
        print("\nCould not build analysis dataset.")
        print("Please run with --download first to fetch EIA data.")
        print("Or run with --demo to test on simulated data.")
        return

    # Summary statistics
    print(f"\nDataset summary:")
    print(f"  Plants:          {df['plant_id'].nunique():,}")
    print(f"  Observations:    {len(df):,}")
    print(f"  Date range:      {df['year'].min()} Q{df['quarter'].min()} "
          f"– {df['year'].max()} Q{df['quarter'].max()}")
    print(f"  Mean RE share:   {df['re_share'].mean():.3f}")
    print(f"  Mean vintage age:{df['vintage_avg_age'].mean():.1f} years")

    # Main horizon regressions
    print("\nRunning main horizon regressions...")
    horizon_df = run_horizon_regressions(
        df,
        horizons=list(range(0, 21)),
        vintage_measure='vintage_share_past',
        outcome='log_re_share'
    )
    horizon_df.to_csv(RESULTS / "horizon_results_main.csv", index=False)

    # Robustness: logit outcome
    print("\nRobustness: logit-transformed outcome...")
    horizon_logit = run_horizon_regressions(
        df,
        horizons=list(range(0, 21)),
        vintage_measure='vintage_share_past',
        outcome='logit_re_share'
    )
    horizon_logit.to_csv(RESULTS / "horizon_results_logit.csv", index=False)

    # Robustness: alternative vintage measures
    for vmeas in ['vintage_avg_age', 'vintage_pre1990']:
        if vmeas in df.columns:
            print(f"\nRobustness: vintage measure = {vmeas}...")
            h_alt = run_horizon_regressions(
                df, horizons=list(range(0, 21)),
                vintage_measure=vmeas, outcome='log_re_share'
            )
            h_alt.to_csv(RESULTS / f"horizon_results_{vmeas}.csv", index=False)

    # Pooled regressions for table
    print("\nPooled regressions...")
    pooled = run_pooled_regression(df)
    pooled_table(pooled)

    # Figures
    print("\nGenerating figures...")
    make_figures(horizon_df, df, pooled)

    print(f"\nResults saved to: {RESULTS}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Putty-clay friction test using EIA plant-level data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python putty_clay_test.py --demo        # Run on simulated data (no internet needed)
  python putty_clay_test.py --download    # Download EIA-923 and EIA-860
  python putty_clay_test.py --build       # Build analysis dataset
  python putty_clay_test.py --estimate    # Run regressions
  python putty_clay_test.py --all         # Download + build + estimate
        """
    )
    parser.add_argument('--demo',     action='store_true',
                        help='Run on simulated putty-clay DGP (no data download needed)')
    parser.add_argument('--download', action='store_true',
                        help='Download EIA-923 and EIA-860 raw data')
    parser.add_argument('--build',    action='store_true',
                        help='Build analysis dataset from raw data')
    parser.add_argument('--estimate', action='store_true',
                        help='Run regressions on built dataset')
    parser.add_argument('--all',      action='store_true',
                        help='Download + build + estimate')
    parser.add_argument('--years',    type=str, default='2008-2023',
                        help='Year range, e.g. 2010-2020 (default: 2008-2023)')

    args = parser.parse_args()

    # Parse years
    if '-' in args.years:
        y1, y2 = args.years.split('-')
        years = range(int(y1), int(y2)+1)
    else:
        years = range(int(args.years), int(args.years)+1)

    if args.demo:
        run_demo()
        return

    if args.download or args.all:
        print("\n── Downloading EIA-923 ─────────────────────────────────")
        download_eia923(years)
        print("\n── Downloading EIA-860 ─────────────────────────────────")
        download_eia860(years)
        print("\n── Downloading prices ──────────────────────────────────")
        download_prices()

    if args.build or args.all:
        print("\n── Building panels ─────────────────────────────────────")
        build_923_panel(years)
        build_vintage_panel(years)
        build_price_panel()
        build_analysis_dataset()

    if args.estimate or args.all:
        run_full_pipeline()

    if not any([args.demo, args.download, args.build, args.estimate, args.all]):
        print(__doc__)
        print("\nRun with --demo to test the estimator on simulated data.")
        print("Run with --help for all options.")


if __name__ == "__main__":
    main()
