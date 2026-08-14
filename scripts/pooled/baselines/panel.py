#!/usr/bin/env python3
"""Panel loading for the baseline suite.

A "panel" directory holds

    <TICKER>.csv      one per stock, date-aligned, header = feature names
    panel_dates.csv   row,date,PRC_<TICKER>...,TC_<TICKER>...

Every per-stock CSV must carry the same header; the feature list is read from
that header rather than hardcoded, so the v2 panels (which add RET_CS,
RET_CS_Z) load without a code change.

Stock order is the sorted per-stock CSV filename order, which score_stream.py
asserts equals the PRC_/TC_ column order in panel_dates.csv.  We assert the
same thing here so a baseline's stock index k always means the same ticker the
scorer means.
"""

import csv
import os

import numpy as np


class Panel:
    """Date-aligned feature/target/cost arrays for one 50-stock panel.

    Attributes
      dates      list[str]                 ISO date per panel row
      tickers    list[str]                 stock order (sorted filenames)
      features   list[str]                 feature column names, from the header
      X          ndarray [n_rows, n_stocks, n_feats]
      Y          ndarray [n_rows, n_stocks]   the target column (default RET)
      prc, tc    list[list[float]]         as score_stream.load_panel returns
    """

    def __init__(self, path, param="RET"):
        self.path = os.path.abspath(path)
        self.param = param
        self.dates, self.tickers, self.prc, self.tc = _load_panel_dates(self.path)

        header = None
        cols = []
        for t in self.tickers:
            p = os.path.join(self.path, t + ".csv")
            with open(p, newline="") as fh:
                rdr = csv.reader(fh)
                h = next(rdr)
                body = [row for row in rdr if row]
            if header is None:
                header = h
            elif h != header:
                raise SystemExit(
                    f"{p}: header {h} differs from {self.tickers[0]}.csv header "
                    f"{header}; the panel is not feature-consistent"
                )
            if len(body) != len(self.dates):
                raise SystemExit(
                    f"{p}: {len(body)} rows but panel_dates.csv has "
                    f"{len(self.dates)}"
                )
            cols.append(np.asarray(body, dtype=np.float64))

        self.features = list(header)
        if param not in self.features:
            raise SystemExit(
                f"target column {param!r} not in panel features {self.features}"
            )
        self.X = np.stack(cols, axis=1)          # [row, stock, feat]
        self.Y = self.X[:, :, self.features.index(param)].copy()
        if not np.isfinite(self.X).all():
            raise SystemExit(f"{self.path}: non-finite values in the feature data")
        self.date_index = {d: i for i, d in enumerate(self.dates)}

    # ------------------------------------------------------------------ util
    @property
    def n_rows(self):
        return len(self.dates)

    @property
    def n_stocks(self):
        return len(self.tickers)

    @property
    def n_feats(self):
        return len(self.features)

    def rows_between(self, start, end):
        """Panel row indices whose date lies in [start, end] (ISO strings)."""
        return [i for i, d in enumerate(self.dates)
                if (start is None or d >= start) and (end is None or d <= end)]

    def first_row_on_or_after(self, date):
        for i, d in enumerate(self.dates):
            if d >= date:
                return i
        return len(self.dates)

    def name(self):
        return os.path.basename(self.path.rstrip("/"))


def _load_panel_dates(path):
    """Mirror of score_stream.load_panel, returning (dates, tickers, prc, tc)."""
    p = os.path.join(path, "panel_dates.csv")
    with open(p, newline="") as fh:
        rdr = csv.reader(fh)
        header = next(rdr)
        prc_cols = [(i, c[4:]) for i, c in enumerate(header) if c.startswith("PRC_")]
        tc_cols = [(i, c[3:]) for i, c in enumerate(header) if c.startswith("TC_")]
        tickers = [t for _, t in prc_cols]
        if [t for _, t in tc_cols] != tickers:
            raise SystemExit("panel_dates.csv: TC_ order does not match PRC_ order")
        date_i = header.index("date")
        dates, prc, tc = [], [], []
        for row in rdr:
            dates.append(row[date_i])
            prc.append([float(row[i]) for i, _ in prc_cols])
            tc.append([float(row[i]) for i, _ in tc_cols])
    csvs = sorted(f[:-4] for f in os.listdir(path)
                  if f.endswith(".csv") and f != "panel_dates.csv")
    if csvs and csvs != tickers:
        raise SystemExit(
            "sorted per-stock CSV names do not match the PRC_ column order in "
            "panel_dates.csv; the stock index mapping would be ambiguous"
        )
    return dates, tickers, prc, tc


# --------------------------------------------------------------- normalisation

class RunningStandardizer:
    """Causal per-feature standardizer over the pooled (stock, time) sample.

    Expanding-window Welford moments; `observe(row_matrix)` must be called with
    a [n_stocks, n_feats] slice only once that panel row is public.  Parameter
    free by design, so it adds no hyperparameter to tune and cannot leak.
    """

    def __init__(self, n_feats):
        self.n = 0
        self.mean = np.zeros(n_feats)
        self.m2 = np.zeros(n_feats)

    def observe(self, mat):
        """Chan et al. parallel (batched) Welford update for a whole day."""
        m = mat.shape[0]
        if m == 0:
            return
        bmean = mat.mean(axis=0)
        bm2 = ((mat - bmean) ** 2).sum(axis=0)
        if self.n == 0:
            self.n, self.mean, self.m2 = m, bmean, bm2
            return
        delta = bmean - self.mean
        total = self.n + m
        self.mean = self.mean + delta * (m / total)
        self.m2 = self.m2 + bm2 + delta ** 2 * (self.n * m / total)
        self.n = total

    def transform(self, mat):
        if self.n < 2:
            return mat.copy()
        sd = np.sqrt(self.m2 / (self.n - 1))
        sd = np.where(sd > 1e-12, sd, 1.0)
        return (mat - self.mean) / sd


def build_design(panel, row, lags, standardizer=None, cs_demean=False,
                 intercept=True):
    """Feature matrix [n_stocks, d] for predicting panel row `row`.

    Uses feature rows `row-1 ... row-lags` only, i.e. strictly information
    available at the close of the day before the target day.  `row-1` is the
    day on which the forecast is issued.
    """
    return build_design_bulk(panel, [row], lags, standardizer, cs_demean,
                             intercept)[0]


def build_design_bulk(panel, rows, lags, standardizer=None, cs_demean=False,
                      intercept=True):
    """[n_rows, n_stocks, d] design for many target rows at once.

    Same semantics as build_design, vectorised so a batch refit does not pay a
    Python loop per day.  Every row r only ever reads feature rows r-1..r-lags.
    """
    rows = np.asarray(rows)
    blocks = []
    for l in range(1, lags + 1):
        m = panel.X[rows - l]                      # [n_rows, n_stocks, n_feats]
        if standardizer is not None:
            m = standardizer.transform(m)
        if cs_demean:
            m = m - m.mean(axis=1, keepdims=True)
        blocks.append(m)
    d = np.concatenate(blocks, axis=2)
    if intercept:
        d = np.concatenate([np.ones(d.shape[:2] + (1,)), d], axis=2)
    return d
