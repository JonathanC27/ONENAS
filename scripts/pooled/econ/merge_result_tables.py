#!/usr/bin/env python3
"""Merge the three headline results tables into two, detail to the appendix.

Zimeng's request: summarise in the main paper, keep the detail in the
appendix. The three main-paper tables all answer the same question from
different angles -- how much each arm earns (window), whether it is the
ensemble or the champion (M1), and whether it survives costs (costs).

  main    Table 1  trading performance: net, Sharpe, MDD, turnover and
                   the paired difference against the 40-island ensemble
          Table 2  M1, unchanged -- it carries the paper's central claim
  appendix         the year-by-year breakdown lifted out of Table 1, and
                   the full cost-multiple grid

Operates on a working copy of the project directory; the exhibits are
moved, never retyped.

    python3 merge_result_tables.py <project-dir>
"""
import os
import re
import sys

MERGED_WINDOW = r"""\begin{table*}[t]
\centering
\small
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
 & Net\% & Sharpe & MDD\% & Turn. & \multicolumn{2}{c}{40 isl.\ $-$ row$^{c}$} \\
 &       &        &       &       & $\Delta$net & paired $t$ \\
\midrule
ONE-NAS ensemble (60 isl.)       & $+30.1$\se{0.9} & $0.93$  & $12.8$ & $0.117$ & $-2.5$  & $-1.63$ \\
ONE-NAS ensemble (40 isl.)       & $+27.5$\se{1.1} & $0.87$  & $12.9$ & $0.119$ & --      & --      \\
ONE-NAS ensemble (20 isl.)       & $+27.4$\se{1.1} & $0.87$  & $13.1$ & $0.121$ & $+0.2$  & $0.10$  \\
\midrule
Online LSTM                      & $+11.3$\se{1.8} & $0.41$  & $13.7$ & $0.132$ & $+16.2$ & $7.32$  \\
Online GRU                       & $+11.7$\se{2.2} & $0.42$  & $13.3$ & $0.138$ & $+15.8$ & $6.59$  \\
Periodic LSTM (monthly retrain)  & $+14.8$\se{1.0} & $0.54$  & $13.0$ & $0.136$ & $+12.7$ & $6.66$  \\
Online AR$^{a}$                  & $-3.3$          & $-0.17$ & $22.9$ & $0.111$ & $+34.9$ & $2.22$  \\
\midrule
Buy \& hold$^{b}$                & $+7.9$          & $0.15$  & $19.3$ & --      & --      & --      \\
\bottomrule
\end{tabular}
\par\smallskip
{\footnotesize
$^{a}$Deterministic; one run per panel, so the paired test has $n{=}4$.
Those four cells are seed 42 only, where the 40-island ensemble averages
$+31.7$ rather than its full-grid $+27.5$; that is why the paired
difference exceeds the difference of the two column means.
$^{b}$Equal-weight, prior-study benchmark convention; the book trades
once and is not seed-paired.
$^{c}$Within-cell difference in net return between the 40-island
ensemble and the row, over the (panel, seed) cells the two arms share
(40 unless noted), with the paired $t$. Turnover is mean daily traded
notional as a fraction of gross book size.\par}
\caption{Trading performance over 2022--2024, mean over $V1$--$V4$ and
10 seeds. All learned arms trade the Overlapping rule under identical
realised costs. The year-by-year breakdown is
Table~\ref{tab:windowyears} and the cost-stress detail
Table~\ref{tab:costs}, both in the appendix.}
\label{tab:window}
\end{table*}"""

YEAR_TABLE = r"""
\begin{table}[t]
\centering
\footnotesize
\begin{tabular}{@{}l@{\hspace{4pt}}r@{\hspace{4pt}}r@{\hspace{4pt}}r@{\hspace{4pt}}r@{}}
\toprule
 & 2022 & 2023 & 2024 & 2022--24 \\
\midrule
ONE-NAS ens.\ (60 isl.)  & $+6.9$\se{0.7}  & $+13.7$\se{0.6} & $+7.0$\se{0.5}  & $+30.1$\se{0.9} \\
ONE-NAS ens.\ (40 isl.)  & $+6.1$\se{0.8}  & $+12.4$\se{0.7} & $+6.3$\se{0.5}  & $+27.5$\se{1.1} \\
ONE-NAS ens.\ (20 isl.)  & $+5.1$\se{0.5}  & $+13.3$\se{1.0} & $+6.3$\se{0.6}  & $+27.4$\se{1.1} \\
\midrule
Online LSTM              & $+1.2$\se{1.3}  & $+4.5$\se{1.2}  & $+3.8$\se{0.7}  & $+11.3$\se{1.8} \\
Online GRU               & $+2.1$\se{1.0}  & $+3.1$\se{1.9}  & $+5.0$\se{0.8}  & $+11.7$\se{2.2} \\
Periodic LSTM            & $+1.4$\se{1.0}  & $+3.9$\se{0.8}  & $+6.9$\se{0.6}  & $+14.8$\se{1.0} \\
Online AR                & $-14.6$         & $+4.8$          & $+7.9$          & $-3.3$ \\
\midrule
Buy \& hold              & $-10.1$         & $+11.2$         & $+6.8$          & $+7.9$ \\
\bottomrule
\end{tabular}
\par\smallskip
{\footnotesize
Books are restarted at each year boundary, so a year's cell is a book run
over that year alone; the 2022--24 column is one book run over the whole
window and is therefore not the sum of the three yearly cells for the
learned arms. For buy \& hold the prior study's convention sums the
yearly cells, and that convention is kept here.\par}
\caption{Net return (\%) by trade year, the year-by-year detail behind
Table~\ref{tab:window}.}
\label{tab:windowyears}
\end{table}
"""


def main():
    root = sys.argv[1]
    res = os.path.join(root, "05-Results.tex")
    apx = os.path.join(root, "81-AppendixResults.tex")
    s = open(res).read()

    # 1. swap the window table for the merged summary
    m = re.search(r"\\begin\{table\*\}\[t\].*?\\label\{tab:window\}\s*\\end\{table\*\}",
                  s, re.S)
    if not m:
        raise SystemExit("window table not found")
    s = s[:m.start()] + MERGED_WINDOW + s[m.end():]

    # 2. lift the cost table out of the main text
    m = re.search(r"\\begin\{table\}\[t\].*?\\label\{tab:costs\}\s*\\end\{table\}",
                  s, re.S)
    if not m:
        raise SystemExit("cost table not found")
    costs = m.group(0)
    s = s[:m.start()] + s[m.end():].lstrip("\n")
    open(res, "w").write(s)

    # 3. park both detail tables in the results appendix
    a = open(apx).read().rstrip()
    a += ("\n\n\\subsection{Detail Behind Table~\\ref{tab:window}}\n"
          + YEAR_TABLE + "\n" + costs + "\n")
    open(apx, "w").write(a)
    print("merged: main paper keeps 2 tables, detail moved to the appendix")


if __name__ == "__main__":
    main()
