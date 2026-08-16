# Related-work notes (verified papers with URLs) — compiled 2026-08-15

Three literature sweeps; every entry verified to exist. Axes per paper:
(online?) (cross-sectional?) (costs?) (deployed?).

## A. Online / continual learning for stock prediction
1. DoubleAdapt — Zhao, Kong, Shen. KDD 2023. Meta-learned incremental updates
   for stock trend forecasting under drift; baselines INCLUDE rolling
   retraining (our core contrast, as a baseline row). Daily IC/rankIC on
   CSI300/500. Online yes / CS yes / costs light / deploy no.
   https://dl.acm.org/doi/10.1145/3580305.3599315
2. DDG-DA — Li, Yang, Liu, Xia, Bian. AAAI 2022. Forecasts the future data
   distribution to improve the NEXT periodic retrain ("smart retraining").
   https://ojs.aaai.org/index.php/AAAI/article/view/20327
3. ReCAP — KDD 2026. Regime-adaptive continual RL trading, framed against
   both rolling retrains and naive online fine-tuning.
   https://arxiv.org/abs/2606.00143
4. MetaDA — Huang et al. arXiv 2024 (arXiv-only; verify before citing as
   peer-reviewed). https://arxiv.org/abs/2401.03865
5. Capponi et al., "Nonstationarity-Complexity Tradeoff" — arXiv/SSRN Dec
   2025. Adaptive window/model selection beats fixed rolling retrains;
   cleanest academic treatment of when-to-retrain.
   https://arxiv.org/abs/2512.23596
6. ONE-NAS — Lyu & Desell, GECCO'22 companion; journal: Lyu, Ororbia, Desell,
   Applied Soft Computing 2023 (wind + univariate DJIA). Our direct ancestor.
   https://arxiv.org/abs/2202.13471 ;
   https://www.sciencedirect.com/science/article/abs/pii/S1568494623005409
7. Proceed — Zhao & Shen, KDD 2025. Online TSF adaptation with delayed labels
   (the ML formalization of prequential updating with label lag).
   https://dl.acm.org/doi/10.1145/3690624.3709210
8. IL-ETransformer — PLOS ONE 2025 (EWC incremental transformer, single-asset).
   https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0316955
9. Philps et al., Continual Learning Augmented Investment Decisions — NeurIPS
   FinML workshop 2018. https://arxiv.org/abs/1812.02340
10. Cavalcante & Oliveira — IJCNN 2015. Drift-detector-triggered retraining
    (the third pole: event-triggered). https://ieeexplore.ieee.org/abstract/document/7280721/
11. Online ARIMA — Liu, Hoi, Zhao, Sun. AAAI 2016. Lineage of our online-AR
    baseline. https://ojs.aaai.org/index.php/AAAI/article/view/10257
12. Li & Hoi, Online Portfolio Selection survey — ACM CSUR 2014. Delimit: OPS
    updates WEIGHTS online without forecasting; we learn FORECASTS online.
    https://dl.acm.org/doi/10.1145/2512962
13. FinRL-X — arXiv 2026. Documented Alpaca paper-trading deployment
    (Oct 2025-Mar 2026): precedent for our pilot. https://arxiv.org/pdf/2603.21330
14. FSNet — Pham et al. ~ICLR 2023. Standard online-TSF fine-tune baseline.
    https://arxiv.org/pdf/2202.11672

## B. Cross-sectional ML ranking + long-short evaluation (our metrics tradition)
1. Gu, Kelly, Xiu — RFS 2020. THE canonical ML cross-section horse race;
   monthly, decile long-shorts, no costs, yearly refits.
   https://academic.oup.com/rfs/article/33/5/2223/5758276
2. Krauss, Do, Huck — EJOR 2017. Template for DAILY cross-sectional ranking
   top/bottom-10 long-short WITH costs (5bps); edge decays post-2001.
   https://hal.science/hal-01515120
3. Fischer & Krauss — EJOR 2018. LSTM version; 0.46%/day pre-cost, mostly
   gone after costs post-2010. Standard "daily edges are small and shrinking."
   https://www.sciencedirect.com/science/article/abs/pii/S0377221717310652
4. Jegadeesh & Titman — JF 1993. Origin of our H-day overlapping-sleeve book.
   https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1993.tb04702.x
5. Poh, Lim, Zohren, Roberts — JFDS 2021. Cross-sectional momentum as
   learning-to-rank; rank-quality metrics; no costs.
   https://jfds.pm-research.com/content/3/2/70
6. FactorVAE — Duan et al. AAAI 2022. Daily IC 0.033 / RankIC 0.045 on CSI300
   (IC-magnitude calibration). https://ojs.aaai.org/index.php/AAAI/article/view/20369
7. MASTER — Li et al. AAAI 2024. Transformer; CSI300 IC 0.064/RankIC 0.076 on
   5-DAY labels with Alpha158 features (calibration: China + wide features +
   multi-day labels inflate ICs). https://ojs.aaai.org/index.php/AAAI/article/view/27767
8. Feng et al., Temporal Relational Ranking — ACM TOIS 2019. Stock prediction
   as ranking, daily US. https://dl.acm.org/doi/10.1145/3309547
9. ListFold — Zhang et al. arXiv 2021. Listwise loss for both tails of the
   long-short list. https://arxiv.org/abs/2104.12484
10. HireVAE — Wei et al. IJCAI 2023. "First online and adaptive factor model";
    one of only two online entries in this tradition.
    https://www.ijcai.org/proceedings/2023/0545.pdf
11. Avramov, Cheng, Metzker — Mgmt Sci 2023. ML profits attenuate sharply
    under liquidity screens + realistic costs (why we use a liquid universe
    and per-stock costs). https://pubsonline.informs.org/doi/abs/10.1287/mnsc.2022.4449
12. Tobek & Hronec — JFM 2021. ML anomaly signals survive on liquid universes
    internationally. https://www.sciencedirect.com/science/article/abs/pii/S1386418120300574
13. Zhang, Guo, Cao — arXiv 2020. "IC from a realistic stock selection model
    can hardly be materially different from zero" — the quotable calibration
    for daily IC ~0.01-0.02 being normal. https://arxiv.org/pdf/2010.08601
14. Guijarro-Ordonez, Pelger, Zanotti — Deep Learning Statistical Arbitrage
    (Mgmt Sci accepted). Daily US with cost analysis; strongest recent daily
    US anchor. https://arxiv.org/abs/2106.04028

## C. Evolutionary computation / NAS in finance
1. Lyu, Saxena, Nadeem, Zhang, Desell — arXiv 2024 (2410.17212). Offline EXAMM
   per-stock RNNs, Dow-30 long-short, beats DJI/S&P in 2022-23; costs not
   stated. CLOSEST PRECEDENT (same group); our deltas: online, pooled
   cross-section, per-stock costs, live pilot. https://arxiv.org/abs/2410.17212
2. EXAMM — Ororbia, ElSaid, Desell. GECCO 2019. https://dl.acm.org/doi/abs/10.1145/3321707.3321795
3. Allen & Karjalainen — JFE 1999. GP trading rules on S&P 500 FAIL to beat
   B&H once 0.25% one-way costs charged. The classic negative finding that
   frames the field. https://www.cs.montana.edu/courses/spring2007/536/materials/Lopez/genetic.pdf
4. Neely, Weller, Dittmar — JFQA 1997. The FX exception: GP rules survived
   costs there. https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/abs/is-technical-analysis-in-the-foreign-exchange-market-profitable-a-genetic-programming-approach/D959AD60856CECB9DB136AEFF6AD48FF
5. Potvin, Soriano, Vallée — C&OR 2004. GP per-stock rules, no costs in
   fitness, wins only in flat/falling markets. https://www.sciencedirect.com/science/article/abs/pii/S0305054803000637
6. AlphaEvolve — Cui et al. SIGMOD 2021. Evolutionary alpha mining on a
   NASDAQ cross-section (evolves formulas, offline). https://dl.acm.org/doi/10.1145/3448016.3457324
7. Nadkarni & Neves — ESWA 2018. NEAT+PCA trading. https://www.sciencedirect.com/science/article/abs/pii/S0957417418301519
8. Levchenko et al. — IJDSA 2024. Non-evolutionary NAS for financial TSF.
   https://link.springer.com/article/10.1007/s41060-024-00690-y
9. Zhang & Yu — IAAI-26 (Vol 40 No 47, DOI 10.1609/aaai.v40i47.41493). Our
   venue precedent; no evolution, no deployment.
   https://ojs.aaai.org/index.php/AAAI/article/view/41493
10. iRDPG — Liu et al. AAAI 2020. https://ojs.aaai.org/index.php/AAAI/article/view/5587
11. Yang et al. — ICAIF 2020 ensemble DRL, Dow-30 with costs.
    https://openfin.engineering.columbia.edu/sites/default/files/content/publications/ensemble.pdf
12. FinRL — Liu et al. ICAIF 2021. https://arxiv.org/abs/2111.09395

## The gap statement (for the intro/related work)
Across all three sweeps, no found paper combines: (1) online/continually
adapting model (architecture AND weights), (2) daily US cross-sectional
ranking, (3) per-stock transaction-cost accounting in a long-short book,
(4) an explicit online-vs-periodic-retraining comparison as the primary
experimental object, and (5) any live deployment element. The nearest
neighbors hold one or two of these each: DoubleAdapt/HireVAE (online, Chinese
universes, no costs), Krauss/Fischer line (daily US costed books, static
models), Lyu et al. 2024 (same algorithm family, offline, no costs),
FinRL-X (deployment, no learning claims).

## Calibration facts worth citing
- Published daily rank ICs of 0.03-0.08 come from Chinese A-shares with
  Alpha158-scale features and multi-day labels (FactorVAE, MASTER); daily US
  price-only studies show ~51% directional accuracy and post-2010 decay
  (Krauss, Fischer-Krauss). A daily rank IC of 0.01-0.02 on 200 liquid US
  names is in the credible published range (Zhang/Guo/Cao 2020 explicitly).
- Evolved-methods history: GP rules died on equity transaction costs (Allen &
  Karjalainen 1999) — we evolve forecasters, charge costs, and report what
  survives; the candor lineage is 25 years old.

## D. Online Portfolio Selection — the field that races online algorithms vs daily B&H (all verified)
Benchmark protocol of the whole field: cumulative wealth vs Market (uniform
B&H), Best Stock, BCRP (defined in the Li & Hoi survey).
1. Cover, Universal Portfolios — Mathematical Finance 1991.
   https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-9965.1991.tb00002.x
2. Helmbold, Schapire, Singer, Warmuth (EG) — Mathematical Finance 1998.
   https://onlinelibrary.wiley.com/doi/abs/10.1111/1467-9965.00058
3. Borodin, El-Yaniv, Gogan (Anticor) — NIPS 2003 / JAIR 2004.
   https://jair.org/index.php/jair/article/view/10380
4. Agarwal, Hazan, Kale, Schapire (Online Newton Step) — ICML 2006.
   https://dl.acm.org/doi/10.1145/1143844.1143846
5. Li, Zhao, Hoi, Gopalkrishnan (PAMR) — Machine Learning 2012.
   https://smusg.elsevierpure.com/en/publications/pamr-passive-aggressive-mean-reversion-strategy-for-portfolio-sel
6. Li & Hoi (OLMAR) — ICML 2012. Claims up to ~10^15x wealth on NYSE data.
   https://icml.cc/2012/papers/168.pdf
7. Li, Hoi, Gopalkrishnan (CORN) — ACM TIST 2011.
   https://dl.acm.org/doi/10.1145/1961189.1961193
8. Huang, Zhou, Li, Hoi, Zhou (RMR) — IJCAI 2013 / TKDE 2016.
   https://ieeexplore.ieee.org/document/7465840/
9. Li & Hoi survey — ACM CSUR 2014; OLPS toolbox — JMLR 2016.
   https://dl.acm.org/doi/10.1145/2512962 ;
   https://www.jmlr.org/papers/volume17/15-317/15-317.pdf

### The two pathologies (our design answers both)
- Costs: Blum & Kalai, Machine Learning 1999 (universality degrades with
  commissions) https://link.springer.com/article/10.1023/A:1007530728748 ;
  Li, Wang, Huang, Hoi (TCO), Quantitative Finance 2018 (the OLMAR/PAMR
  authors' own cost-aware fix — tacit admission naive OPS deteriorates
  sharply under costs) https://www.tandfonline.com/doi/full/10.1080/14697688.2017.1357831 ;
  Uziel & El-Yaniv, AISTATS 2020 http://proceedings.mlr.press/v108/uziel20a/uziel20a.pdf
- Data: Moon, Kim, Moon, arXiv 2019 — THE debunking: on NYSE(O) B&H makes
  14x while PAMR/OLMAR make >10 trillion x; on modern S&P 500 data with
  explicit+implicit costs the strategies "may fail even in favorable market
  conditions." https://arxiv.org/abs/1909.04327 (survivorship of NYSE(O)
  acknowledged in OLMAR's own arXiv limitations section.)
- Recent cost-aware OPS still benchmarking vs B&H: Guo, Gu, Fok, Ching, EJOR
  2023 https://www.sciencedirect.com/science/article/abs/pii/S0377221723003454 ;
  Moon & Yoon, Mathematics 2022 https://www.mdpi.com/2227-7390/10/7/1073 ;
  Zhang, Li, Yang, Lin (CAEGc), JORS 2023
  https://www.tandfonline.com/doi/full/10.1080/01605682.2022.2122737

### Positioning sentence
OPS learns the WEIGHTS online with no forecaster and historically "beat B&H"
only before costs and on biased datasets; we learn the FORECASTER online and
evaluate under the exact frictions (per-stock costs, survivorship
disclosure, turnover-capped book) that unraveled those claims.

## E. Deployment-flavored trading papers: what live results are benchmarked against (verified)
1. FinRL-Meta — NeurIPS 2021 DCAI wksp / NeurIPS 2022 D&B. **2-week Alpaca
   paper-trading window compared vs DJIA** (crypto vs BTC B&H), five-metric
   panel (cum ret, ann ret, vol, Sharpe, MDD). THE citable template for a
   short paper-trading pilot. https://arxiv.org/abs/2112.06753 ;
   https://arxiv.org/abs/2211.03107
2. FinRL-X — arXiv 2026. ~6-month Alpaca deployment; results vs SPY/QQQ in
   repo; paper body frames deployment as operational consistency.
   https://arxiv.org/abs/2603.21330
3. Agent Market Arena — arXiv 2025. Live LLM-agent leaderboard; per-asset
   B&H is the primary baseline. https://arxiv.org/abs/2510.11695
4. StockBench — arXiv 2025. "Most LLM agents struggle to beat buy-and-hold"
   as the headline finding. https://arxiv.org/abs/2510.02209
5. Intelligent Systematic Investment Agent — arXiv 2022. Real-money
   Robinhood deployment ~1yr; benchmark = INCUMBENT PROCESS (dollar-cost
   averaging), not B&H. https://arxiv.org/abs/2203.13125
6. Increase Alpha — arXiv 2025 (firm report). 4y production; vs S&P B&H but
   SELLS the near-zero market correlation + Sharpe/MDD, not raw wealth.
   https://arxiv.org/abs/2509.16707
7. Numerai fund reporting — live market-neutral fund; benchmarks vs
   STRATEGY-CLASS PEERS (quant equity market-neutral indices, AQR MN fund),
   correlation ~0, crisis drawdown anecdotes vs S&P. The market-neutral
   reporting norm. https://medium.com/numerai/numerai-outperforms-market-neutral-hedge-funds-by-29-raises-up-to-150m-9df9a0ce642
8. AI-Trader — arXiv 2025. Live agent benchmark with NO traditional baseline
   (cross-agent + risk behavior only). https://arxiv.org/abs/2512.10971
9. NEGATIVE FINDING: no IAAI Deployed/Emerging paper on live trading found —
   IAAI finance deployments cluster in fraud/credit/underwriting. A broker-
   API pilot is novel positioning in-track; no convention forces the
   benchmark choice.
10. Industry (JPM LOXM etc.): operational metrics vs incumbent execution
    algos; no peer-reviewed deployment papers.

### Norms takeaway
Academic live/paper-trading sections DO print an index/B&H row with the
five-metric panel even for ~2-week windows (FinRL-Meta) — descriptively,
never as significance claims. Market-neutral systems (Numerai, Increase
Alpha) foreground risk-adjusted/correlation metrics and peer or incumbent
comparisons instead of raw wealth races. Our pilot sheet = both norms:
descriptive five-metric panel incl. EW B&H reference + fidelity/ops metrics,
no alpha claims.
