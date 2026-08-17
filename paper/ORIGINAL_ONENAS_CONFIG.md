# ORIGINAL ONE-NAS PAPER (Lyu, Ororbia, Desell, arXiv 2302.10347 / Applied Soft Computing 2023)
# Verified from the PDF, sections 3.2, 4.2, 4.3:
# 1. PREDICTION IS FROM A SINGLE GENOME: Fig 1 'Select -> Global Best Genome -> Online
#    Prediction'; sec 4.3.4 'utilizing the previous best genome to provide predictions'.
#    => our island-champions ENSEMBLE is an extension of ours, not the published method.
# 2. ISLANDS: 'utilized 10, 20, 30, or 40 islands, each having its own elite population of 5
#    genomes which generated an additional 10 genomes per generation' (sec 4.2).
#    Sec 4.3.2: 'As the number of islands increases, the prediction performance improves.'
#    Fig 7: 40 islands + frequent repopulation gives the lowest MSE.
#    => OUR 8-ISLAND DEFAULT WAS BELOW THE PUBLISHED RANGE. Tonight's sweep replicates the
#    original authors' own finding in a new domain and on economic (not MSE) metrics.
#    Their split was elite 5 / generated 10 per island; ours is elite 8 / generated 5.
# 3. REPEATS: 'Each experiment was repeated 10 times' - matches our 10 seeds.
# 4. METRIC: MSE only. No trading, returns, portfolios, or transaction costs anywhere.
# 5. BASELINES: naive, moving average, exponential smoothing, online linear regression,
#    one- and two-layer online LSTM/GRU, online ARIMA - ALL SINGLE MODELS, no ensembles.
# 6. DATA: wind turbine (59k pts, 22 vars) and DJIA daily 1885-1962 (35,701 samples, univariate).
#    Training: 25-timestep subsequences, 600 sampled per genome per generation, validation on
#    the most recent 100 subsequences; 2000 generations for wind.
# 7. Repopulation frequency 100-500 tested; more frequent was better (ours: 50).
#
# CONSEQUENCE FOR OUR PAPER: adopting 20-40 islands is CONFIGURING THE METHOD AS PUBLISHED,
# not tuning on the test span. The 2016-2019 sweep (Amendment 6) remains a useful
# confirmation but is no longer the sole justification.
