README
Codes for agro-ecological regions in Sri Lanka: Ampara (DZ), Kurunegala (IZ), and
Ratnapura (WZ).

SCRIPTS
The location is selected at the top of each script with location = 'ampara' / 'kuru' / 'rath'
(switch case).
CODE_MCMC_and_BSI.m
Parameter estimation of the D-vine copula (DRAM) and BSI vs SSI index construction.
Contains the functions marginals and Loglikelihood_location at the end of the file.
CODE_ContourPlots.m
Contour plots of the fitted copula densities on the original scale: soil moisture vs
temperature (XT), rainfall vs temperature (RT) and soil moisture vs rainfall (RX).
CODE_Forecasting.m
SARIMA forecasting of rainfall, temperature and soil moisture: 1-month-ahead forecasts
on the test period and forecasts for 2026.

DATA FILES (.mat)
Ampara
rainfall_series_ampara.mat rainfall (R)
temp_series_ampara.mat temperature (T)
sm_series_ampara.mat soil moisture (X)
Kurunegala
rainfall_series_kuru.mat rainfall (R)
temp_series_kuru.mat temperature (T)
sm_series_kuru.mat soil moisture (X)
Ratnapura
rainfall_series_rath.mat rainfall (R)
temp_series_rath.mat temperature (T)
sm_series_rath.mat soil moisture (X)

Posterior chains (after burn-in)
chain_post_ampara.mat
chain_post_kuru.mat
chain_post_rath.mat

FUNCTIONS
bb7Density.m BB7 copula density
bb7PartialV.m BB7 h-function (conditional distribution)
figSettings.m figure formatting

MCMC TOOLBOX
mcmc toolbox
