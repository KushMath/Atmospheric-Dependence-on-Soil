%% CODE_MCMC_and_BSI.m  -  Ampara / Kurunegala / Ratnapura 
%% 
clear; 
clc; 
close all;
%% Choose the location
location = 'ampara';      % 'ampara'|'kuru'|'rath'
addpath('mcmc toolbox');  % mcmc toolbox folder 

%% SWITCH CASE: location settings
switch location

    case 'ampara'
        load('rainfall_series_ampara.mat');
        load('temp_series_ampara.mat');
        load('sm_series_ampara.mat');

        lag_r = 1;   lag_t = 0; % R(k-1), T(k)
        xt_rot = 90;            % (X,T): 90-deg BB7

        M.sm_dist = 'gamma';  M.a = 19.6514;  M.b = 0.0310397;     % SM gamma
        M.a_r = 1.1669;       M.b_r = 3.06513;                     % rainfall gamma
        M.w  = [0.25 0.53 0.22];                                   % temp GMM
        M.mu = [2.160233 4.073531 5.381618];
        M.sg = sqrt([0.09767001 0.89245861 0.09497745]);

        param_names = {'$\theta_{1}^{(a)}$','$\theta_{1}^{(b)}$', ...
                       '$\theta_{2}^{(a)}$','$\theta_{2}^{(b)}$', ...
                       '$\theta_{3}^{(a)}$','$\theta_{3}^{(b)}$'};
        zone = 'DZ';

    case 'kuru'
        load('rainfall_series_kuru.mat');
        load('temp_series_kuru.mat');
        load('sm_series_kuru.mat');

        lag_r = 1;   lag_t = 0;          % R(k-1), T(k)
        xt_rot = 270;                     % (X,T): 270-deg BB7

        M.sm_dist = 'gamma';  M.a = 19.1604;  M.b = 0.0321105;
        M.a_r = 1.11802;      M.b_r = 2.78776;
        M.w  = [0.65 0.35];
        M.mu = [6.9237 10.5783];
        M.sg = sqrt([0.7621 6.6237]);

        param_names = {'$\theta_{xt,IZ:K}$','$\delta_{xt,IZ:K}$', ...
                       '$\theta_{rt,IZ:K}$','$\delta_{rt,IZ:K}$', ...
                       '$\theta_{IZ:K}$',   '$\delta_{IZ:K}$'};
        zone = 'IZ';

    case 'rath'
        load('rainfall_series_rath.mat');
        load('temp_series_rath.mat');
        load('sm_series_rath.mat');

        lag_r = 1;   lag_t = 1;          % R(k-1), T(k-1)
        xt_rot = 270;                     % (X,T): 270-deg BB7

        M.sm_dist = 'normal'; M.mu_x = 0.613369;  M.sg_x = 0.121837;   % SM normal
        M.a_r = 1.43629;      M.b_r = 2.49128;
        M.w  = [0.57 0.43];
        M.mu = [6.3803 9.6168];
        M.sg = sqrt([0.722 5.4619]);

        param_names = {'$\theta_{xt,WZ:R}$','$\delta_{xt,WZ:R}$', ...
                       '$\theta_{rt,WZ:R}$','$\delta_{rt,WZ:R}$', ...
                       '$\theta_{WZ:R}$',   '$\delta_{WZ:R}$'};
        zone = 'WZ';

    otherwise
        error('location must be ''ampara'', ''kuru'' or ''rath''');
end
M.xt_rot = xt_rot;

%% Full series (Jan 1981 - Dec 2025)
rain_all = rainfall_series(:);
temp_all = temp_series(:);
sm_all   = sm_series(:);
N = numel(sm_all);

%% Lag effect
%  lag 1: R(1:end-1), SM(2:end)     T: lag 0 -> T(2:end),  lag 1 -> T(1:end-1)
L        = max(lag_r, lag_t);
rain_lag = rain_all(1+L-lag_r : N-lag_r);
temp_lag = temp_all(1+L-lag_t : N-lag_t);
sm_lag   = sm_all(1+L : N);

%% Training and testing 70%, 30%
data       = [rain_lag, temp_lag, sm_lag];
n          = size(data,1);
n_train    = round(0.7 * n);
data_train = data(1:n_train, :);
data_test  = data(n_train+1:end, :);

%% DRAM
model.ssfun  = @(params, data_train) -2*Loglikelihood_location(params, data_train, M);
model.sigma2 = 1;

params = {
    {'theta_xt', 1.2, 1,     10, NaN, Inf, 1, 0}
    {'delta_xt', 0.1, 0.001, 10, NaN, Inf, 1, 0}
    {'theta_rt', 1.2, 1,     10, NaN, Inf, 1, 0}
    {'delta_rt', 0.1, 0.001, 10, NaN, Inf, 1, 0}
    {'theta',    1.2, 1,     10, NaN, Inf, 1, 0}
    {'delta',    0.1, 0.001, 10, NaN, Inf, 1, 0}
};

options.nsimu       = 20000;
options.method      = 'dram';
options.qcov        = diag([0.1 0.1 0.1 0.1 0.1 0.1].^2);
options.verbosity   = 1;
options.adaptint    = 100;
options.drscale     = 3;
options.updatesigma = 0;

[res, chain, s2chain] = mcmc_kush(model, data_train, params, options);

%% Burn-in
burnin     = round(0.2 * size(chain,1));
chain_post = chain(burnin+1:end, :);
%save(['chain_post_' location '.mat'], 'chain_post');
%% Posterior means
theta_xt = mean(chain_post(:,1));
delta_xt = mean(chain_post(:,2));
theta_rt = mean(chain_post(:,3));
delta_rt = mean(chain_post(:,4));
theta    = mean(chain_post(:,5));
delta    = mean(chain_post(:,6));
disp([theta_xt delta_xt theta_rt delta_rt theta delta])

%% Trace plots
figure;
for i = 1:6
    subplot(6,1,i);
    plot(chain_post(:,i), 'LineWidth', 1.2);
    ylabel(param_names{i}, 'Interpreter','latex');
    if i == 6, xlabel('Iteration','Interpreter','latex'); end
    figSettings
end

%% Histograms 
figure('Color','w','Position',[100 100 1100 650]);
tiledlayout(1, 6, 'TileSpacing','compact', 'Padding','compact');
for i = 1:6
    nexttile;
    histogram(chain_post(:,i), 25, 'Normalization','probability', ...
              'FaceColor',[0.27 0.51 0.71], 'EdgeColor','w');
    xlabel(param_names{i}, 'Interpreter','latex');
    if i == 1 || i == 4, ylabel('Probability','Interpreter','latex'); end
    box on;
    figSettings
end

%% Summary statistics
fprintf('\n--------------------------------------------------------------------------\n');
fprintf('%-12s %10s %10s %10s %10s %10s %10s %20s\n', ...
    'Parameter','Min','Q1','Median','Q3','Max','Mean','50% CI');
fprintf('--------------------------------------------------------------------------\n');
for i = 1:6
    x = chain_post(:,i);
    fprintf('%-12s %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f [%7.4f,%7.4f]\n', ...
        param_names{i}, min(x), prctile(x,25), median(x), prctile(x,75), max(x), ...
        mean(x), prctile(x,25), prctile(x,75));
end
fprintf('--------------------------------------------------------------------------\n');

%% BSI vs SSI 
e  = 1e-3;
cl = @(u) min(max(u, e), 1 - e);

[u_sm, u_r, v_t] = marginals(rain_lag, temp_lag, sm_lag, M);

C_xrgt = NaN(n,1);
for i = 1:n
    if xt_rot == 90     % Ampara: 90-deg BB7 on (X,T)
        F_xgt = 1 - bb7PartialV(1 - u_sm(i), 1 - v_t(i), theta_xt, delta_xt);
    else                % Kuru, Rath: 270-deg BB7 on (X,T)
        F_xgt = bb7PartialV(u_sm(i), v_t(i), theta_xt, delta_xt);
    end
    F_rgt = bb7PartialV(u_r(i), v_t(i), theta_rt, delta_rt);   % (R,T): 270-deg BB7
    C_xrgt(i) = bb7PartialV(cl(F_xgt), cl(F_rgt), theta, delta);
end

% put back on full monthly grid (first L month(s) = NaN)
BSI = [NaN(L,1); norminv(cl(C_xrgt))];
if strcmp(M.sm_dist, 'gamma')
    SSI = norminv(cl(gamcdf(sm_all, M.a, M.b)));
else
    SSI = norminv(cl(normcdf(sm_all, M.mu_x, M.sg_x)));
end
SSI(1:L) = NaN;

%Dates
dates = datetime(1981,1,1) + calmonths(0:N-1);
dates = dates(:);
dates.Format = 'MMM yyyy';

% Plot BSI and SSI in 2 windows
starts   = [1981 2004];
ends     = [2003 2025];
series   = {BSI, SSI};
names    = {'BSI', 'SSI'};
bar_cols = {[0.53 0.81 0.92], [0.12 0.30 0.62]};
for p = 1:2
    t0  = datetime(starts(p),1,1);
    t1  = datetime(ends(p),12,1);
    idx = dates >= t0 & dates <= t1;
    figure('Color','w','Position',[100 100 1300 550]);
    for s = 1:2
        subplot(2,1,s);
        bar(dates(idx), series{s}(idx), 1, 'FaceColor', bar_cols{s}, 'EdgeColor','none');
        hold on;
        yline(0, 'k', 'LineWidth', 1);
        yline(-1.65, 'k--', 'LineWidth', 1);
        yline(-2.06, 'k--', 'LineWidth', 1);
        hold off;
        xticks(t0:calmonths(24):t1);
        xtickformat('MMM yyyy');
        xtickangle(45);
        xlim([t0 - days(30), t1 + days(30)]);
        ylim([-3.2 3.2]);
        ylabel(names{s}, 'Interpreter','latex');
        figSettings
    end
    % exportgraphics(gcf, sprintf('%s_%d.eps', zone, p), 'ContentType','vector');
end

%% RMSE between BSI and SSI for selected periods
periods = [1982 1983; 1985 1989; 1992 1996; 1999 2004; 2010 2017;2023 2023];
np     = size(periods,1);
RMSE_p = NaN(np,1);
n_p    = zeros(np,1);
lbl    = strings(np,1);
yr     = year(dates);
for p = 1:np
    idx       = yr >= periods(p,1) & yr <= periods(p,2) & ~isnan(BSI) & ~isnan(SSI);
    n_p(p)    = sum(idx);
    RMSE_p(p) = sqrt(mean((BSI(idx) - SSI(idx)).^2));
    if periods(p,1) == periods(p,2)
        lbl(p) = string(periods(p,1));
    else
        lbl(p) = sprintf('%d--%d', periods(p,1), periods(p,2));
    end
end
T = table(lbl, n_p, RMSE_p, 'VariableNames', {'Period','Months','RMSE'});
disp(T);


%% Functions

function [u_sm, u_r, v_t] = marginals(rain, temp, sm, M)
e  = 1e-3;
cl = @(u) min(max(u, e), 1 - e);
if strcmp(M.sm_dist, 'gamma')
    u_sm = gamcdf(sm, M.a, M.b);              % SM gamma
else
    u_sm = normcdf(sm, M.mu_x, M.sg_x);       % SM normal (Rath)
end
u_r = gamcdf(rain, M.a_r, M.b_r);             % rainfall gamma
u_t = zeros(size(temp));
for j = 1:numel(M.w)                          % temperature GMM (2 or 3 comps)
    u_t = u_t + M.w(j) * normcdf(temp, M.mu(j), M.sg(j));
end
u_sm = cl(u_sm);  u_r = cl(u_r);  u_t = cl(u_t);
v_t  = 1 - u_t;                               % transformed temperature T'
end

function LogLikelihood = Loglikelihood_location(params, data_train, M)
% Same as Loglikelihood_ampara / _kuru / _rath, chosen by M.xt_rot
theta_xt = params(1);  delta_xt = params(2);
theta_rt = params(3);  delta_rt = params(4);
theta    = params(5);  delta    = params(6);

[u_sm, u_r, v_t] = marginals(data_train(:,1), data_train(:,2), data_train(:,3), M);

LogLikelihood = 0;
for i = 1:size(data_train,1)
    if M.xt_rot == 90       % Ampara: 90-deg BB7 on (X,T)
        F_xgt = 1 - bb7PartialV(1-u_sm(i), 1-v_t(i), theta_xt, delta_xt);
        c_xt  = bb7Density(1-u_sm(i), 1-v_t(i), theta_xt, delta_xt);
    else                    % Kuru, Rath: 270-deg BB7 on (X,T)
        F_xgt = bb7PartialV(u_sm(i), v_t(i), theta_xt, delta_xt);
        c_xt  = bb7Density(u_sm(i),  v_t(i), theta_xt, delta_xt);
    end
    F_rgt  = bb7PartialV(u_r(i), v_t(i), theta_rt, delta_rt);    % (R,T): 270-deg BB7
    c_rt   = bb7Density(u_r(i),  v_t(i), theta_rt, delta_rt);
    F_xgt  = min(max(F_xgt, 1e-3), 1-1e-3);
    F_rgt  = min(max(F_rgt, 1e-3), 1-1e-3);
    c_xrgt = bb7Density(F_xgt, F_rgt, theta, delta);              % (XR|T): BB7
    LogLikelihood = LogLikelihood + log(c_xt) + log(c_rt) + log(c_xrgt);
end
if ~isreal(LogLikelihood) || isnan(LogLikelihood) || isinf(LogLikelihood)
    LogLikelihood = -1e10;
end
end