%% CODE_ContourPlots.m  -  Ampara / Kurunegala / Ratnapura
%%
clc;
close all;
clear;
%% Choose the location
location = 'rath';      % 'ampara'|'kuru'|'rath'

%% SWITCH CASE: location settings
switch location

    case 'ampara'
        load('rainfall_series_ampara.mat');
        load('temp_series_ampara.mat');
        load('sm_series_ampara.mat');
        load('chain_post_ampara.mat');

        lag_r = 1;   lag_t = 0;          % R(k-1), T(k)
        xt_rot = 90;                      % (X,T): 90-deg BB7

        % SM gamma, rainfall gamma, temp GMM (3 components)
        Fx = @(v) gamcdf(v, 19.6514, 0.0310397);   fx = @(v) gampdf(v, 19.6514, 0.0310397);
        Fr = @(v) gamcdf(v, 1.1669, 3.06513);      fr = @(v) gampdf(v, 1.1669, 3.06513);
        w  = [0.25 0.53 0.22];
        mu = [2.160233 4.073531 5.381618];
        sg = sqrt([0.09767001 0.89245861 0.09497745]);
        Ft = @(v) w(1)*normcdf(v,mu(1),sg(1)) + w(2)*normcdf(v,mu(2),sg(2)) + w(3)*normcdf(v,mu(3),sg(3));
        ft = @(v) w(1)*normpdf(v,mu(1),sg(1)) + w(2)*normpdf(v,mu(2),sg(2)) + w(3)*normpdf(v,mu(3),sg(3));

    case 'kuru'
        load('rainfall_series_kuru.mat');
        load('temp_series_kuru.mat');
        load('sm_series_kuru.mat');
        load('chain_post_kuru.mat');

        lag_r = 1;   lag_t = 0;          % R(k-1), T(k)
        xt_rot = 270;                     % (X,T): 270-deg BB7

        % SM gamma, rainfall gamma, temp GMM (2 components)
        Fx = @(v) gamcdf(v, 19.1604, 0.0321105);   fx = @(v) gampdf(v, 19.1604, 0.0321105);
        Fr = @(v) gamcdf(v, 1.11802, 2.78776);     fr = @(v) gampdf(v, 1.11802, 2.78776);
        w  = [0.65 0.35];
        mu = [6.9237 10.5783];
        sg = sqrt([0.7621 6.6237]);
        Ft = @(v) w(1)*normcdf(v,mu(1),sg(1)) + w(2)*normcdf(v,mu(2),sg(2));
        ft = @(v) w(1)*normpdf(v,mu(1),sg(1)) + w(2)*normpdf(v,mu(2),sg(2));

    case 'rath'
        load('rainfall_series_rath.mat');
        load('temp_series_rath.mat');
        load('sm_series_rath.mat');
        load('chain_post_rath.mat');

        lag_r = 1;   lag_t = 1;          % R(k-1), T(k-1)
        xt_rot = 270;                     % (X,T): 270-deg BB7

        % SM normal, rainfall gamma, temp GMM (2 components)
        Fx = @(v) normcdf(v, 0.613369, 0.121837);  fx = @(v) normpdf(v, 0.613369, 0.121837);
        Fr = @(v) gamcdf(v, 1.43629, 2.49128);     fr = @(v) gampdf(v, 1.43629, 2.49128);
        w  = [0.57 0.43];
        mu = [6.3803 9.6168];
        sg = sqrt([0.722 5.4619]);
        Ft = @(v) w(1)*normcdf(v,mu(1),sg(1)) + w(2)*normcdf(v,mu(2),sg(2));
        ft = @(v) w(1)*normpdf(v,mu(1),sg(1)) + w(2)*normpdf(v,mu(2),sg(2));

    otherwise
        error('location must be ''ampara'', ''kuru'' or ''rath''');
end

%% Posterior means (6-parameter model)
theta_xt = mean(chain_post(:,1));  delta_xt = mean(chain_post(:,2));   % XT
theta_rt = mean(chain_post(:,3));  delta_rt = mean(chain_post(:,4));   % RT
theta    = mean(chain_post(:,5));  delta    = mean(chain_post(:,6));   % RX

%% Whole series with the same lags as the fitted model
rain_all = rainfall_series(:);
temp_all = temp_series(:);
sm_all   = sm_series(:);
N = numel(sm_all);

L   = max(lag_r, lag_t);
r_d = rain_all(1+L-lag_r : N-lag_r);
t_d = temp_all(1+L-lag_t : N-lag_t);
x_d = sm_all(1+L : N);
ok  = ~isnan(r_d) & ~isnan(t_d) & ~isnan(x_d);
r_d = r_d(ok);  t_d = t_d(ok);  x_d = x_d(ok);

cl = @(u) min(max(u, 1e-3), 1 - 1e-3);       % keep probabilities away from 0 and 1
E2 = @(f, A, B) arrayfun(f, A, B);

%% Copula densities, written in u_t (v_t = 1 - u_t)
% XT: Ampara 90-deg BB7  c(1-u_x, u_t) | Kuru, Rath 270-deg BB7  c(u_x, 1-u_t)
% RT: 270-deg BB7  c(u_r, 1-u_t)
if xt_rot == 90
    cXT_f = @(ux, ut) E2(@(p,q) bb7Density(1-p, q, theta_xt, delta_xt), ux, ut);
else
    cXT_f = @(ux, ut) E2(@(p,q) bb7Density(p, 1-q, theta_xt, delta_xt), ux, ut);
end
cRT_f = @(ur, ut) E2(@(p,q) bb7Density(p, 1-q, theta_rt, delta_rt), ur, ut);

%% Original-scale grids and joint densities
n  = 120;
xg = linspace(min(x_d), max(x_d), n);
rg = linspace(max(1e-4, min(r_d)), max(r_d), n);
tg = linspace(min(t_d), max(t_d), n);

[GT, GX]  = meshgrid(tg, xg);                % XT panel: x = temperature, y = soil moisture
fXT = cXT_f(cl(Fx(GX)), cl(Ft(GT))) .* fx(GX) .* ft(GT);

[GT2, GR] = meshgrid(tg, rg);                % RT panel: x = temperature, y = rainfall
fRT = cRT_f(cl(Fr(GR)), cl(Ft(GT2))) .* fr(GR) .* ft(GT2);

[GR3, GX3] = meshgrid(rg, xg);               % RX panel: x = rainfall, y = soil moisture
uX  = cl(Fx(GX3));
uR  = cl(Fr(GR3));
cRX = E2(@(p,q) bb7Density(p, q, theta, delta), uX, uR);
fXR = cRX .* fx(GX3) .* fr(GR3);

%% Plot
sCol = [0.4 0.4 0.4];  cCol = [0.85 0.37 0.01];
tLab = '$T_{\max} - T_{\min}$ at 2 m ($^\circ$C)';
figure('Color', 'w', 'Position', [40 100 1800 480]);
tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% XT  (x = T, y = X)
nexttile;
scatter(t_d, x_d, 28, sCol, 'filled', 'MarkerFaceAlpha', 0.75, 'MarkerEdgeColor', 'w');
hold on; contour(GT, GX, fXT, 15, 'LineColor', cCol, 'LineWidth', 1.4); hold off;
axis square;
xlabel(tLab, 'Interpreter', 'latex'); ylabel('Soil Moisture (\%)', 'Interpreter', 'latex');
xlim([0,16]);
ylim([0.15,1]);
box on; figSettings

% RT  (x = T, y = R)
nexttile;
scatter(t_d, r_d, 28, sCol, 'filled', 'MarkerFaceAlpha', 0.75, 'MarkerEdgeColor', 'w');
hold on; contour(GT2, GR, fRT, 15, 'LineColor', cCol, 'LineWidth', 1.4); hold off;
axis square;
xlabel(tLab, 'Interpreter', 'latex'); ylabel('Rainfall (mm/day)', 'Interpreter', 'latex');
xlim([0,16]);
ylim([0,25]);
box on; figSettings

% RX: BB7  (x = R, y = X)
nexttile;
scatter(r_d, x_d, 28, sCol, 'filled', 'MarkerFaceAlpha', 0.75, 'MarkerEdgeColor', 'w');
hold on; contour(GR3, GX3, fXR, 25, 'LineColor', cCol, 'LineWidth', 1.4); hold off;
axis square;
xlabel('Rainfall (mm/day)', 'Interpreter', 'latex'); ylabel('Soil Moisture (\%)', 'Interpreter', 'latex');
xlim([0,25]);
ylim([0.15,1]);
box on; figSettings