%% CODE_Forecasting.m  -  Ampara / Kurunegala / Ratnapura
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

        lag_r = 1;   lag_t = 0;          % R(k-1), T(k)
        loc_name = 'Ampara';

    case 'kuru'
        load('rainfall_series_kuru.mat');
        load('temp_series_kuru.mat');
        load('sm_series_kuru.mat');
        
        lag_r = 1;   lag_t = 0;          % R(k-1), T(k)
        loc_name = 'Kurunegala';

    case 'rath'
        load('rainfall_series_rath.mat');
        load('temp_series_rath.mat');
        load('sm_series_rath.mat');
        
        lag_r = 1;   lag_t = 1;          % R(k-1), T(k-1)
        loc_name = 'Ratnapura';

    otherwise
        error('location must be ''ampara'', ''kuru'' or ''rath''');
end

%% Full series (Jan 1981 - Dec 2025)
rain_all = rainfall_series(:);
temp_all = temp_series(:);
sm_all   = sm_series(:);
N = numel(sm_all);


%% Training and testing 70%, 30%
data       = [rain_all, temp_all, sm_all];
n          = size(data,1);
n_train    = round(0.7 * n);
data_train = data(1:n_train, :);
data_test  = data(n_train+1:end, :);

Ytr = {data_train(:,1), data_train(:,2), data_train(:,3)};   % R, T, X
Yte = {data_test(:,1),  data_test(:,2),  data_test(:,3)};

%% Variables, labels, colours
names  = {'R', 'T', 'X'};
ylabs  = {'Rainfall (mm/day)', ...
          '$T_{\max} - T_{\min}$ at 2 m ($^\circ$C)', ...
          'Soil Moisture (\%)'};
ylims  = {[0 30], [0 20], [0 1]};
lowerPhys = [0, -Inf, 0];
upperPhys = [Inf, Inf, 1];

cTrain = {[0.122 0.306 0.612], [0.698 0.094 0.169], [0.549 0.318 0.039]};  % deep blue | dark red | earth brown
cTest  = [0.400 0.400 0.400];      % dark grey
cFcst  = [0.106 0.620 0.467];      % teal green
cLine  = [0.200 0.200 0.200];      % dashed separators

z = 1.96;          % 95% uncertainty bounds
h = 12;            % months in 2026

%% Storage
acc = table('Size', [3 9], ...
    'VariableTypes', {'string','double','string','string','string','double','double','double','double'}, ...
    'VariableNames', {'Var','d','AR','MA','Seasonal','AIC','LjungBox_p','RMSE','MAE'});
F1    = struct();
F2026 = table();

%% SARIMA: model selection, 1-month-ahead forecast, 2026 forecast
for k = 1:3
    y_tr = Ytr{k};  y_te = Yte{k};
    y    = [y_tr; y_te];
    n    = numel(y_te);
    t    = datetime(1981,1,1) + calmonths(0:numel(y)-1)';
    t_tr = t(1:numel(y_tr));
    t_te = t(numel(y_tr)+1:end);
    tFut = t(end) + calmonths(1:h)';

    % model selection (AIC)
    [Mdl, sel] = selectSARIMA(y_tr);
    EstMdl = estimate(Mdl, y_tr, 'Display', 'off');
    fprintf('%s: d = %d | AR [%s] | MA [%s] | Seasonal %s | AIC = %.2f\n', ...
            names{k}, sel.d, num2str(sel.AR), num2str(sel.MA), sel.sname, sel.AIC);

    % residual check
    res = infer(EstMdl, y_tr);  res = res(25:end);
    [~, p_lb] = lbqtest(res, 'Lags', 24);

    % test period: 1-month-ahead
    y_fc = NaN(n,1);  mse = NaN(n,1);
    for i = 1:n
        [y_fc(i), mse(i)] = forecast(EstMdl, 1, 'Y0', [y_tr; y_te(1:i-1)]);
    end
    RMSE = sqrt(mean((y_te - y_fc).^2));        % before clipping
    MAE  = mean(abs(y_te - y_fc));
    [lo1, hi1] = bands(y_fc, mse, z, lowerPhys(k), upperPhys(k));
    y_fc = min(max(y_fc, lowerPhys(k)), upperPhys(k));   % clip after RMSE

    % 2026: refit on all data, forecast Jan-Dec
    EstAll = estimate(Mdl, y, 'Display', 'off');
    [y_new, mseF] = forecast(EstAll, h, 'Y0', y);
    [loF, hiF] = bands(y_new, mseF, z, lowerPhys(k), upperPhys(k));
    y_new = min(max(y_new, lowerPhys(k)), upperPhys(k)); % no negative rainfall / SM

    % store
    acc(k,:) = {names{k}, sel.d, num2str(sel.AR), num2str(sel.MA), sel.sname, ...
                sel.AIC, p_lb, RMSE, MAE};
    F1.(names{k}) = struct('dates', t_te, 'obs', y_te, 'fcst', y_fc, ...
                           'lo95', lo1, 'hi95', hi1, 'EstMdl', EstMdl);
    if k == 1
        tFut.Format = 'MMM yyyy';
        F2026.Month = tFut;
    end
    F2026.(names{k})           = y_new;
    F2026.([names{k} '_lo95']) = loF;
    F2026.([names{k} '_hi95']) = hiF;

    % plot
    figure('Color', 'w', 'Units', 'centimeters', 'Name', [loc_name '_' names{k}]);
    ax = gca; hold on;

    % 95% uncertainty bands (test period + 2026)
    hU = fill([t_te; flipud(t_te)], [lo1; flipud(hi1)], cFcst, ...
              'FaceAlpha', 0.18, 'EdgeColor', 'none');
    fill([tFut; flipud(tFut)], [loF; flipud(hiF)], cFcst, ...
         'FaceAlpha', 0.18, 'EdgeColor', 'none');

    % observed: train / test
    hTr = plot(t_tr, y_tr, '-o', 'Color', cTrain{k}, 'LineWidth', 1.0, ...
               'MarkerSize', 3, 'MarkerFaceColor', cTrain{k}, 'MarkerEdgeColor', cTrain{k});
    hTe = plot(t_te, y_te, '-o', 'Color', cTest, 'LineWidth', 1.0, ...
               'MarkerSize', 3, 'MarkerFaceColor', cTest, 'MarkerEdgeColor', cTest);

    % 1-month-ahead forecasts (test period + 2026)
    hF = plot(t_te, y_fc, '-o', 'Color', cFcst, 'LineWidth', 1.0, ...
              'MarkerSize', 3, 'MarkerFaceColor', cFcst, 'MarkerEdgeColor', cFcst);
    plot(tFut, y_new, '-o', 'Color', cFcst, 'LineWidth', 1.0, ...
         'MarkerSize', 3, 'MarkerFaceColor', cFcst, 'MarkerEdgeColor', cFcst);

    xlim([t(1) tFut(end)]);
    xticks(datetime(1981:5:2026, 1, 1));
    xtickformat('yyyy');  xtickangle(0);
    ylim(ylims{k});

    % dashed separators: train|test and start of 2026
    yl = ylim;
    plot([t_te(1) t_te(1)], yl, '--', 'Color', cLine, 'LineWidth', 1.0, 'HandleVisibility', 'off');
    plot([tFut(1) tFut(1)], yl, '--', 'Color', cLine, 'LineWidth', 1.0, 'HandleVisibility', 'off');

    ylabel(ylabs{k}, 'Interpreter', 'latex');
    xlabel('Year-Month', 'Interpreter', 'latex');
    box on;
    figSettings
    ax.XAxis.MinorTick = 'off';

    % legend above the axes
    legend([hTr hTe hF hU], {'Train', 'Test', '1-month-ahead forecast', '95\% CI'}, ...
           'Interpreter', 'latex', 'Location', 'northoutside', ...
           'Orientation', 'horizontal', 'NumColumns', 4, 'Box', 'off');
end

disp(acc);
disp('Forecast 2026:');  disp(F2026);

%% Save .mat files
%save([loc_name '_1month_ahead.mat'],  'F1', 'acc', 'F2026');
%save([loc_name '_forecast_2026.mat'], 'F2026');
%fprintf('Saved: %s_1month_ahead.mat, %s_forecast_2026.mat\n', loc_name, loc_name);


%% Functions

function [Mdl, sel] = selectSARIMA(y_tr)
% SARIMA selection by AIC
h_adf = adftest(y_tr);
h_pp  = pptest(y_tr);
d = 0;
if h_adf == 0 || h_pp == 0
    d = 1;
end

w = y_tr(13:end) - y_tr(1:end-12);
[acf, ~, bA]  = autocorr(w, 'NumLags', 3);
[pacf, ~, bP] = parcorr(w, 'NumLags', 3);
maLags = find(abs(acf(2:end))  > bA(1))';
arLags = find(abs(pacf(2:end)) > bP(1))';

sar = {[], 12, 12};  sma = {12, [], 12};
sname = {'(0,1,1)', '(1,1,0)', '(1,1,1)'};
bestAIC = inf;
for k = 1:3
for i = 0:2^length(arLags)-1
for j = 0:2^length(maLags)-1
    ar = arLags(bitget(i, 1:length(arLags)) == 1);
    ma = maLags(bitget(j, 1:length(maLags)) == 1);
    M  = arima('ARLags',ar,'D',d,'MALags',ma,'Seasonality',12, ...
               'SARLags',sar{k},'SMALags',sma{k});
    [~, ~, logL] = estimate(M, y_tr, 'Display', 'off');
    aic = aicbic(logL, length(ar) + length(ma) + length(sar{k}) + length(sma{k}) + 2);
    if aic < bestAIC
        bestAIC = aic;  bestAR = ar;  bestMA = ma;  bestK = k;
    end
end
end
end
Mdl = arima('ARLags',bestAR,'D',d,'MALags',bestMA,'Seasonality',12, ...
            'SARLags',sar{bestK},'SMALags',sma{bestK});
sel = struct('d', d, 'AR', bestAR, 'MA', bestMA, ...
             'sname', [sname{bestK} '_12'], 'AIC', bestAIC);
end

function [lo, hi] = bands(f, mse, z, lb, ub)
lo = min(max(f - z*sqrt(mse), lb), ub);
hi = min(max(f + z*sqrt(mse), lb), ub);
end