function results = checkEstimationBias(nReps)
% results = checkEstimationBias(nReps)
%
% Runs the YesNoTSD staircase simulation and MLE fitting nReps times to
% assess estimation bias in A, b, and d-prime thresholds.
%
% Input:
%   nReps  - number of simulation replications (default: 50)
%
% Output:
%   results - struct with fields:
%     A_hat, b_hat          - (nReps x 1) parameter estimates
%     I_hat                 - (nReps x nDPrime) threshold estimates
%     A_true, b_true        - true parameter values
%     I_true                - (1 x nDPrime) true threshold intensities
%     dPrimeTargets         - (1 x nDPrime) target d-prime values
%
% Example:
%   results = checkEstimationBias(50);
%
% History:
%   2026-04-08  DHB, HES, ClaudeAI  wrote it.

if nargin < 1
    nReps = 50;
end

%% ---- True parameters (must match YesNoTSDSimulation.m) -------------------

A_true = 1.0;
b_true = 0.8;

nResp           = 6;
Icrit_true      = linspace(0.3, 2, nResp-1);
beta_true       = responseFunction(Icrit_true, [A_true, b_true]);
nCrit           = numel(beta_true);
nStaircaseRespondNo = 3;
dPrimeTargets   = [0.75, 1, 1.25];
nDPrime         = numel(dPrimeTargets);

% True threshold intensities: invert R = A*I^b  =>  I = (d'/A)^(1/b)
I_true = (dPrimeTargets / A_true).^(1 / b_true);

%% ---- Staircase / simulation settings ------------------------------------

pCatch              = 0.20;
nUp                 = 1;
nDown               = 2;
nStaircases         = 1;
nTrialsPerStaircase = 150;
nTrials             = nTrialsPerStaircase * nStaircases;
I0                  = 2.0;
Imin                = 0.01;
Imax                = 3.0;
stepSizes           = log(1.15) * [2, 1];

cMax  = 3.5;
opts  = optimoptions('fmincon', 'Display', 'off', ...
    'MaxIterations', 5000, 'MaxFunctionEvaluations', 50000);

%% ---- Storage -------------------------------------------------------------

A_hat_all = zeros(nReps, 1);
b_hat_all = zeros(nReps, 1);
I_hat_all = zeros(nReps, nDPrime);

%% ---- Replication loop ----------------------------------------------------

fprintf('Running %d replications...\n', nReps);
for rep = 1:nReps

    rng(rep);

    % --- Staircase simulation ---
    sc = Staircase('standard', log(I0), ...
        'StepSizes', stepSizes, ...
        'NUp',       nDown, ...
        'NDown',     nUp,   ...
        'MaxValue',  log(Imax), ...
        'MinValue',  log(Imin));

    I = zeros(nTrials, 1);
    y = zeros(nTrials, 1);

    for t = 1:nTrials
        if rand < pCatch
            I(t) = 0;
            y(t) = sum(randn() > beta_true) + 1;
        else
            I(t) = exp(getCurrentValue(sc));
            x    = responseFunction(I(t), [A_true, b_true]) + randn();
            y(t) = sum(x > beta_true) + 1;
            sc   = updateForTrial(sc, log(I(t)), double(y(t) > nStaircaseRespondNo));
        end
    end

    % --- MLE fit ---
    c_init   = linspace(0.5, 3.0, nCrit);
    theta0   = [log(1); log(1); c_init(1); log(diff(c_init))'];

    thetaHat = fmincon(@(th) negLogLik(th, I, y), theta0, ...
        [], [], [], [], [], [], @(th) critBounds(th, cMax), opts);

    A_hat = exp(thetaHat(1));
    b_hat = exp(thetaHat(2));

    A_hat_all(rep) = A_hat;
    b_hat_all(rep) = b_hat;

    % Threshold estimates: invert fitted model
    I_hat_all(rep, :) = (dPrimeTargets / A_hat).^(1 / b_hat);

    fprintf('  Rep %3d/%d:  A_hat=%.3f  b_hat=%.3f\n', rep, nReps, A_hat, b_hat);
end

%% ---- Summary table -------------------------------------------------------

fprintf('\n=== Estimation bias summary (%d reps) ===\n\n', nReps);
fprintf('Parameter  True    Mean     Bias    SD\n');
fprintf('-------------------------------------------\n');
fprintf('A          %.3f   %.3f    %+.3f   %.3f\n', ...
    A_true, mean(A_hat_all), mean(A_hat_all) - A_true, std(A_hat_all));
fprintf('b          %.3f   %.3f    %+.3f   %.3f\n', ...
    b_true, mean(b_hat_all), mean(b_hat_all) - b_true, std(b_hat_all));

fprintf('\nThreshold intensities (I_hat vs I_true):\n');
fprintf('d-prime   I_true   Mean I_hat   Bias      SD\n');
fprintf('------------------------------------------------\n');
for k = 1:nDPrime
    fprintf('  %.2f     %.3f    %.3f       %+.3f     %.3f\n', ...
        dPrimeTargets(k), I_true(k), mean(I_hat_all(:,k)), ...
        mean(I_hat_all(:,k)) - I_true(k), std(I_hat_all(:,k)));
end

%% ---- Figure --------------------------------------------------------------

figure('Color', 'w');
tiledlayout(2, 2);

nexttile;
histogram(A_hat_all, 12); hold on;
xline(A_true, 'r-', 'LineWidth', 2);
xline(mean(A_hat_all), 'b--', 'LineWidth', 1.5);
xlabel('A_{hat}');  title(sprintf('A  (true=%.2f, mean=%.2f)', A_true, mean(A_hat_all)));

nexttile;
histogram(b_hat_all, 12); hold on;
xline(b_true, 'r-', 'LineWidth', 2);
xline(mean(b_hat_all), 'b--', 'LineWidth', 1.5);
xlabel('b_{hat}');  title(sprintf('b  (true=%.2f, mean=%.2f)', b_true, mean(b_hat_all)));

nexttile([1 2]);   % span both columns in the second row
colors = lines(nDPrime);
hold on;
for k = 1:nDPrime
    histogram(I_hat_all(:,k), 10, 'FaceColor', colors(k,:), 'FaceAlpha', 0.5, ...
        'DisplayName', sprintf("d'=%.2f", dPrimeTargets(k)));
    xline(I_true(k),            '-',  'Color', colors(k,:), 'LineWidth', 2,   'HandleVisibility', 'off');
    xline(mean(I_hat_all(:,k)), '--', 'Color', colors(k,:), 'LineWidth', 1.5, 'HandleVisibility', 'off');
end
xlabel('I_{hat}');
title('Threshold estimates  (solid=true, dashed=mean)');
legend('Location', 'northeast');

%% ---- Pack results --------------------------------------------------------

results.A_hat         = A_hat_all;
results.b_hat         = b_hat_all;
results.I_hat         = I_hat_all;
results.A_true        = A_true;
results.b_true        = b_true;
results.I_true        = I_true;
results.dPrimeTargets = dPrimeTargets;

end

%% ======== Local functions =================================================

function [c, ceq] = critBounds(theta, cMax)
    beta = unpackCrit(theta(3:end));
    c    = beta(:) - cMax;
    ceq  = [];
end

function nll = negLogLik(theta, I, y)
    A    = exp(theta(1));
    b    = exp(theta(2));
    beta = unpackCrit(theta(3:end));
    R    = responseFunction(I, [A, b]);
    bounds = [-Inf, beta, Inf];
    lo   = bounds(y);
    hi   = bounds(y + 1);
    p    = max(Phi(hi(:) - R) - Phi(lo(:) - R), realmin);
    nll  = -sum(log(p));
    if ~isfinite(nll);  nll = realmax;  end
end

function beta = unpackCrit(v)
    beta    = zeros(1, numel(v));
    beta(1) = v(1);
    for k = 2:numel(v)
        beta(k) = beta(k-1) + exp(v(k));
    end
end

function p = Phi(z)
    p = 0.5 * erfc(-z ./ sqrt(2));
end

function R = responseFunction(I, theta)
    A = theta(1);
    b = theta(2);
    R = A .* (I .^ b);
end
