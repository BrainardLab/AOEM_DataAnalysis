function results = checkEstimationBias(nReps)
% results = checkEstimationBias(nReps)
%
% Runs the YesNoTSD staircase simulation and MLE fitting nReps times,
% comparing two parameterisations:
%
%   AB param   : theta = [log(A), log(b), criteria...]
%                R(I) = A * I^b
%
%   Thresh param : theta = [log(I_thresh), log(b), criteria...]
%                R(I) = (I / I_thresh)^b
%                where I_thresh is the intensity producing d' = dPrimeRef = 1.
%                Anchors one parameter at the staircase intensity, reducing
%                correlation between the two free parameters.
%
% Input:
%   nReps  - number of simulation replications (default: 50)
%
% Output:
%   results - struct with fields for both parameterisations:
%     AB:     A_hat, b_hat, I_hat       (nReps x 1 or nReps x nDPrime)
%     Thresh: Ithresh_hat, b_hat, I_hat
%     true values and dPrimeTargets
%
% Example:
%   results = checkEstimationBias(50);
%
% History:
%   2026-04-08  DHB, HES, ClaudeAI  wrote it.
%   2026-04-08  DHB, HES, ClaudeAI  added threshold parameterisation.

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
dPrimeRef       = 1.0;   % reference d' for threshold parameterisation

% True threshold intensities: invert R = A*I^b  =>  I = (d'/A)^(1/b)
I_true       = (dPrimeTargets / A_true).^(1 / b_true);
Ithresh_true = (dPrimeRef    / A_true).^(1 / b_true);   % true I at d'=dPrimeRef

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

% AB parameterisation
A_hat_all     = zeros(nReps, 1);
b_hat_ab_all  = zeros(nReps, 1);
I_hat_ab_all  = zeros(nReps, nDPrime);

% Threshold parameterisation
Ithresh_hat_all  = zeros(nReps, 1);
b_hat_th_all     = zeros(nReps, 1);
I_hat_th_all     = zeros(nReps, nDPrime);

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

    c_init = linspace(0.5, 3.0, nCrit);

    % --- AB parameterisation fit ---
    theta0_ab  = [log(1); log(1); c_init(1); log(diff(c_init))'];
    thetaHat   = fmincon(@(th) negLogLikAB(th, I, y), theta0_ab, ...
        [], [], [], [], [], [], @(th) critBounds(th, cMax, 3), opts);

    A_hat              = exp(thetaHat(1));
    b_hat_ab           = exp(thetaHat(2));
    A_hat_all(rep)     = A_hat;
    b_hat_ab_all(rep)  = b_hat_ab;
    I_hat_ab_all(rep,:) = (dPrimeTargets / A_hat).^(1 / b_hat_ab);

    % --- Threshold parameterisation fit ---
    % R(I) = (I / I_thresh)^b,  so d' at I_thresh = dPrimeRef = 1
    theta0_th  = [log(Ithresh_true); log(1); c_init(1); log(diff(c_init))'];
    thetaHat_th = fmincon(@(th) negLogLikThresh(th, I, y, dPrimeRef), theta0_th, ...
        [], [], [], [], [], [], @(th) critBounds(th, cMax, 3), opts);

    Ithresh_hat           = exp(thetaHat_th(1));
    b_hat_th              = exp(thetaHat_th(2));
    Ithresh_hat_all(rep)  = Ithresh_hat;
    b_hat_th_all(rep)     = b_hat_th;
    % Invert thresh model: I = I_thresh * (d' / dPrimeRef)^(1/b)
    I_hat_th_all(rep,:)   = Ithresh_hat .* (dPrimeTargets / dPrimeRef).^(1 / b_hat_th);

    fprintf('  Rep %3d/%d:  AB: A=%.3f b=%.3f  |  Thresh: Ith=%.3f b=%.3f\n', ...
        rep, nReps, A_hat, b_hat_ab, Ithresh_hat, b_hat_th);
end

%% ---- Summary tables ------------------------------------------------------

fprintf('\n=== AB parameterisation (%d reps) ===\n', nReps);
fprintf('Parameter  True    Mean     Bias    SD\n');
fprintf('-------------------------------------------\n');
fprintf('A          %.3f   %.3f    %+.3f   %.3f\n', ...
    A_true, mean(A_hat_all), mean(A_hat_all) - A_true, std(A_hat_all));
fprintf('b          %.3f   %.3f    %+.3f   %.3f\n', ...
    b_true, mean(b_hat_ab_all), mean(b_hat_ab_all) - b_true, std(b_hat_ab_all));

fprintf('\n=== Threshold parameterisation (%d reps) ===\n', nReps);
fprintf('Parameter    True    Mean     Bias    SD\n');
fprintf('-------------------------------------------\n');
fprintf('I_thresh     %.3f   %.3f    %+.3f   %.3f\n', ...
    Ithresh_true, mean(Ithresh_hat_all), mean(Ithresh_hat_all) - Ithresh_true, std(Ithresh_hat_all));
fprintf('b            %.3f   %.3f    %+.3f   %.3f\n', ...
    b_true, mean(b_hat_th_all), mean(b_hat_th_all) - b_true, std(b_hat_th_all));

fprintf('\n=== Threshold intensity estimates: I vs log(I) ===\n');
fprintf('\n  %-12s  %6s  %10s  %10s  %6s  %10s  %10s  %6s\n', ...
    '', 'I_true', 'AB mean', 'AB bias', 'AB SD', 'Th mean', 'Th bias', 'Th SD');
fprintf('  %s\n', repmat('-',1,75));
for k = 1:nDPrime
    fprintf('  d''=%.2f (I)   %6.3f  %10.3f  %10+.3f  %6.3f  %10.3f  %10+.3f  %6.3f\n', ...
        dPrimeTargets(k), I_true(k), ...
        mean(I_hat_ab_all(:,k)), mean(I_hat_ab_all(:,k)) - I_true(k), std(I_hat_ab_all(:,k)), ...
        mean(I_hat_th_all(:,k)), mean(I_hat_th_all(:,k)) - I_true(k), std(I_hat_th_all(:,k)));
end
fprintf('\n  %-12s  %6s  %10s  %10s  %6s  %10s  %10s  %6s\n', ...
    '', 'logI_true', 'AB mean', 'AB bias', 'AB SD', 'Th mean', 'Th bias', 'Th SD');
fprintf('  %s\n', repmat('-',1,75));
for k = 1:nDPrime
    fprintf('  d''=%.2f (logI) %6.3f  %10.3f  %10+.3f  %6.3f  %10.3f  %10+.3f  %6.3f\n', ...
        dPrimeTargets(k), log(I_true(k)), ...
        mean(log(I_hat_ab_all(:,k))), mean(log(I_hat_ab_all(:,k))) - log(I_true(k)), std(log(I_hat_ab_all(:,k))), ...
        mean(log(I_hat_th_all(:,k))), mean(log(I_hat_th_all(:,k))) - log(I_true(k)), std(log(I_hat_th_all(:,k))));
end

%% ---- Figure --------------------------------------------------------------
% Row 1: b_hat from both parameterisations (b is common to both)
%        A_hat (AB) | I_thresh_hat (Thresh)
% Row 2: I_hat comparison — AB (left) vs Thresh (right)

colors = lines(nDPrime);

figure('Color', 'w');
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% --- [1,1] b_hat: both parameterisations overlaid ---
nexttile;
histogram(b_hat_ab_all, 12, 'FaceColor', [0.2 0.5 0.8], 'FaceAlpha', 0.5, ...
    'DisplayName', 'AB param');
hold on;
histogram(b_hat_th_all, 12, 'FaceColor', [0.9 0.4 0.1], 'FaceAlpha', 0.5, ...
    'DisplayName', 'Thresh param');
xline(b_true, 'k-', 'LineWidth', 2, 'HandleVisibility', 'off');
xlabel('b_{hat}');
title(sprintf('b  (true=%.2f)', b_true));
legend('Location', 'northeast');

% --- [1,2] A_hat (AB) and I_thresh_hat (Thresh) ---
nexttile;
histogram(A_hat_all, 12, 'FaceColor', [0.2 0.5 0.8], 'FaceAlpha', 0.6, ...
    'DisplayName', 'A_{hat} (AB)');
hold on;
histogram(Ithresh_hat_all, 12, 'FaceColor', [0.9 0.4 0.1], 'FaceAlpha', 0.6, ...
    'DisplayName', 'I_{thresh} (Thresh)');
xline(A_true,       'b-', 'LineWidth', 2, 'HandleVisibility', 'off');
xline(Ithresh_true, 'r-', 'LineWidth', 2, 'HandleVisibility', 'off');
xlabel('Estimate');
title(sprintf('A_{hat} (true=%.2f)  |  I_{thresh} (true=%.2f)', A_true, Ithresh_true));
legend('Location', 'northeast');

% --- [2,1] I_hat from AB parameterisation ---
nexttile;
hold on;
for k = 1:nDPrime
    histogram(I_hat_ab_all(:,k), 10, 'FaceColor', colors(k,:), 'FaceAlpha', 0.6, ...
        'DisplayName', sprintf("d'=%.2f", dPrimeTargets(k)));
    xline(I_true(k),               '-',  'Color', colors(k,:), 'LineWidth', 2,   'HandleVisibility', 'off');
    xline(mean(I_hat_ab_all(:,k)), '--', 'Color', colors(k,:), 'LineWidth', 1.5, 'HandleVisibility', 'off');
end
xlabel('I_{hat}');
title('AB param  (solid=true, dashed=mean)');
legend('Location', 'northeast');

% --- [2,2] I_hat from threshold parameterisation ---
nexttile;
hold on;
for k = 1:nDPrime
    histogram(I_hat_th_all(:,k), 10, 'FaceColor', colors(k,:), 'FaceAlpha', 0.6, ...
        'DisplayName', sprintf("d'=%.2f", dPrimeTargets(k)));
    xline(I_true(k),               '-',  'Color', colors(k,:), 'LineWidth', 2,   'HandleVisibility', 'off');
    xline(mean(I_hat_th_all(:,k)), '--', 'Color', colors(k,:), 'LineWidth', 1.5, 'HandleVisibility', 'off');
end
xlabel('I_{hat}');
title('Thresh param  (solid=true, dashed=mean)');
legend('Location', 'northeast');

%% ---- Pack results --------------------------------------------------------

results.AB.A_hat         = A_hat_all;
results.AB.b_hat         = b_hat_ab_all;
results.AB.I_hat         = I_hat_ab_all;

results.Thresh.Ithresh_hat = Ithresh_hat_all;
results.Thresh.b_hat       = b_hat_th_all;
results.Thresh.I_hat       = I_hat_th_all;

results.A_true        = A_true;
results.b_true        = b_true;
results.Ithresh_true  = Ithresh_true;
results.I_true        = I_true;
results.dPrimeTargets = dPrimeTargets;
results.dPrimeRef     = dPrimeRef;

end

%% ======== Local functions =================================================

function [c, ceq] = critBounds(theta, cMax, critStart)
% critStart = index of first criterion parameter in theta (3 for both models)
    beta = unpackCrit(theta(critStart:end));
    c    = beta(:) - cMax;
    ceq  = [];
end

function nll = negLogLikAB(theta, I, y)
% AB parameterisation: R(I) = A * I^b
    A    = exp(theta(1));
    b    = exp(theta(2));
    beta = unpackCrit(theta(3:end));
    R    = A .* (max(I, 0) .^ b);
    R(I == 0) = 0;
    nll  = ratingNLL(R, y, beta);
end

function nll = negLogLikThresh(theta, I, y, dPrimeRef)
% Threshold parameterisation: R(I) = dPrimeRef * (I / I_thresh)^b
% I_thresh is the intensity producing d' = dPrimeRef.
    I_thresh = exp(theta(1));
    b        = exp(theta(2));
    beta     = unpackCrit(theta(3:end));
    R        = dPrimeRef .* (max(I, 0) ./ I_thresh) .^ b;
    R(I == 0) = 0;
    nll      = ratingNLL(R, y, beta);
end

function nll = ratingNLL(R, y, beta)
% Shared negative log-likelihood for ordinal rating data.
    bounds = [-Inf, beta, Inf];
    lo     = bounds(y);
    hi     = bounds(y + 1);
    p      = max(Phi(hi(:) - R) - Phi(lo(:) - R), realmin);
    nll    = -sum(log(p));
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
