function results = checkEstimationBias(nReps, b_noise_sd, staircaseType)
% results = checkEstimationBias(nReps, b_noise_sd, staircaseType)
%
% Runs the YesNoTSD staircase simulation and MLE fitting nReps times,
% comparing two fitting parameterisations:
%
%   AB param      : R(I) = A * I^b   (both A and b free)
%   Fixed-b param : R(I) = (I/I_thresh)^b_fixed
%                   b fixed at b_true + N(0, b_noise_sd); only I_thresh free.
%
% Staircase type is controlled by staircaseType:
%   'standard' : 1-up/2-down staircase in log(I) space (default)
%   'quest'    : 3 interleaved QUESTs targeting P(yes) = [0.6, 0.75, 0.9]
%                for broader intensity coverage
%
% Inputs:
%   nReps         - number of replications (default: 50)
%   b_noise_sd    - SD of calibration error on b (default: 0.1)
%   staircaseType - 'standard' or 'quest' (default: 'standard')
%
% Output:
%   results - struct with fields for both fitting parameterisations
%
% Examples:
%   results_std   = checkEstimationBias(50, 0.1, 'standard');
%   results_quest = checkEstimationBias(50, 0.1, 'quest');
%
% History:
%   2026-04-08  DHB, HES, ClaudeAI  wrote it.
%   2026-04-08  DHB, HES, ClaudeAI  added fixed-b parameterisation.
%   2026-04-08  DHB, HES, ClaudeAI  added staircaseType parameter.

if nargin < 1;  nReps         = 50;         end
if nargin < 2;  b_noise_sd    = 0.1;        end
if nargin < 3;  staircaseType = 'standard'; end

%% ---- True parameters (must match YesNoTSDSimulation.m) -------------------

A_true = 1.0;
b_true = 0.8;

nResp               = 6;
Icrit_true          = linspace(0.3, 2, nResp-1);
beta_true           = responseFunction(Icrit_true, [A_true, b_true]);
nCrit               = numel(beta_true);
nStaircaseRespondNo = 3;
dPrimeTargets       = [0.75, 1, 1.25];
nDPrime             = numel(dPrimeTargets);
dPrimeRef           = 1.0;   % reference d' for I_thresh definition

% True threshold intensities
I_true       = (dPrimeTargets / A_true).^(1 / b_true);
Ithresh_true = (dPrimeRef    / A_true).^(1 / b_true);

%% ---- Staircase / simulation settings ------------------------------------

pCatch = 0.20;
I0     = 2.0;
Imin   = 0.01;
Imax   = 3.0;

% Standard staircase settings
nUp       = 1;
nDown     = 2;
stepSizes = log(1.15) * [2, 1];

% QUEST settings — 3 interleaved QUESTs targeting different P(yes) levels
questTargetProbs = [0.6, 0.75, 0.9];
questBeta        = 3.5;
questDelta       = 0.01;
questGamma       = 0.0;
questPriorSD     = 10;

% Total signal trials kept equal across staircase types:
%   standard: 1 staircase x 150 trials = 150
%   quest:    3 staircases x 50 trials = 150
if strcmp(staircaseType, 'standard')
    nStaircases         = numel(nDown);
    nTrialsPerStaircase = 150;
else
    nStaircases         = numel(questTargetProbs);
    nTrialsPerStaircase = 50;
end
nTrials = nTrialsPerStaircase * nStaircases;

cMax = 3.5;
opts = optimoptions('fmincon', 'Display', 'off', ...
    'MaxIterations', 5000, 'MaxFunctionEvaluations', 50000);

%% ---- Storage -------------------------------------------------------------

% AB parameterisation (both A and b free)
A_hat_all    = zeros(nReps, 1);
b_hat_ab_all = zeros(nReps, 1);
I_hat_ab_all = zeros(nReps, nDPrime);

% Fixed-b parameterisation (b fixed at b_true + noise, I_thresh free)
b_fixed_all      = zeros(nReps, 1);
Ithresh_hat_all  = zeros(nReps, 1);
I_hat_fb_all     = zeros(nReps, nDPrime);

%% ---- Replication loop ----------------------------------------------------

fprintf('Running %d replications (staircaseType=%s, b_noise_sd=%.2f)...\n', ...
    nReps, staircaseType, b_noise_sd);
for rep = 1:nReps

    rng(rep);

    % --- Initialise staircases ---
    staircases = cell(nStaircases, 1);
    for s = 1:nStaircases
        if strcmp(staircaseType, 'standard')
            staircases{s} = Staircase('standard', log(I0), ...
                'StepSizes', stepSizes, ...
                'NUp',       nDown,     ...
                'NDown',     nUp,       ...
                'MaxValue',  log(Imax), ...
                'MinValue',  log(Imin));
        else
            staircases{s} = Staircase('quest', I0, ...
                'Beta',            questBeta,           ...
                'Delta',           questDelta,          ...
                'Gamma',           questGamma,          ...
                'TargetThreshold', questTargetProbs(s), ...
                'PriorSD',         questPriorSD,        ...
                'MaxValue',        Imax,                ...
                'MinValue',        Imin);
        end
    end

    % --- Simulate trials ---
    I = zeros(nTrials, 1);
    y = zeros(nTrials, 1);
    signalCount = 0;

    for t = 1:nTrials
        if rand < pCatch
            I(t) = 0;
            y(t) = sum(randn() > beta_true) + 1;
        else
            signalCount = signalCount + 1;
            s = mod(signalCount - 1, nStaircases) + 1;
            if strcmp(staircaseType, 'standard')
                I(t) = exp(getCurrentValue(staircases{s}));
            else
                I(t) = getCurrentValue(staircases{s});
            end
            x    = responseFunction(I(t), [A_true, b_true]) + randn();
            y(t) = sum(x > beta_true) + 1;
            response = double(y(t) > nStaircaseRespondNo);
            if strcmp(staircaseType, 'standard')
                staircases{s} = updateForTrial(staircases{s}, log(I(t)), response);
            else
                staircases{s} = updateForTrial(staircases{s}, I(t), response);
            end
        end
    end

    c_init = linspace(0.5, 3.0, nCrit);

    % --- AB parameterisation ---
    theta0_ab = [log(1); log(1); c_init(1); log(diff(c_init))'];
    thAB      = fmincon(@(th) negLogLikAB(th, I, y), theta0_ab, ...
        [], [], [], [], [], [], @(th) critBounds(th, cMax, 3), opts);

    A_hat            = exp(thAB(1));
    b_hat_ab         = exp(thAB(2));
    A_hat_all(rep)   = A_hat;
    b_hat_ab_all(rep)= b_hat_ab;
    I_hat_ab_all(rep,:) = (dPrimeTargets / A_hat).^(1 / b_hat_ab);

    % --- Fixed-b parameterisation ---
    % b drawn from calibration distribution: b_true + noise
    b_fixed          = b_true + b_noise_sd * randn();
    b_fixed_all(rep) = b_fixed;

    theta0_fb = [log(Ithresh_true); c_init(1); log(diff(c_init))'];
    thFB      = fmincon(@(th) negLogLikFixedB(th, I, y, b_fixed, dPrimeRef), theta0_fb, ...
        [], [], [], [], [], [], @(th) critBounds(th, cMax, 2), opts);

    Ithresh_hat            = exp(thFB(1));
    Ithresh_hat_all(rep)   = Ithresh_hat;
    % Invert: I = I_thresh * (d' / dPrimeRef)^(1/b_fixed)
    I_hat_fb_all(rep,:)    = Ithresh_hat .* (dPrimeTargets / dPrimeRef).^(1 / b_fixed);

    fprintf('  Rep %3d/%d:  AB: A=%.3f b=%.3f  |  Fixed-b: b_used=%.3f Ith=%.3f\n', ...
        rep, nReps, A_hat, b_hat_ab, b_fixed, Ithresh_hat);
end

%% ---- Summary tables ------------------------------------------------------

fprintf('\n=== AB parameterisation — scale parameters (%d reps) ===\n', nReps);
fprintf('Param   True    Mean     Bias    SD\n');
fprintf('--------------------------------------\n');
fprintf('A       %.3f   %.3f    %+.3f   %.3f\n', ...
    A_true, mean(A_hat_all), mean(A_hat_all) - A_true, std(A_hat_all));
fprintf('b       %.3f   %.3f    %+.3f   %.3f\n', ...
    b_true, mean(b_hat_ab_all), mean(b_hat_ab_all) - b_true, std(b_hat_ab_all));

fprintf('\n=== Fixed-b parameterisation — scale parameters (%d reps) ===\n', nReps);
fprintf('Param      True    Mean     Bias    SD\n');
fprintf('-----------------------------------------\n');
fprintf('b (fixed)  %.3f   %.3f    %+.3f   %.3f\n', ...
    b_true, mean(b_fixed_all), mean(b_fixed_all) - b_true, std(b_fixed_all));
fprintf('I_thresh   %.3f   %.3f    %+.3f   %.3f\n', ...
    Ithresh_true, mean(Ithresh_hat_all), mean(Ithresh_hat_all) - Ithresh_true, std(Ithresh_hat_all));

fprintf('\n=== Threshold intensity estimates ===\n');
hdr = '  %-13s  %7s  %8s  %8s  %6s  %8s  %8s  %6s\n';
sep = repmat('-', 1, 72);
fprintf(hdr, '', 'I_true', 'AB mean', 'AB bias', 'AB SD', 'FB mean', 'FB bias', 'FB SD');
fprintf('  %s\n', sep);
for k = 1:nDPrime
    fprintf('  d''=%.2f  (I)    %7.3f  %8.3f  %8+.3f  %6.3f  %8.3f  %8+.3f  %6.3f\n', ...
        dPrimeTargets(k), I_true(k), ...
        mean(I_hat_ab_all(:,k)), mean(I_hat_ab_all(:,k)) - I_true(k), std(I_hat_ab_all(:,k)), ...
        mean(I_hat_fb_all(:,k)), mean(I_hat_fb_all(:,k)) - I_true(k), std(I_hat_fb_all(:,k)));
end
fprintf('  %s\n', sep);
for k = 1:nDPrime
    fprintf('  d''=%.2f  (logI) %7.3f  %8.3f  %8+.3f  %6.3f  %8.3f  %8+.3f  %6.3f\n', ...
        dPrimeTargets(k), log(I_true(k)), ...
        mean(log(I_hat_ab_all(:,k))), mean(log(I_hat_ab_all(:,k))) - log(I_true(k)), std(log(I_hat_ab_all(:,k))), ...
        mean(log(I_hat_fb_all(:,k))), mean(log(I_hat_fb_all(:,k))) - log(I_true(k)), std(log(I_hat_fb_all(:,k))));
end

%% ---- Figure --------------------------------------------------------------
% [1,1] b_hat (AB, estimated) vs b_fixed (fixed-b, drawn from calibration)
% [1,2] A_hat (AB) and I_thresh_hat (fixed-b) — the free scale parameters
% [2,1] I_hat from AB param
% [2,2] I_hat from fixed-b param

colors = lines(nDPrime);

figure('Color', 'w');
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% --- [1,1] b values used ---
nexttile;
histogram(b_hat_ab_all, 12, 'FaceColor', [0.2 0.5 0.8], 'FaceAlpha', 0.6, ...
    'DisplayName', 'b_{hat} (AB, estimated)');
hold on;
histogram(b_fixed_all,  12, 'FaceColor', [0.9 0.4 0.1], 'FaceAlpha', 0.6, ...
    'DisplayName', sprintf('b_{fixed} (calibration, SD=%.2f)', b_noise_sd));
xline(b_true, 'k-', 'LineWidth', 2, 'HandleVisibility', 'off');
xlabel('b');
title(sprintf('b  (true = %.2f)', b_true));
legend('Location', 'best');

% --- [1,2] Free scale parameter per method ---
nexttile;
histogram(A_hat_all,       12, 'FaceColor', [0.2 0.5 0.8], 'FaceAlpha', 0.6, ...
    'DisplayName', 'A_{hat} (AB)');
hold on;
histogram(Ithresh_hat_all, 12, 'FaceColor', [0.9 0.4 0.1], 'FaceAlpha', 0.6, ...
    'DisplayName', 'I_{thresh} (fixed-b)');
xline(A_true,       'b-', 'LineWidth', 2, 'HandleVisibility', 'off');
xline(Ithresh_true, 'r-', 'LineWidth', 2, 'HandleVisibility', 'off');
xlabel('Estimate');
title(sprintf('Free scale param: A_{true}=%.2f, I_{thresh,true}=%.2f', A_true, Ithresh_true));
legend('Location', 'best');

% --- [2,1] I_hat from AB ---
nexttile;
hold on;
for k = 1:nDPrime
    histogram(I_hat_ab_all(:,k), 10, 'FaceColor', colors(k,:), 'FaceAlpha', 0.6, ...
        'DisplayName', sprintf("d'=%.2f", dPrimeTargets(k)));
    xline(I_true(k),                '-',  'Color', colors(k,:), 'LineWidth', 2,   'HandleVisibility', 'off');
    xline(mean(I_hat_ab_all(:,k)), '--', 'Color', colors(k,:), 'LineWidth', 1.5, 'HandleVisibility', 'off');
end
xlabel('I_{hat}');
title('AB param  (solid=true, dashed=mean)');
legend('Location', 'northeast');

% --- [2,2] I_hat from fixed-b ---
nexttile;
hold on;
for k = 1:nDPrime
    histogram(I_hat_fb_all(:,k), 10, 'FaceColor', colors(k,:), 'FaceAlpha', 0.6, ...
        'DisplayName', sprintf("d'=%.2f", dPrimeTargets(k)));
    xline(I_true(k),                '-',  'Color', colors(k,:), 'LineWidth', 2,   'HandleVisibility', 'off');
    xline(mean(I_hat_fb_all(:,k)), '--', 'Color', colors(k,:), 'LineWidth', 1.5, 'HandleVisibility', 'off');
end
xlabel('I_{hat}');
title(sprintf('Fixed-b param  (b_{noise} SD=%.2f)', b_noise_sd));
legend('Location', 'northeast');

%% ---- Pack results --------------------------------------------------------

results.AB.A_hat    = A_hat_all;
results.AB.b_hat    = b_hat_ab_all;
results.AB.I_hat    = I_hat_ab_all;

results.FixedB.b_fixed    = b_fixed_all;
results.FixedB.Ithresh_hat = Ithresh_hat_all;
results.FixedB.I_hat      = I_hat_fb_all;

results.A_true        = A_true;
results.b_true        = b_true;
results.Ithresh_true  = Ithresh_true;
results.I_true        = I_true;
results.dPrimeTargets = dPrimeTargets;
results.dPrimeRef     = dPrimeRef;
results.b_noise_sd    = b_noise_sd;
results.staircaseType = staircaseType;

end

%% ======== Local functions =================================================

function [c, ceq] = critBounds(theta, cMax, critStart)
    beta = unpackCrit(theta(critStart:end));
    c    = beta(:) - cMax;
    ceq  = [];
end

function nll = negLogLikAB(theta, I, y)
    A    = exp(theta(1));
    b    = exp(theta(2));
    beta = unpackCrit(theta(3:end));
    R    = A .* (max(I, 0) .^ b);
    R(I == 0) = 0;
    nll  = ratingNLL(R, y, beta);
end

function nll = negLogLikFixedB(theta, I, y, b_fixed, dPrimeRef)
    I_thresh = exp(theta(1));
    beta     = unpackCrit(theta(2:end));
    R        = dPrimeRef .* (max(I, 0) ./ I_thresh) .^ b_fixed;
    R(I == 0) = 0;
    nll      = ratingNLL(R, y, beta);
end

function nll = ratingNLL(R, y, beta)
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
    R = theta(1) .* (I .^ theta(2));
end
