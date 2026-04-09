% YesNoTSDSimulation.m
%
% End-to-end simulation of a rating-scale detection experiment using
% signal detection theory (SDT), followed by maximum-likelihood fitting
% and threshold estimation.
%
% GENERATIVE MODEL
%   Internal response on a trial with stimulus intensity I:
%       x = R(I) + epsilon,   epsilon ~ N(0,1)
%   where the mean response (d-prime) is a power-law:
%       R(I) = A * I^b          [responseFunction]
%
%   The observer maps x to one of nResp = nCrit+1 rating categories by
%   comparing x to an ordered set of criteria beta_1 < ... < beta_nCrit:
%       rating k  iff  beta_{k-1} < x <= beta_k
%   (with beta_0 = -Inf, beta_{nCrit+1} = +Inf).
%
%   On catch trials (I = 0), R = 0, so x = epsilon (pure noise).
%
% STAIRCASE
%   Multiple interleaved staircases (BrainardLabToolbox Staircase object)
%   adapt intensity to keep hit rate near a target level.  A "hit" is
%   defined as rating > nStaircaseRespondNo.  Staircases operate in log(I)
%   space so that additive steps correspond to multiplicative steps in I.
%
% FITTING
%   MLE fits A, b, and all criteria jointly from the staircase data using
%   fmincon.  A and b are reparameterized as exp(.) to enforce positivity;
%   criteria are reparameterized as a first value plus cumulative sums of
%   exp(.) gaps to enforce strict ordering (see unpackCrit).
%
% THRESHOLD ESTIMATION
%   For a set of target d-prime values the fitted model is inverted to give
%   estimated intensities.  Fixed-intensity simulations at those levels then
%   verify the estimates via two d-prime recovery methods:
%   (1) AUC of the empirical rating ROC, and
%   (2) MLE of d-prime with criteria treated as nuisance parameters.
%
% REQUIRES
%   BrainardLabToolbox (Staircase class) and Psychtoolbox-3 (QuestCreate,
%   needed by the 'quest' staircaseType).  Run initSession.m once at the
%   start of each MATLAB session to load both toolboxes together:
%{
    initSession
%}

% History:
%   2026-04-08 - DHB, HES, ClaudeAI - wrote it.
%   2026-04-08 - DHB, HES, ClaudeAI - refactored: simulation and AB fit
%                moved to runYesNoTSDSession.m.

clear; clc; close all;
%rng(1);   % uncomment for reproducible runs

%% ---- Parameters ----------------------------------------------------------

% True model
params.A_true              = 1.0;
params.b_true              = 0.8;
params.nResp               = 6;
params.Icrit_true          = linspace(0.3, 2, params.nResp - 1);
params.nStaircaseRespondNo = 3;
params.dPrimeTargets       = [0.75, 1, 1.25];

% Staircase type: 'standard' or 'quest'
% params.staircaseType = 'standard';
params.staircaseType = 'quest';

% Common staircase settings
params.pCatch = 0.20;
params.I0     = 2.0;
params.Imin   = 0.01;
params.Imax   = 3.0;

% Standard staircase: 3 interleaved staircases with nDown=[2,5,9].
% Convergence probabilities P ≈ [0.618, 0.755, 0.822] (from P^nDown = (1-P)^nUp),
% chosen to approximately match QUEST targets [0.6, 0.75, 0.9].
% Total trials = 50 per staircase x 3 = 150 (same as 3-QUEST design).
params.nUp                 = [1, 1, 1];
params.nDown               = [2, 5, 9];
params.stepSizes           = log(1.15) * [2, 1];
params.nTrialsPerStaircase = 50;

% QUEST settings (used when staircaseType = 'quest')
% questTargetProbs: one QUEST per entry, each targeting a different P(yes).
params.questTargetProbs = [0.6, 0.75, 0.9];
params.questBeta        = 3.5;
params.questDelta       = 0.01;
params.questGamma       = 0.0;
params.questPriorSD     = 10;

% Fitting
params.cMax        = 3.5;
params.fitDisplay  = 'iter';   % show fmincon iterations for this interactive run

%% ---- Run one session -----------------------------------------------------

session = runYesNoTSDSession(params);

% Unpack session results and params for use in print/plot sections below.
A_true       = params.A_true;
b_true       = params.b_true;
Icrit_true   = params.Icrit_true;
dPrimeTargets= params.dPrimeTargets;
nDPrime      = numel(dPrimeTargets);
Imax         = params.Imax;
cMax         = params.cMax;
nStaircaseRespondNo = params.nStaircaseRespondNo;
staircaseType= params.staircaseType;

beta_true    = responseFunction(Icrit_true, [A_true, b_true]);
nCrit        = numel(beta_true);

A_hat        = session.A_hat;
b_hat        = session.b_hat;
beta_hat     = session.beta_hat;
Icrit_hat    = session.Icrit_hat;
I_hat_dp_vec = session.I_hat;
I            = session.I;
y            = session.y;
sIdx         = session.sIdx;
staircases   = session.staircases;
nStaircases  = session.nStaircases;

%% ---- Print parameter comparison ------------------------------------------

fprintf('\nTrue vs estimated parameters\n');
fprintf('----------------------------------------------------------\n');
fprintf('A     true = %8.4f   hat = %8.4f\n', A_true, A_hat);
fprintf('b     true = %8.4f   hat = %8.4f\n', b_true, b_hat);
fprintf('\n  Crit   Icrit_true  Icrit_hat   beta_true   beta_hat\n');
for k = 1:nCrit
    fprintf('  c%-2d    %8.4f   %8.4f    %8.4f   %8.4f\n', ...
        k, Icrit_true(k), Icrit_hat(k), beta_true(k), beta_hat(k));
end

I_true_dp_vec = (dPrimeTargets / A_true).^(1 / b_true);
fprintf('\nTarget d-prime intensities\n');
fprintf('----------------------------------------------------------\n');
fprintf('  d-prime   I_true    I_hat\n');
for k = 1:nDPrime
    fprintf('  %6.2f    %8.4f  %8.4f\n', dPrimeTargets(k), I_true_dp_vec(k), I_hat_dp_vec(k));
end

%% ---- Figure 1: staircase data and fitted model ----------------------------

Igrid = linspace(0, max(max(I), Imax) * 1.05, 300)';
rTrue = responseFunction(Igrid, [A_true, b_true]);
rHat  = responseFunction(Igrid, [A_hat,  b_hat]);
Rmax  = max([rTrue; rHat]) * 1.05;
Rgrid = linspace(0, Rmax, 300)';

scColors = lines(nStaircases);

figure('Color', 'w');
tiledlayout(3, 1);

% --- Panel 1: staircase intensity trajectories ---
nexttile;
correct    = (y > nStaircaseRespondNo);
lgdHandles = gobjects(nStaircases + 2, 1);
hold on;
for s = 1:nStaircases
    mask  = (sIdx == s);
    cMask = mask & correct;
    plot(find(mask), I(mask), '-', 'Color', scColors(s,:), 'LineWidth', 1);
    lgdHandles(s) = plot(find(cMask), I(cMask), 'o', ...
        'Color', scColors(s,:), 'MarkerFaceColor', scColors(s,:), 'MarkerSize', 8);
    plot(find(mask & ~correct), I(mask & ~correct), 'x', ...
        'Color', scColors(s,:), 'MarkerSize', 10, 'LineWidth', 1.5);
end
lgdHandles(nStaircases+1) = plot(nan, nan, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 8);
lgdHandles(nStaircases+2) = plot(nan, nan, 'kx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Trial');  ylabel('Intensity');  ylim([0, Imax]);
title('Interleaved staircase trajectories');
if strcmp(staircaseType, 'standard')
    lgdLabels = [arrayfun(@(u,d) sprintf('%d-up %d-down', u, d), ...
                     params.nUp  .* ones(1, nStaircases), ...
                     params.nDown .* ones(1, nStaircases), ...
                     'UniformOutput', false), {'Correct', 'Incorrect'}];
else
    lgdLabels = [arrayfun(@(p) sprintf('QUEST P=%.2f', p), ...
                     params.questTargetProbs, 'UniformOutput', false), ...
                 {'Correct', 'Incorrect'}];
end
legend(lgdHandles, lgdLabels, 'Location', 'northeastoutside');

% --- Panel 2: intensity response function R = A*I^b ---
nexttile;
plot(Igrid, rTrue, 'k-', 'LineWidth', 2);  hold on;
plot(Igrid, rHat,  'r--', 'LineWidth', 2);
isYes2 = (y > nStaircaseRespondNo);
yRug   = double(isYes2) * Rmax + 0.02 * Rmax * (rand(size(y)) - 0.5);
scatter(I(sIdx==0), yRug(sIdx==0), 18, [0.6 0.6 0.6], 'filled', 'MarkerFaceAlpha', 0.20);
for s = 1:nStaircases
    mask = (sIdx == s);
    scatter(I(mask), yRug(mask), 18, scColors(s,:), 'filled', 'MarkerFaceAlpha', 0.20);
end
for k = 1:nCrit
    xline(Icrit_true(k), 'k:', 'LineWidth', 1.2);
    xline(Icrit_hat(k),  'r:', 'LineWidth', 1.2);
end
xlim([0, max(max(I), Imax) * 1.05]);
xlabel('Intensity');  ylabel('R = A \cdot I^b');
title('Intensity response function');
legend('True', 'Fitted', 'Trials', 'Location', 'southeast');

% --- Panel 3: psychometric function P(yes | R) ---
nexttile;
pYesTrue = Phi(Rgrid - beta_true(nStaircaseRespondNo));
pYesHat  = Phi(Rgrid - beta_hat(nStaircaseRespondNo));
plot(Rgrid, pYesTrue, 'k-',  'LineWidth', 2);  hold on;
plot(Rgrid, pYesHat,  'r--', 'LineWidth', 2);
for k = 1:nCrit
    xline(beta_true(k), 'k:', 'LineWidth', 1.2);
    xline(beta_hat(k),  'r:', 'LineWidth', 1.2);
end
xlim([0, Rmax]);
Robs  = responseFunction(I, [A_true, b_true]);
isYes = (y > nStaircaseRespondNo);
yj    = double(isYes) + 0.04 * (rand(size(y)) - 0.5);
scatter(Robs(sIdx==0), yj(sIdx==0), 18, [0.6 0.6 0.6], 'filled', 'MarkerFaceAlpha', 0.20);
for s = 1:nStaircases
    mask = (sIdx == s);
    scatter(Robs(mask), yj(mask), 18, scColors(s,:), 'filled', 'MarkerFaceAlpha', 0.20);
end
ylim([-0.1, 1.1]);
xlabel('R = A \cdot I^b  (d-prime)');  ylabel('P(yes)');
title(sprintf('P(yes) in response space  (yes = rating > %d)', nStaircaseRespondNo));
legend('True', 'Fitted', '\beta true', '\beta fitted', 'Observed', 'Location', 'southeast');

%% ---- Figure 2: fixed-intensity ROC curves ---------------------------------
%
% For each target d-prime we simulate nFixedSignal signal trials at the
% fitted threshold intensity I_hat, plus nFixedCatch catch trials (I = 0),
% using the TRUE model parameters and criteria.
%
% Three d-prime estimates are compared in each ROC panel:
%   Theoretical : true d-prime at I_hat under the true model
%   AUC         : d-prime back-calculated from the empirical ROC area
%   MLE         : d-prime estimated with criteria as nuisance parameters

nFixedSignal = 150;
nFixedCatch  = 75;

fprintf('\nFixed-intensity simulation results\n');
fprintf('----------------------------------------------------------\n');
fprintf('  Target d-prime   I_hat     True d-prime at I_hat\n');

rocColors      = lines(nDPrime);
R_true_at_Ihat = responseFunction(I_hat_dp_vec, [A_true, b_true]);
opts_quiet     = optimoptions('fmincon', 'Display', 'off', ...
    'MaxIterations', 5000, 'MaxFunctionEvaluations', 50000);

figure('Color', 'w', 'Position', [100 100 500*nDPrime 500]);
tiledlayout(1, nDPrime, 'TileSpacing', 'compact', 'Padding', 'compact');

for k = 1:nDPrime
    I_fixed      = I_hat_dp_vec(k);
    R_true_fixed = R_true_at_Ihat(k);

    fprintf('  %6.2f           %8.4f  %8.4f\n', dPrimeTargets(k), I_fixed, R_true_fixed);

    % Simulate signal trials.
    ySignal = zeros(nFixedSignal, 1);
    for t = 1:nFixedSignal
        x = R_true_fixed + randn();
        ySignal(t) = sum(x > beta_true) + 1;
    end

    % Simulate catch trials.
    yCatch = zeros(nFixedCatch, 1);
    for t = 1:nFixedCatch
        x = responseFunction(0, [A_true, b_true]) + randn();
        yCatch(t) = sum(x > beta_true) + 1;
    end

    % Empirical ROC.
    FA  = [1; zeros(nCrit, 1); 0];
    Hit = [1; zeros(nCrit, 1); 0];
    for j = 1:nCrit
        FA(j+1)  = mean(yCatch  > j);
        Hit(j+1) = mean(ySignal > j);
    end

    % Theoretical ROC.
    betaVec = linspace(-2, R_true_fixed + 3, 500)';
    FA_th   = Phi(-betaVec);
    Hit_th  = Phi(R_true_fixed - betaVec);

    % AUC-based d-prime.
    AUC         = abs(trapz(FA, Hit));
    AUC         = min(max(AUC, 0.501), 0.999);
    d_auc       = -2 * erfcinv(2 * AUC);
    betaVec_auc = linspace(-2, d_auc + 3, 500)';
    FA_auc      = Phi(-betaVec_auc);
    Hit_auc     = Phi(d_auc - betaVec_auc);

    % MLE d-prime.
    theta0_fixed   = [1.0; beta_hat(1); log(diff(beta_hat))'];
    lb_fixed       = [0; -Inf(nCrit, 1)];
    thetaHat_fixed = fmincon( ...
        @(th) negLogLikFixed(th, ySignal, yCatch), ...
        theta0_fixed, [], [], [], [], lb_fixed, [], ...
        @(th) critBoundsFixed(th, cMax), opts_quiet);
    d_mle       = thetaHat_fixed(1);
    betaVec_mle = linspace(-2, d_mle + 3, 500)';
    FA_mle      = Phi(-betaVec_mle);
    Hit_mle     = Phi(d_mle - betaVec_mle);

    nexttile;
    plot(FA_th,  Hit_th,  '-',  'Color', rocColors(k,:), 'LineWidth', 2);   hold on;
    plot(FA,     Hit,     'o',  'Color', rocColors(k,:), ...
        'MarkerFaceColor', rocColors(k,:), 'MarkerSize', 6);
    plot(FA_auc, Hit_auc, '--', 'Color', rocColors(k,:), 'LineWidth', 1.5);
    plot(FA_mle, Hit_mle, '-.', 'Color', rocColors(k,:), 'LineWidth', 1.5);
    plot([0 1],  [0 1],   'k--', 'LineWidth', 1);
    axis square;  xlim([0 1]);  ylim([0 1]);
    xlabel('False alarm rate');  ylabel('Hit rate');
    title(sprintf("d' target=%.2f, true=%.2f, AUC=%.2f, MLE=%.2f", ...
        dPrimeTargets(k), R_true_fixed, d_auc, d_mle));
    legend('Theoretical', 'Empirical', 'AUC est.', 'MLE est.', 'Chance', ...
        'Location', 'southeast');
end

%% ======== Local functions (fixed-intensity ROC only) ======================
% Note: the AB fitting functions live in runYesNoTSDSession.m.
% unpackCrit, Phi, responseFunction are duplicated here as local functions
% because MATLAB local functions are file-scoped.

function [c, ceq] = critBoundsFixed(theta, cMax)
    beta = unpackCrit(theta(2:end));
    c    = beta(:) - cMax;
    ceq  = [];
end

function nll = negLogLikFixed(theta, ySignal, yCatch)
    dPrime = theta(1);
    beta   = unpackCrit(theta(2:end));
    bounds = [-Inf, beta, Inf];
    lo_s = bounds(ySignal);   hi_s = bounds(ySignal + 1);
    lo_c = bounds(yCatch);    hi_c = bounds(yCatch  + 1);
    p_s = max(Phi(hi_s(:) - dPrime) - Phi(lo_s(:) - dPrime), realmin);
    p_c = max(Phi(hi_c(:))          - Phi(lo_c(:)),           realmin);
    nll = -(sum(log(p_s)) + sum(log(p_c)));
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
