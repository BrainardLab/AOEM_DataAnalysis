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
%   Uses staircase functions from BrainardLabToolbox, and probably relies
%   on Psychtoolbox as well.  If you use the ToolboxToolbox, the following
%   command is sufficient (select and run):
%{
    tbUse('BrainardLabBase'); 
%}

% History:
%   2026-04-08 - DHB, HES, ClaudeAI - wrote it.

clear; clc; close all;
%rng(1);   % uncomment for reproducible runs

%% ---- True (simulation) parameters ----------------------------------------

A_true = 1.0;    % response-function gain
b_true = 0.8;      % response-function exponent  (R = A * I^b)

% Decision criteria specified in intensity space; converted to response
% (d-prime) space via the response function.
nResp      = 6;
Icrit_true = linspace(0.3, 2, nResp-1);                          % 5 values
beta_true  = responseFunction(Icrit_true, [A_true, b_true]);     % in R space
nCrit      = numel(beta_true);

% Threshold for "yes" in staircase updating and P(yes) plots.
% Ratings > nStaircaseRespondNo are treated as "yes"; others as "no".
nStaircaseRespondNo = 3;

% Target d-prime values for which we want intensity thresholds.
dPrimeTargets = [0.75, 1, 1.25];

%% ---- Staircase and simulation settings ------------------------------------
pCatch = 0.20;   % fraction of trials with I = 0 (catch / noise-only trials)

% Staircase type: 'standard' or 'quest'
staircaseType = 'standard';

I0   = 2.0;    % starting intensity for all staircases
Imin = 0.01;   % hard lower bound on intensity
Imax = 3.0;    % hard upper bound on intensity

% --- Standard staircase settings (used when staircaseType = 'standard') ---
% nUp(s)/nDown(s): incorrect/correct responses needed to step up/down.
% The length of nDown implicitly defines nStaircases.
nUp   = 1;
nDown = 2;

stepFactor = 1.15;   % multiplicative step size in intensity space
% Staircases operate in log(I) space; additive steps here = multiplicative in I.
stepSizes  = log(stepFactor) * [2, 1];   % two step sizes, shrinking after reversals

% --- QUEST settings (used when staircaseType = 'quest') ---
% questTargetProbs: target P(yes) for each interleaved QUEST.
% Multiple values => multiple interleaved QUESTs converging to different
% intensity levels, giving broader coverage of the psychometric function.
questTargetProbs = [0.6, 0.75, 0.9];   % one QUEST per entry
questBeta        = 3.5;    % Weibull slope (approximate for power-law SDT)
questDelta       = 0.01;   % lapse rate
questGamma       = 0.0;    % guess rate (0 for yes/no detection)
questPriorSD     = 10;     % log10(questPriorSD) = prior SD in log10(I) units
                            % (10 => 1 log10 unit of prior uncertainty)

% --- Derive nStaircases from the chosen type ---
if strcmp(staircaseType, 'standard')
    nStaircases = numel(nDown);
else
    nStaircases = numel(questTargetProbs);
end

nTrialsPerStaircase = 150;
nTrials             = nTrialsPerStaircase * nStaircases;

%% ---- Initialise staircases ------------------------------------------------

staircases = cell(nStaircases, 1);
for s = 1:nStaircases
    if strcmp(staircaseType, 'standard')
        % Standard staircase operates in log(I) space internally.
        staircases{s} = Staircase('standard', log(I0), ...
            'StepSizes', stepSizes, ...
            'NUp',       nDown(s), ...   % correct responses to step down
            'NDown',     nUp(s),   ...   % incorrect responses to step up
            'MaxValue',  log(Imax), ...
            'MinValue',  log(Imin));
    else
        % QUEST handles log10(I) conversion internally;
        % getCurrentValue returns linear I, updateForTrial takes linear I.
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

%% ---- Simulate trial sequence ----------------------------------------------

I    = zeros(nTrials, 1);   % intensity presented on each trial
y    = zeros(nTrials, 1);   % rating response (1..nResp) on each trial
sIdx = zeros(nTrials, 1);   % staircase index for signal trials; 0 for catch

% Signal trials cycle through staircases in strict round-robin order
% (1,2,...,nStaircases,1,2,...) so each staircase gets an equal share.
signalCount = 0;

for t = 1:nTrials

    if rand < pCatch
        % Catch trial: no signal, no staircase update.
        I(t)    = 0;
        sIdx(t) = 0;
    else
        % Signal trial: select next staircase in the cycle.
        signalCount = signalCount + 1;
        s           = mod(signalCount - 1, nStaircases) + 1;
        sIdx(t)     = s;
        if strcmp(staircaseType, 'standard')
            I(t) = exp(getCurrentValue(staircases{s}));
        else
            I(t) = getCurrentValue(staircases{s});
        end

        % Sample noisy internal response and assign rating.
        x    = responseFunction(I(t), [A_true, b_true]) + randn();
        y(t) = sum(x > beta_true) + 1;

        % Update staircase.  Response = 1 ("yes") iff rating exceeds threshold.
        response = double(y(t) > nStaircaseRespondNo);
        if strcmp(staircaseType, 'standard')
            staircases{s} = updateForTrial(staircases{s}, log(I(t)), response);
        else
            staircases{s} = updateForTrial(staircases{s}, I(t), response);
        end
    end

    % Catch trials also get a rating (used in MLE fitting and ROC analysis).
    if I(t) == 0
        x    = responseFunction(0, [A_true, b_true]) + randn();
        y(t) = sum(x > beta_true) + 1;
    end
end

%% ---- Fit model by MLE -----------------------------------------------------
%
% Parameter vector theta passed to fmincon:
%   theta = [log(A), log(b), c1, log(c2-c1), log(c3-c2), ..., log(c_nCrit - c_{nCrit-1})]
%
% log(A) and log(b) enforce A, b > 0 without box constraints.
% The criteria are encoded via unpackCrit: c1 is free; each subsequent
% criterion is c_{k} = c_{k-1} + exp(theta_{k}), enforcing strict ordering.
%
% cMax caps all estimated criteria in R space.  Criteria above this cannot
% be reliably estimated from the available intensity range; without the cap
% the optimiser can produce astronomically large values that break plots.

c_init = linspace(0.5, 3.0, nCrit);
theta0 = [log(1); log(1); c_init(1); log(diff(c_init))'];

cMax = 3.5;   % upper bound on criteria in R (d-prime) space

opts = optimoptions('fmincon', 'Display', 'iter', ...
    'MaxIterations', 5000, 'MaxFunctionEvaluations', 50000);

thetaHat = fmincon(@(th) negLogLik(th, I, y), theta0, ...
    [], [], [], [], [], [], @(th) critBounds(th, cMax), opts);

A_hat    = exp(thetaHat(1));
b_hat    = exp(thetaHat(2));
beta_hat = unpackCrit(thetaHat(3:end));

% Back-transform estimated criteria from R space to intensity space.
% Inverse of R = A*I^b  =>  I = (R/A)^(1/b).
Icrit_hat = (beta_hat / A_hat).^(1 / b_hat);

%% ---- Print parameter comparison -------------------------------------------

fprintf('\nTrue vs estimated parameters\n');
fprintf('----------------------------------------------------------\n');
fprintf('A     true = %8.4f   hat = %8.4f\n', A_true, A_hat);
fprintf('b     true = %8.4f   hat = %8.4f\n', b_true, b_hat);
fprintf('\n  Crit   Icrit_true  Icrit_hat   beta_true   beta_hat\n');
for k = 1:nCrit
    fprintf('  c%-2d    %8.4f   %8.4f    %8.4f   %8.4f\n', ...
        k, Icrit_true(k), Icrit_hat(k), beta_true(k), beta_hat(k));
end

% Compute intensities that produce each target d-prime under true and fitted model.
% Invert R = A*I^b  =>  I = (d'/A)^(1/b).
nDPrime       = numel(dPrimeTargets);
I_true_dp_vec = zeros(1, nDPrime);
I_hat_dp_vec  = zeros(1, nDPrime);

fprintf('\nTarget d-prime intensities\n');
fprintf('----------------------------------------------------------\n');
fprintf('  d-prime   I_true    I_hat\n');
for k = 1:nDPrime
    I_true_dp_vec(k) = (dPrimeTargets(k) / A_true)^(1 / b_true);
    I_hat_dp_vec(k)  = (dPrimeTargets(k) / A_hat )^(1 / b_hat );
    fprintf('  %6.2f    %8.4f  %8.4f\n', dPrimeTargets(k), I_true_dp_vec(k), I_hat_dp_vec(k));
end

%% ---- Figure 1: staircase data and fitted model ----------------------------

% Compute response grids used across all three panels.
Igrid = linspace(0, max(max(I), Imax) * 1.05, 300)';
rTrue = responseFunction(Igrid, [A_true, b_true]);
rHat  = responseFunction(Igrid, [A_hat,  b_hat]);
Rmax  = max([rTrue; rHat]) * 1.05;
Rgrid = linspace(0, Rmax, 300)';

scColors = lines(nStaircases);   % consistent colour per staircase across all panels

figure('Color', 'w');
tiledlayout(3, 1);

% --- Panel 1: staircase intensity trajectories ---
% Each staircase is a distinct colour.
% Filled circles = "yes" (correct for staircase update); x = "no".
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
    lgdLabels = [arrayfun(@(u,d) sprintf('%d-up %d-down', u, d), nUp.*ones(1,nStaircases), nDown.*ones(1,nStaircases), 'UniformOutput', false), ...
                 {'Correct', 'Incorrect'}];
else
    lgdLabels = [arrayfun(@(p) sprintf('QUEST P=%.2f', p), questTargetProbs, 'UniformOutput', false), ...
                 {'Correct', 'Incorrect'}];
end
legend(lgdHandles, lgdLabels, 'Location', 'northeastoutside');

% --- Panel 2: intensity response function R = A*I^b ---
% Criteria shown as vertical dotted lines (black = true, red = fitted).
% Trial rug: "yes" responses near top of axis, "no" near bottom.
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
% P(yes | R) = Phi(R - beta_{nStaircaseRespondNo}), i.e. the probability of
% exceeding the "yes" criterion as a function of d-prime.
% All five criteria are shown as vertical dotted lines.
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
%                 using the equal-variance identity  AUC = Phi(d'/sqrt(2))
%   MLE         : d-prime estimated by maximum likelihood, with criteria
%                 treated as nuisance parameters (negLogLikFixed)

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
    R_true_fixed = R_true_at_Ihat(k);   % true d-prime at this intensity

    fprintf('  %6.2f           %8.4f  %8.4f\n', dPrimeTargets(k), I_fixed, R_true_fixed);

    % Simulate signal trials.  Internal response mean = R_true_fixed.
    ySignal = zeros(nFixedSignal, 1);
    for t = 1:nFixedSignal
        x = R_true_fixed + randn();
        ySignal(t) = sum(x > beta_true) + 1;
    end

    % Simulate catch trials.  Internal response mean = 0.
    yCatch = zeros(nFixedCatch, 1);
    for t = 1:nFixedCatch
        x = responseFunction(0, [A_true, b_true]) + randn();
        yCatch(t) = sum(x > beta_true) + 1;
    end

    % Compute empirical ROC by sweeping all nCrit criterion splits.
    % For split at j: "yes" = rating > j.
    % Endpoints (FA=1,Hit=1) and (FA=0,Hit=0) are appended analytically.
    FA  = [1; zeros(nCrit, 1); 0];
    Hit = [1; zeros(nCrit, 1); 0];
    for j = 1:nCrit
        FA(j+1)  = mean(yCatch  > j);
        Hit(j+1) = mean(ySignal > j);
    end

    % Theoretical ROC from true d-prime, parameterised by a continuous criterion.
    betaVec = linspace(-2, R_true_fixed + 3, 500)';
    FA_th   = Phi(-betaVec);
    Hit_th  = Phi(R_true_fixed - betaVec);

    % AUC-based d-prime estimate.
    % Equal-variance Gaussian model gives AUC = Phi(d'/sqrt(2)),
    % so d' = sqrt(2) * norminv(AUC) = -2 * erfcinv(2*AUC)
    % (using norminv(p) = -sqrt(2)*erfcinv(2p) to avoid Statistics Toolbox).
    AUC         = abs(trapz(FA, Hit));
    AUC         = min(max(AUC, 0.501), 0.999);   % clamp to keep erfcinv finite
    d_auc       = -2 * erfcinv(2 * AUC);
    betaVec_auc = linspace(-2, d_auc + 3, 500)';
    FA_auc      = Phi(-betaVec_auc);
    Hit_auc     = Phi(d_auc - betaVec_auc);

    % MLE d-prime estimate.
    % Fit d-prime and all criteria jointly from ySignal + yCatch.
    % Initialise criteria from the staircase-phase fit (beta_hat).
    theta0_fixed   = [1.0; beta_hat(1); log(diff(beta_hat))'];
    lb_fixed       = [0; -Inf(nCrit, 1)];   % enforce d-prime >= 0
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

%% ======== Local functions ================================================

function [c, ceq] = critBounds(theta, cMax)
% Nonlinear inequality constraint for the staircase-phase MLE.
% theta = [log(A), log(b), c1, log-gaps...]; criteria are in theta(3:end).
% Enforces beta_k <= cMax for all k.
    beta = unpackCrit(theta(3:end));
    c    = beta(:) - cMax;
    ceq  = [];
end

function [c, ceq] = critBoundsFixed(theta, cMax)
% Nonlinear inequality constraint for the fixed-intensity MLE.
% theta = [d_prime, c1, log-gaps...]; criteria are in theta(2:end).
    beta = unpackCrit(theta(2:end));
    c    = beta(:) - cMax;
    ceq  = [];
end

function nll = negLogLik(theta, I, y)
% Negative log-likelihood for the staircase-phase data.
% Fits A, b, and all nCrit criteria jointly across varying intensities.
%
% theta = [log(A), log(b), c1, log(c2-c1), ..., log(c_nCrit - c_{nCrit-1})]
%
% P(rating = k | I) = Phi(beta_k - R) - Phi(beta_{k-1} - R)
% where R = A * I^b and beta_0 = -Inf, beta_{nCrit+1} = +Inf.
    A    = exp(theta(1));
    b    = exp(theta(2));
    beta = unpackCrit(theta(3:end));

    R      = responseFunction(I, [A, b]);
    bounds = [-Inf, beta, Inf];
    lo     = bounds(y);
    hi     = bounds(y + 1);

    p   = max(Phi(hi(:) - R) - Phi(lo(:) - R), realmin);
    nll = -sum(log(p));
    if ~isfinite(nll);  nll = realmax;  end
end

function nll = negLogLikFixed(theta, ySignal, yCatch)
% Negative log-likelihood for fixed-intensity data.
% Fits a single d-prime value (for the one signal level) jointly with all
% criteria, treating criteria as nuisance parameters.
%
% theta = [d_prime, c1, log(c2-c1), ..., log(c_nCrit - c_{nCrit-1})]
%
% Signal trials: internal response ~ N(d_prime, 1)
% Catch  trials: internal response ~ N(0, 1)
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
% Reconstruct ordered criteria from the MLE parameterisation.
% v(1)   = c1  (free, unconstrained)
% v(k>1) = log(c_k - c_{k-1})  (log of positive gap, enforces ordering)
    beta    = zeros(1, numel(v));
    beta(1) = v(1);
    for k = 2:numel(v)
        beta(k) = beta(k-1) + exp(v(k));
    end
end

function p = Phi(z)
% Standard normal CDF implemented without the Statistics Toolbox.
% Uses the complementary error function identity: Phi(z) = 0.5*erfc(-z/sqrt(2)).
    p = 0.5 * erfc(-z ./ sqrt(2));
end

function R = responseFunction(I, theta)
% Mean internal response (d-prime) as a function of stimulus intensity.
% Model: R = A * I^b   (power-law / Stevens-law nonlinearity)
% theta = [A, b].  Accepts scalar or array I; returns same size.
    A = theta(1);
    b = theta(2);
    R = A .* (I .^ b);
end
