function results = checkEstimationBias(nReps, b_noise_sd, staircaseType)
% results = checkEstimationBias(nReps, b_noise_sd, staircaseType)
%
% Runs nReps staircase sessions via runYesNoTSDSession and compares two
% fitting approaches:
%
%   AB param    : fits A and b freely  (done inside runYesNoTSDSession)
%   Fixed-b     : fixes b at b_true + N(0, b_noise_sd), fits only I_thresh
%                 where R(I) = (I/I_thresh)^b_fixed.
%
% Staircase type is shared with YesNoTSDSimulation.m:
%   'standard' : 3 interleaved staircases (nDown=[2,5,9]) x 50 trials = 150 total
%   'quest'    : 3 interleaved QUESTs x 50 trials = 150 total
%
% Inputs:
%   nReps         - number of replications (default: 50)
%   b_noise_sd    - SD of calibration error on b (default: 0.1)
%   staircaseType - 'standard' or 'quest' (default: 'standard')
%
% Output:
%   results - struct with AB and FixedB sub-structs and metadata
%
% Examples:
%   r_std   = checkEstimationBias(50, 0.1, 'standard');
%   r_quest = checkEstimationBias(50, 0.1, 'quest');
%
% Requires:
%   BrainardLabToolbox and Psychtoolbox-3.  Run initSession.m once at the
%   start of each MATLAB session before calling this function.
%
% History:
%   2026-04-08  DHB, HES, ClaudeAI  wrote it.
%   2026-04-08  DHB, HES, ClaudeAI  added fixed-b parameterisation.
%   2026-04-08  DHB, HES, ClaudeAI  added staircaseType parameter.
%   2026-04-08  DHB, HES, ClaudeAI  refactored to use runYesNoTSDSession.

if nargin < 1;  nReps         = 50;         end
if nargin < 2;  b_noise_sd    = 0.1;        end
if nargin < 3;  staircaseType = 'standard'; end

%% ---- Build params struct (mirrors YesNoTSDSimulation.m defaults) ---------

params.A_true              = 1.0;
params.b_true              = 0.8;
params.nResp               = 6;
params.Icrit_true          = linspace(0.3, 2, params.nResp - 1);
params.nStaircaseRespondNo = 3;
params.dPrimeTargets       = [0.75, 1, 1.25];

params.staircaseType = staircaseType;
params.pCatch        = 0.20;
params.I0            = 2.0;
params.Imin          = 0.01;
params.Imax          = 3.0;

% Standard staircase: 3 interleaved staircases with nDown=[2,5,9].
% Convergence probabilities: P^nDown = (1-P)^nUp gives P ≈ [0.618, 0.755, 0.822],
% chosen to approximately match the QUEST targets [0.6, 0.75, 0.9].
% (Exactly matching 0.9 would require nDown≈22, which is impractical.)
params.nUp       = [1, 1, 1];
params.nDown     = [2, 5, 9];
params.stepSizes = log(1.15) * [2, 1];

params.questTargetProbs = [0.6, 0.75, 0.9];
params.questBeta        = 3.5;
params.questDelta       = 0.01;
params.questGamma       = 0.0;
params.questPriorSD     = 10;

% Total signal trials fixed at 150, divided evenly across staircases.
%   standard: 3 x 50 = 150,   quest: 3 x 50 = 150
if strcmp(staircaseType, 'standard')
    nSC = numel(params.nDown);
else
    nSC = numel(params.questTargetProbs);
end
params.nTrialsPerStaircase = 150 / nSC;

params.cMax       = 3.5;
params.fitDisplay = 'off';   % suppress fmincon output in Monte Carlo loop

nFixedSignal = 150;   % signal trials per fixed-intensity validation rep
nFixedCatch  = 75;    % catch trials per fixed-intensity validation rep

%% ---- Derived true quantities ---------------------------------------------

A_true    = params.A_true;
b_true    = params.b_true;
dPrimeTargets = params.dPrimeTargets;
nDPrime   = numel(dPrimeTargets);
dPrimeRef = 1.0;

I_true       = (dPrimeTargets / A_true).^(1 / b_true);
Ithresh_true = (dPrimeRef    / A_true).^(1 / b_true);
beta_true    = A_true .* (params.Icrit_true .^ b_true);

nCrit  = numel(params.Icrit_true);
cMax   = params.cMax;
opts   = optimoptions('fmincon', 'Display', 'off', ...
    'MaxIterations', 5000, 'MaxFunctionEvaluations', 50000);

%% ---- Storage -------------------------------------------------------------

% AB parameterisation (from runYesNoTSDSession)
A_hat_all    = zeros(nReps, 1);
b_hat_ab_all = zeros(nReps, 1);
I_hat_ab_all = zeros(nReps, nDPrime);

% Fixed-b parameterisation
b_fixed_all     = zeros(nReps, 1);
Ithresh_hat_all = zeros(nReps, 1);
I_hat_fb_all    = zeros(nReps, nDPrime);

% Fixed-intensity ROC validation (AB parameterisation)
d_auc_all = zeros(nReps, nDPrime);
d_mle_all = zeros(nReps, nDPrime);

%% ---- Replication loop ----------------------------------------------------

fprintf('Running %d replications (staircaseType=%s, b_noise_sd=%.2f)...\n', ...
    nReps, staircaseType, b_noise_sd);

for rep = 1:nReps

    rng(rep);

    % --- Run one staircase session and AB fit via shared function ---
    session = runYesNoTSDSession(params);

    A_hat_all(rep)      = session.A_hat;
    b_hat_ab_all(rep)   = session.b_hat;
    I_hat_ab_all(rep,:) = session.I_hat;

    % --- Fixed-b parameterisation: b fixed, only I_thresh free ---
    b_fixed          = b_true + b_noise_sd * randn();
    b_fixed_all(rep) = b_fixed;

    c_init    = linspace(0.5, 3.0, nCrit);
    theta0_fb = [log(Ithresh_true); c_init(1); log(diff(c_init))'];

    thFB = fmincon(@(th) negLogLikFixedB(th, session.I, session.y, b_fixed, dPrimeRef), ...
        theta0_fb, [], [], [], [], [], [], @(th) critBoundsFixedB(th, cMax), opts);

    Ithresh_hat          = exp(thFB(1));
    Ithresh_hat_all(rep) = Ithresh_hat;
    I_hat_fb_all(rep,:)  = Ithresh_hat .* (dPrimeTargets / dPrimeRef).^(1 / b_fixed);

    % --- Fixed-intensity ROC validation (AB parameterisation) ---
    for k = 1:nDPrime
        I_fixed      = I_hat_ab_all(rep, k);
        R_true_fixed = A_true * I_fixed ^ b_true;

        ySignal = zeros(nFixedSignal, 1);
        for t = 1:nFixedSignal
            x = R_true_fixed + randn();
            ySignal(t) = sum(x > beta_true) + 1;
        end
        yCatch = zeros(nFixedCatch, 1);
        for t = 1:nFixedCatch
            x = randn();
            yCatch(t) = sum(x > beta_true) + 1;
        end

        % AUC d-prime
        FA  = [1; zeros(nCrit, 1); 0];
        Hit = [1; zeros(nCrit, 1); 0];
        for j = 1:nCrit
            FA(j+1)  = mean(yCatch  > j);
            Hit(j+1) = mean(ySignal > j);
        end
        AUC = abs(trapz(FA, Hit));
        AUC = min(max(AUC, 0.501), 0.999);
        d_auc_all(rep, k) = -2 * erfcinv(2 * AUC);

        % MLE d-prime (criteria as nuisance parameters)
        theta0_fixed = [1.0; session.beta_hat(1); log(diff(session.beta_hat))'];
        thFixed = fmincon(@(th) negLogLikFixed(th, ySignal, yCatch), ...
            theta0_fixed, [], [], [], [], [0; -Inf(nCrit, 1)], [], ...
            @(th) critBoundsFixed(th, cMax), opts);
        d_mle_all(rep, k) = thFixed(1);
    end

    fprintf('  Rep %3d/%d:  AB: A=%.3f b=%.3f  |  Fixed-b: b_used=%.3f Ith=%.3f\n', ...
        rep, nReps, session.A_hat, session.b_hat, b_fixed, Ithresh_hat);
end

%% ---- Summary tables ------------------------------------------------------

fprintf('\n=== AB parameterisation (%d reps) ===\n', nReps);
fprintf('Param   True    Mean     Bias    SD\n');
fprintf('--------------------------------------\n');
fprintf('A       %.3f   %.3f    %+.3f   %.3f\n', ...
    A_true, mean(A_hat_all), mean(A_hat_all) - A_true, std(A_hat_all));
fprintf('b       %.3f   %.3f    %+.3f   %.3f\n', ...
    b_true, mean(b_hat_ab_all), mean(b_hat_ab_all) - b_true, std(b_hat_ab_all));

fprintf('\n=== Fixed-b parameterisation (%d reps) ===\n', nReps);
fprintf('Param      True    Mean     Bias    SD\n');
fprintf('-----------------------------------------\n');
fprintf('b (fixed)  %.3f   %.3f    %+.3f   %.3f\n', ...
    b_true, mean(b_fixed_all), mean(b_fixed_all) - b_true, std(b_fixed_all));
fprintf('I_thresh   %.3f   %.3f    %+.3f   %.3f\n', ...
    Ithresh_true, mean(Ithresh_hat_all), mean(Ithresh_hat_all) - Ithresh_true, std(Ithresh_hat_all));

fprintf('\n=== Threshold intensity estimates ===\n');
hdr = '  %-14s  %7s  %8s  %8s  %6s  %8s  %8s  %6s\n';
sep = repmat('-', 1, 74);
fprintf(hdr, '', 'I_true', 'AB mean', 'AB bias', 'AB SD', 'FB mean', 'FB bias', 'FB SD');
fprintf('  %s\n', sep);
for k = 1:nDPrime
    fprintf('  d''=%.2f  (I)     %7.3f  %8.3f  %8+.3f  %6.3f  %8.3f  %8+.3f  %6.3f\n', ...
        dPrimeTargets(k), I_true(k), ...
        mean(I_hat_ab_all(:,k)), mean(I_hat_ab_all(:,k)) - I_true(k), std(I_hat_ab_all(:,k)), ...
        mean(I_hat_fb_all(:,k)), mean(I_hat_fb_all(:,k)) - I_true(k), std(I_hat_fb_all(:,k)));
end
fprintf('  %s\n', sep);
for k = 1:nDPrime
    fprintf('  d''=%.2f  (logI)  %7.3f  %8.3f  %8+.3f  %6.3f  %8.3f  %8+.3f  %6.3f\n', ...
        dPrimeTargets(k), log(I_true(k)), ...
        mean(log(I_hat_ab_all(:,k))), mean(log(I_hat_ab_all(:,k))) - log(I_true(k)), std(log(I_hat_ab_all(:,k))), ...
        mean(log(I_hat_fb_all(:,k))), mean(log(I_hat_fb_all(:,k))) - log(I_true(k)), std(log(I_hat_fb_all(:,k))));
end

fprintf('\n=== Fixed-intensity ROC validation — AB param (%d signal + %d catch per rep) ===\n', ...
    nFixedSignal, nFixedCatch);
hdr2 = '  %-14s  %7s  %8s  %8s  %6s  %8s  %8s  %6s\n';
sep2 = repmat('-', 1, 74);
fprintf(hdr2, '', 'd'' target', 'AUC mean', 'AUC bias', 'AUC SD', 'MLE mean', 'MLE bias', 'MLE SD');
fprintf('  %s\n', sep2);
for k = 1:nDPrime
    fprintf('  d''=%.2f          %7.3f  %8.3f  %8+.3f  %6.3f  %8.3f  %8+.3f  %6.3f\n', ...
        dPrimeTargets(k), dPrimeTargets(k), ...
        mean(d_auc_all(:,k)), mean(d_auc_all(:,k)) - dPrimeTargets(k), std(d_auc_all(:,k)), ...
        mean(d_mle_all(:,k)), mean(d_mle_all(:,k)) - dPrimeTargets(k), std(d_mle_all(:,k)));
end
fprintf('  %s\n', sep2);

%% ---- Figure --------------------------------------------------------------

colors = lines(nDPrime);

figure('Color', 'w');
tiledlayout(3, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% [1,1] b estimates: AB (estimated) vs fixed-b (drawn from calibration)
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

% [1,2] Free scale parameter: A_hat (AB) and I_thresh_hat (fixed-b)
nexttile;
histogram(A_hat_all,       12, 'FaceColor', [0.2 0.5 0.8], 'FaceAlpha', 0.6, ...
    'DisplayName', 'A_{hat} (AB)');
hold on;
histogram(Ithresh_hat_all, 12, 'FaceColor', [0.9 0.4 0.1], 'FaceAlpha', 0.6, ...
    'DisplayName', 'I_{thresh} (fixed-b)');
xline(A_true,       'b-', 'LineWidth', 2, 'HandleVisibility', 'off');
xline(Ithresh_true, 'r-', 'LineWidth', 2, 'HandleVisibility', 'off');
xlabel('Estimate');
title(sprintf('A_{hat} (true=%.2f)  |  I_{thresh} (true=%.2f)', A_true, Ithresh_true));
legend('Location', 'best');

% [2,1] I_hat from AB parameterisation
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

% [2,2] I_hat from fixed-b parameterisation
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

% [3,1] AUC d-prime estimates from fixed-intensity validation
nexttile;
hold on;
for k = 1:nDPrime
    histogram(d_auc_all(:,k), 10, 'FaceColor', colors(k,:), 'FaceAlpha', 0.6, ...
        'DisplayName', sprintf("d'=%.2f", dPrimeTargets(k)));
    xline(dPrimeTargets(k),          '-',  'Color', colors(k,:), 'LineWidth', 2,   'HandleVisibility', 'off');
    xline(mean(d_auc_all(:,k)), '--', 'Color', colors(k,:), 'LineWidth', 1.5, 'HandleVisibility', 'off');
end
xlabel("d' estimate");
title('AUC d-prime  (solid=target, dashed=mean)');
legend('Location', 'northeast');

% [3,2] MLE d-prime estimates from fixed-intensity validation
nexttile;
hold on;
for k = 1:nDPrime
    histogram(d_mle_all(:,k), 10, 'FaceColor', colors(k,:), 'FaceAlpha', 0.6, ...
        'DisplayName', sprintf("d'=%.2f", dPrimeTargets(k)));
    xline(dPrimeTargets(k),          '-',  'Color', colors(k,:), 'LineWidth', 2,   'HandleVisibility', 'off');
    xline(mean(d_mle_all(:,k)), '--', 'Color', colors(k,:), 'LineWidth', 1.5, 'HandleVisibility', 'off');
end
xlabel("d' estimate");
title('MLE d-prime  (solid=target, dashed=mean)');
legend('Location', 'northeast');

%% ---- Pack results --------------------------------------------------------

results.AB.A_hat    = A_hat_all;
results.AB.b_hat    = b_hat_ab_all;
results.AB.I_hat    = I_hat_ab_all;

results.FixedB.b_fixed     = b_fixed_all;
results.FixedB.Ithresh_hat = Ithresh_hat_all;
results.FixedB.I_hat       = I_hat_fb_all;

results.ROC.d_auc        = d_auc_all;
results.ROC.d_mle        = d_mle_all;
results.ROC.nFixedSignal = nFixedSignal;
results.ROC.nFixedCatch  = nFixedCatch;

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

function [c, ceq] = critBoundsFixedB(theta, cMax)
    beta = unpackCrit(theta(2:end));
    c    = beta(:) - cMax;
    ceq  = [];
end

function nll = negLogLikFixedB(theta, I, y, b_fixed, dPrimeRef)
    I_thresh  = exp(theta(1));
    beta      = unpackCrit(theta(2:end));
    R         = dPrimeRef .* (max(I, 0) ./ I_thresh) .^ b_fixed;
    R(I == 0) = 0;
    bounds    = [-Inf, beta, Inf];
    lo        = bounds(y);
    hi        = bounds(y + 1);
    p         = max(Phi(hi(:) - R) - Phi(lo(:) - R), realmin);
    nll       = -sum(log(p));
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
