function results = checkLRTPower(nReps, sigma_z_vec, nTrials_vec)
% results = checkLRTPower(nReps, sigma_z_vec, nTrials_vec)
%
% Simulates power of a likelihood-ratio test for a trial-wise d-prime
% covariate.
%
% The experimenter measures a noise-free trial-wise covariate z_i (in
% d-prime units) reflecting additional variability in observer sensitivity.
% Trials are run at a fixed intensity corresponding to d-prime = 1 (the
% level with highest expected power), plus catch trials at pCatch = 0.20.
%
% Generative model (alternative, gamma_true = 1):
%   x_i = d_prime_true * signal_i + z_i + epsilon_i,  epsilon_i ~ N(0,1)
%   z_i ~ N(0, sigma_z)  for signal trials
%   z_i = 0              for catch trials
%
% Models fitted to each dataset:
%   Null        : (d_prime, criteria)         -- gamma fixed at 0
%   Alternative : (d_prime, criteria, gamma)  -- gamma free
%
% Test statistic:
%   LR = 2 * (nll_null - nll_alt)  ~  chi^2(1) under H0
%   Reject H0 if LR > 3.84  (alpha = 0.05)
%
% Common random numbers are used across sigma_z values at the same
% (nTrials, rep) to reduce Monte Carlo variance in power estimates.
%
% Inputs:
%   nReps       - Monte Carlo replications per condition (default: 200)
%   sigma_z_vec - covariate SDs in d-prime units  (default: [0.1 0.2 0.5 1.0])
%   nTrials_vec - trial counts to sweep           (default: [50 100 200 400 800 1600])
%
% Output:
%   results struct with fields:
%     power       - nSZ x nNT matrix of power estimates
%     LR          - nSZ x nNT x nReps array of LR statistics
%     sigma_z_vec, nTrials_vec, nReps, chi2_crit, alpha, d_prime_true, pCatch
%
% Example:
%   r = checkLRTPower(200, [0.1 0.2 0.5 1.0], [50 100 200 400 800 1600]);
%
% Requires: no toolboxes beyond base MATLAB and Optimization Toolbox.
%
% History:
%   2026-04-09  DHB, HES, ClaudeAI  wrote it.

if nargin < 1;  nReps       = 200;                        end
if nargin < 2;  sigma_z_vec = [0.1 0.2 0.5 1.0];         end
if nargin < 3;  nTrials_vec = [50 100 200 400 800 1600];  end

%% ---- True model parameters -----------------------------------------------

A_true       = 1.0;
b_true       = 0.8;
d_prime_true = 1.0;                               % fixed intensity targets d' = 1
Icrit_true   = linspace(0.3, 2, 5);
beta_true    = A_true .* (Icrit_true .^ b_true);  % true criteria in R space
nCrit        = numel(beta_true);
pCatch       = 0.20;
cMax         = 3.5;
chi2_crit    = 3.84;    % chi^2(1) critical value at alpha = 0.05
alpha        = 0.05;

opts = optimoptions('fmincon', 'Display', 'off', ...
    'MaxIterations', 5000, 'MaxFunctionEvaluations', 50000);

%% ---- Storage -------------------------------------------------------------

nSZ      = numel(sigma_z_vec);
nNT      = numel(nTrials_vec);
power_all = zeros(nSZ, nNT);
LR_all    = zeros(nSZ, nNT, nReps);

%% ---- Main loop -----------------------------------------------------------

fprintf('LRT power analysis: %d reps x %d sigma_z x %d nTrials...\n', ...
    nReps, nSZ, nNT);

c_init = linspace(0.5, 3.0, nCrit);
theta0 = [d_prime_true; c_init(1); log(diff(c_init))'];   % shared starting point

for iNT = 1:nNT
    nTrials = nTrials_vec(iNT);

    for rep = 1:nReps
        rng(rep);   % same draws for all sigma_z at this (nTrials, rep)

        % Trial structure
        isCatch  = rand(nTrials, 1) < pCatch;
        isSignal = ~isCatch;

        % Base random draws (shared across sigma_z)
        eps    = randn(nTrials, 1);   % response noise
        z_unit = randn(nTrials, 1);   % unit-normal covariate draws
        z_unit(isCatch) = 0;          % no covariate on catch trials

        for iSZ = 1:nSZ
            sigma_z = sigma_z_vec(iSZ);
            z       = sigma_z * z_unit;

            % Generate ratings under alternative model (gamma_true = 1)
            R = d_prime_true * double(isSignal) + z;
            x = R + eps;
            y = sum(x > beta_true(:)', 2) + 1;

            % Fit null model
            thetaNull = fmincon(@(th) negLogLikNull(th, y, isSignal), ...
                theta0, [], [], [], [], [], [], ...
                @(th) critBoundsNull(th, cMax), opts);
            nll_null = negLogLikNull(thetaNull, y, isSignal);

            % Fit alternative model, warm-started from null
            theta0_alt = [thetaNull; 0];
            thetaAlt   = fmincon(@(th) negLogLikAlt(th, y, isSignal, z), ...
                theta0_alt, [], [], [], [], [], [], ...
                @(th) critBoundsAlt(th, cMax), opts);
            nll_alt = negLogLikAlt(thetaAlt, y, isSignal, z);

            LR_all(iSZ, iNT, rep) = max(2 * (nll_null - nll_alt), 0);
        end
    end

    % Power for this nTrials
    for iSZ = 1:nSZ
        power_all(iSZ, iNT) = mean(squeeze(LR_all(iSZ, iNT, :)) > chi2_crit);
    end

    fprintf('  nTrials = %4d:', nTrials);
    for iSZ = 1:nSZ
        fprintf('  sz=%.2f: %.2f', sigma_z_vec(iSZ), power_all(iSZ, iNT));
    end
    fprintf('\n');
end

%% ---- Figure --------------------------------------------------------------

colors = lines(nSZ);
figure('Color', 'w');
hold on;
for iSZ = 1:nSZ
    plot(nTrials_vec, power_all(iSZ,:), 'o-', ...
        'Color', colors(iSZ,:), 'LineWidth', 2, ...
        'MarkerFaceColor', colors(iSZ,:), 'MarkerSize', 7, ...
        'DisplayName', sprintf('\\sigma_z = %.2f', sigma_z_vec(iSZ)));
end
yline(0.80, 'k--', 'LineWidth', 1.5, 'HandleVisibility', 'off');
yline(alpha,  'k:', 'LineWidth', 1.5, 'HandleVisibility', 'off');
text(nTrials_vec(end), 0.82, '0.80', 'FontSize', 9, 'HorizontalAlignment', 'right');
text(nTrials_vec(end), alpha + 0.02, sprintf('\\alpha=%.2f', alpha), ...
    'FontSize', 9, 'HorizontalAlignment', 'right');
set(gca, 'XScale', 'log');
xticks(nTrials_vec);
xticklabels(string(nTrials_vec));
xlabel('Number of trials');
ylabel('Power');
title(sprintf("LRT power: trial-wise d' covariate  (%d reps, d'_{true}=%.1f, p_{catch}=%.2f)", ...
    nReps, d_prime_true, pCatch));
legend('Location', 'southeast');
ylim([0 1]);
grid on;
box on;

%% ---- Pack results --------------------------------------------------------

results.power        = power_all;
results.LR           = LR_all;
results.sigma_z_vec  = sigma_z_vec;
results.nTrials_vec  = nTrials_vec;
results.nReps        = nReps;
results.chi2_crit    = chi2_crit;
results.alpha        = alpha;
results.d_prime_true = d_prime_true;
results.pCatch       = pCatch;

end

%% ======== Local functions =================================================

function [c, ceq] = critBoundsNull(theta, cMax)
    % theta = [d_prime, crit_params(1:nCrit)]
    beta = unpackCrit(theta(2:end));
    c    = beta(:) - cMax;
    ceq  = [];
end

function [c, ceq] = critBoundsAlt(theta, cMax)
    % theta = [d_prime, crit_params(1:nCrit), gamma]
    beta = unpackCrit(theta(2:end-1));
    c    = beta(:) - cMax;
    ceq  = [];
end

function nll = negLogLikNull(theta, y, isSignal)
    dPrime = theta(1);
    beta   = unpackCrit(theta(2:end));
    R      = dPrime * double(isSignal);   % 0 on catch trials
    bounds = [-Inf, beta, Inf];
    lo     = bounds(y);
    hi     = bounds(y + 1);
    p      = max(Phi(hi(:) - R) - Phi(lo(:) - R), realmin);
    nll    = -sum(log(p));
    if ~isfinite(nll);  nll = realmax;  end
end

function nll = negLogLikAlt(theta, y, isSignal, z)
    dPrime = theta(1);
    gamma  = theta(end);
    beta   = unpackCrit(theta(2:end-1));
    R      = dPrime * double(isSignal) + gamma * z;   % z=0 on catch trials
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
    for k   = 2:numel(v)
        beta(k) = beta(k-1) + exp(v(k));
    end
end

function p = Phi(z)
    p = 0.5 * erfc(-z ./ sqrt(2));
end
