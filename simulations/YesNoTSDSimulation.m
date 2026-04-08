% simulate_fit_tsd.m
% Simulate yes/no data from a signal detection model,
% fit A, b, and beta by MLE, and compare true vs estimated values.

clear; clc; close all;
%rng(1);

%% True parameters
A_true    = 3.0;
b_true    = 0.8;
beta_true = 1.2;

%% Simulation settings
nTrials   = 1000;
pCatch    = 0.20;     % fraction of catch trials at I = 0

% 2-up / 1-down staircase settings for signal-present trials
I0         = 1.0;     % starting intensity
Imin       = 0.01;
Imax       = 10.0;
stepFactor = 1.15;    % multiplicative step size

%% Simulate trial sequence
I = zeros(nTrials,1);      % presented intensity on each trial
y = zeros(nTrials,1);      % response: 1 = yes, 0 = no

currentI = I0;
correctStreak = 0;

for t = 1:nTrials

    if rand < pCatch
        % Catch trial: intensity = 0, but do not update staircase
        I(t) = 0;

        % Internal response and yes/no decision
        pYes = Phi(A_true * (I(t)^b_true) - beta_true);
        y(t) = rand < pYes;

    else
        % Signal-present trial at current staircase intensity
        I(t) = currentI;

        pYes = Phi(A_true * (I(t)^b_true) - beta_true);
        y(t) = rand < pYes;

        % 2-up / 1-down rule:
        % on signal-present trials, "yes" is correct
        if y(t) == 1
            correctStreak = correctStreak + 1;
            if correctStreak >= 2
                currentI = max(Imin, currentI / stepFactor);
                correctStreak = 0;
            end
        else
            currentI = min(Imax, currentI * stepFactor);
            correctStreak = 0;
        end
    end
end

%% Fit the model by MLE
% Reparameterize A and b to keep them positive:
% A = exp(a), b = exp(c), beta free.
theta0 = [log(1); log(1); 0];   % initial guess: [a, c, beta]

opts = optimset('Display','iter', 'MaxIter', 5000, 'MaxFunEvals', 20000);

thetaHat = fminsearch(@(th) negLogLik(th, I, y), theta0, opts);

A_hat    = exp(thetaHat(1));
b_hat    = exp(thetaHat(2));
beta_hat = thetaHat(3);

%% Compare true vs estimated values
fprintf('\nTrue vs estimated parameters\n');
fprintf('-----------------------------------------\n');
fprintf('A     true = %10.4f   hat = %10.4f\n', A_true, A_hat);
fprintf('b     true = %10.4f   hat = %10.4f\n', b_true, b_hat);
fprintf('beta  true = %10.4f   hat = %10.4f\n', beta_true, beta_hat);

% Optional comparison at criterion point P(yes)=0.5
if beta_true > 0
    I50_true = (beta_true / A_true)^(1 / b_true);
else
    I50_true = NaN;
end

if beta_hat > 0
    I50_hat = (beta_hat / A_hat)^(1 / b_hat);
else
    I50_hat = NaN;
end
fprintf('I50   true = %10.4f   hat = %10.4f\n', I50_true, I50_hat);

%% Plot fitted vs true psychometric function and staircase trajectory
Igrid = linspace(0, max(max(I), Imax) * 1.05, 300)';
pTrue = Phi(A_true * (Igrid.^b_true) - beta_true);
pHat  = Phi(A_hat  * (Igrid.^b_hat)  - beta_hat);

figure('Color','w');
tiledlayout(2,1);

nexttile;
plot(Igrid, pTrue, 'k-', 'LineWidth', 2); hold on;
plot(Igrid, pHat,  'r--', 'LineWidth', 2);

% Jitter observed responses vertically just for display
yj = y + 0.04 * (rand(size(y)) - 0.5);
scatter(I, yj, 18, 'filled', 'MarkerFaceAlpha', 0.25);

ylim([-0.1, 1.1]);
xlabel('Intensity');
ylabel('P(yes)');
title('True and fitted psychometric functions');
legend('True', 'Fitted', 'Observed trials', 'Location', 'southeast');

nexttile;
plot(1:nTrials, I, 'b-');
xlabel('Trial');
ylabel('Intensity used');
title('2-up / 1-down staircase trajectory');

%% ---- Local functions ----

function nll = negLogLik(theta, I, y)
    % Negative log-likelihood for the probit SDT model:
    % p_i = Phi(A * I_i^b - beta)
    % with A = exp(theta(1)), b = exp(theta(2))

    A    = exp(theta(1));
    b    = exp(theta(2));
    beta = theta(3);

    eta = A .* (I.^b) - beta;

    % Stable log-likelihood:
    % log Phi(eta) and log(1 - Phi(eta)) = log Phi(-eta)
    ll = sum(y .* logPhi(eta) + (1 - y) .* logPhi(-eta));

    nll = -ll;

    if ~isfinite(nll)
        nll = realmax;
    end
end

function lp = logPhi(z)
    % Stable log(Phi(z)) using erfc and log1p
    lp = zeros(size(z));

    idx = (z < 0);

    % For z < 0: Phi(z) = 0.5 * erfc(-z / sqrt(2))
    lp(idx) = log(0.5) + log(erfc(-z(idx) ./ sqrt(2)));

    % For z >= 0: Phi(z) = 1 - 0.5 * erfc(z / sqrt(2))
    lp(~idx) = log1p(-0.5 * erfc(z(~idx) ./ sqrt(2)));
end

function p = Phi(z)
    % Standard normal CDF without requiring toolboxes
    p = 0.5 * erfc(-z ./ sqrt(2));
end