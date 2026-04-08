function results = runYesNoTSDSession(params)
% results = runYesNoTSDSession(params)
%
% Runs one session of the YesNoTSD staircase experiment and fits the AB model.
% Called by YesNoTSDSimulation.m (for a single interactive run) and by
% checkEstimationBias.m (in a Monte Carlo loop).
%
% Input:
%   params - struct with the following fields:
%
%     True model
%       A_true              - response-function gain
%       b_true              - response-function exponent (R = A * I^b)
%       Icrit_true          - criteria in intensity space (1 x nCrit)
%       nStaircaseRespondNo - ratings > this count as "yes"
%       dPrimeTargets       - target d-prime values for threshold estimation
%
%     Staircase (common)
%       staircaseType       - 'standard' or 'quest'
%       pCatch              - fraction of catch trials (I = 0)
%       I0, Imin, Imax      - initial / bounds on intensity (linear)
%       nTrialsPerStaircase - trials per staircase
%
%     Standard staircase only
%       nUp, nDown          - step-up / step-down rules (scalar or vector)
%       stepSizes           - step sizes in log(I) space
%
%     QUEST only
%       questTargetProbs    - target P(yes) per interleaved QUEST
%       questBeta           - Weibull slope
%       questDelta          - lapse rate
%       questGamma          - guess rate
%       questPriorSD        - prior SD in linear I (log10(questPriorSD) used internally)
%
%     Fitting
%       cMax                - upper bound on criteria in R space
%       fitDisplay          - fmincon 'Display' option ('iter' or 'off')
%
% Output:
%   results struct:
%     I, y, sIdx      - trial intensities, ratings, staircase indices
%     staircases      - cell array of staircase objects after the session
%     nStaircases     - number of interleaved staircases used
%     A_hat, b_hat    - fitted model parameters
%     beta_hat        - fitted criteria (in R space)
%     Icrit_hat       - fitted criteria (in I space)
%     thetaHat        - full fmincon parameter vector (for re-use in callers)
%     I_hat           - estimated threshold intensities for dPrimeTargets
%
% History:
%   2026-04-08  DHB, HES, ClaudeAI  extracted from YesNoTSDSimulation.m

%% ---- Unpack params -------------------------------------------------------

A_true              = params.A_true;
b_true              = params.b_true;
beta_true           = responseFunction(params.Icrit_true, [A_true, b_true]);
nCrit               = numel(beta_true);
nStaircaseRespondNo = params.nStaircaseRespondNo;
dPrimeTargets       = params.dPrimeTargets;

staircaseType = params.staircaseType;
pCatch        = params.pCatch;
I0            = params.I0;
Imin          = params.Imin;
Imax          = params.Imax;

if strcmp(staircaseType, 'standard')
    nStaircases = numel(params.nDown);
else
    nStaircases = numel(params.questTargetProbs);
end
nTrials = params.nTrialsPerStaircase * nStaircases;

cMax       = params.cMax;
fitDisplay = params.fitDisplay;

%% ---- Initialise staircases -----------------------------------------------

staircases = cell(nStaircases, 1);
for s = 1:nStaircases
    if strcmp(staircaseType, 'standard')
        % Standard staircase: pass log(I) values; steps are additive in log(I)
        % = multiplicative in I.
        staircases{s} = Staircase('standard', log(I0), ...
            'StepSizes', params.stepSizes,  ...
            'NUp',       params.nDown(s),   ...   % correct responses to step down
            'NDown',     params.nUp(s),     ...   % incorrect responses to step up
            'MaxValue',  log(Imax),         ...
            'MinValue',  log(Imin));
    else
        % QUEST: handles log10(I) conversion internally.
        % getCurrentValue returns linear I; updateForTrial takes linear I.
        staircases{s} = Staircase('quest', I0,                         ...
            'Beta',            params.questBeta,            ...
            'Delta',           params.questDelta,           ...
            'Gamma',           params.questGamma,           ...
            'TargetThreshold', params.questTargetProbs(s),  ...
            'PriorSD',         params.questPriorSD,         ...
            'MaxValue',        Imax,                        ...
            'MinValue',        Imin);
    end
end

%% ---- Simulate trial sequence ---------------------------------------------

I    = zeros(nTrials, 1);
y    = zeros(nTrials, 1);
sIdx = zeros(nTrials, 1);
signalCount = 0;

for t = 1:nTrials

    if rand < pCatch
        % Catch trial: no signal, no staircase update.
        I(t)    = 0;
        sIdx(t) = 0;
    else
        % Signal trial: select next staircase in round-robin order.
        signalCount = signalCount + 1;
        s           = mod(signalCount - 1, nStaircases) + 1;
        sIdx(t)     = s;

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

    % Catch trials get a rating too (used in MLE fitting and ROC analysis).
    if I(t) == 0
        x    = responseFunction(0, [A_true, b_true]) + randn();
        y(t) = sum(x > beta_true) + 1;
    end
end

%% ---- Fit AB model by MLE -------------------------------------------------
%
% theta = [log(A), log(b), c1, log(c2-c1), ..., log(c_nCrit - c_{nCrit-1})]
% A and b are reparameterised as exp(.) to enforce positivity.
% Criteria are encoded via unpackCrit to enforce strict ordering.
% cMax caps criteria in R space to prevent runaway values.

c_init   = linspace(0.5, 3.0, nCrit);
theta0   = [log(1); log(1); c_init(1); log(diff(c_init))'];

opts = optimoptions('fmincon', 'Display', fitDisplay, ...
    'MaxIterations', 5000, 'MaxFunctionEvaluations', 50000);

thetaHat = fmincon(@(th) negLogLikAB(th, I, y), theta0, ...
    [], [], [], [], [], [], @(th) critBounds(th, cMax), opts);

A_hat     = exp(thetaHat(1));
b_hat     = exp(thetaHat(2));
beta_hat  = unpackCrit(thetaHat(3:end));
Icrit_hat = (beta_hat / A_hat).^(1 / b_hat);
I_hat     = (dPrimeTargets / A_hat).^(1 / b_hat);

%% ---- Pack results --------------------------------------------------------

results.I           = I;
results.y           = y;
results.sIdx        = sIdx;
results.staircases  = staircases;
results.nStaircases = nStaircases;
results.A_hat       = A_hat;
results.b_hat       = b_hat;
results.beta_hat    = beta_hat;
results.Icrit_hat   = Icrit_hat;
results.thetaHat    = thetaHat;
results.I_hat       = I_hat;

end

%% ======== Local functions =================================================

function [c, ceq] = critBounds(theta, cMax)
    beta = unpackCrit(theta(3:end));
    c    = beta(:) - cMax;
    ceq  = [];
end

function nll = negLogLikAB(theta, I, y)
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
    R = theta(1) .* (I .^ theta(2));
end
