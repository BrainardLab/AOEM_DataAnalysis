# CLAUDE.md — AOEM_DataAnalysis project

## Project overview
Simulation and analysis code for an adaptive optics eye-movement (AOEM) detection
experiment using rating-scale signal detection theory.

## Key file
`simulations/YesNoTSDSimulation.m` — the main simulation script (see below).

## Model
Power-law SDT with unit Gaussian noise:
- Response function: R = A * I^b  (responseFunction local fn)
- Rating rule: rating k iff beta_{k-1} < x <= beta_k, x = R + N(0,1)
- nResp = 6 rating categories, nCrit = 5 criteria
- Catch trials: I = 0, so x ~ N(0,1)

## Simulation pipeline (YesNoTSDSimulation.m)
1. **Staircase data collection** — interleaved staircases (BrainardLabToolbox
   Staircase object), cycling round-robin through staircases.  nUp/nDown vectors
   define each staircase's rule.  Staircase operates in log(I) space.
2. **MLE fitting** — fits [A, b, criteria] jointly via fmincon.
   - theta = [log(A), log(b), c1, log(c2-c1), ..., log(c5-c4)]
   - unpackCrit() reconstructs ordered criteria from this parameterisation
   - cMax = 3.5 caps criteria in R space (prevents runaway fits)
3. **Threshold estimation** — inverts fitted model to get I for target d-primes
4. **Fixed-intensity ROC verification** — simulates 150 signal + 75 catch trials
   at each estimated I_hat; compares three d-prime estimators:
   - Theoretical (true parameters)
   - AUC method: d' = -2 * erfcinv(2 * AUC)
   - MLE: negLogLikFixed fits d' + criteria jointly (criteria as nuisance)

## Staircase convention (BrainardLabToolbox)
- Staircase 'NUp'   = nDown(s)  — correct responses needed to step DOWN intensity
- Staircase 'NDown' = nUp(s)    — incorrect responses needed to step UP intensity
- Operates in log(I); getCurrentValue / updateForTrial work in linear I units

## Key parameters (current values as of 2026-04-08)
- A_true=1, b_true=1, Icrit_true=linspace(0.5,2,5), nStaircaseRespondNo=3
- nUp=[2,1,1,1], nDown=[1,1,2,3], nTrialsPerStaircase=50
- dPrimeTargets=[0.75,1,1.25], nFixedSignal=150, nFixedCatch=75

## Factorisation recommendations (pending, not yet implemented)
Functions worth extracting from the script:
1. simulateStaircaseSession(model, staircases, nTrials, pCatch, beta, threshold)
2. fitSDTModel(I, y, nCrit, cMax, opts)
3. simulateFixedIntensitySession(R_signal, beta, nSignal, nCatch)
4. computeRatingROC(ySignal, yCatch, nCrit)
5. estimateDPrimeAUC(FA, Hit)
6. estimateDPrimeMLE(ySignal, yCatch, beta_hat, cMax, opts)
Plot panels may also be worth extracting when real data analysis begins.

## Collaborators
David Brainard (DHB), HES, ClaudeAI — started 2026-04-08
