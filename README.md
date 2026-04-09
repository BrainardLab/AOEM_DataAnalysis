# AOEM_DataAnalysis

Simulation and analysis code for an adaptive optics eye-movement (AOEM) detection
experiment using rating-scale signal detection theory (SDT).

---

## Setup

### Required toolboxes

| Toolbox | Used for |
|---|---|
| BrainardLabToolbox | `Staircase` class (adaptive staircases) |
| Psychtoolbox-3 | `QuestCreate` (required by the `'quest'` staircase type) |
| Optimization Toolbox | `fmincon` (MLE fitting, all scripts) |

### Starting a MATLAB session

Run `initSession` once at the start of each MATLAB session before calling any
simulation script:

```matlab
initSession
```

This loads BrainardLabToolbox and Psychtoolbox-3 via ToolboxToolbox.

> **Note:** If you switch between `'standard'` and `'quest'` staircase types
> in the same session, you may need to run `clear classes` first. This is a
> limitation of BrainardLabToolbox's `Staircase` class.
>
> `checkLRTPower` does **not** require `initSession` — it uses only base
> MATLAB and the Optimization Toolbox.

---

## Model

Power-law SDT with unit Gaussian noise and a 6-category rating scale:

- Response function: `R(I) = A * I^b`
- Internal response: `x = R(I) + epsilon`, `epsilon ~ N(0,1)`
- Rating rule: `y = k` iff `beta_{k-1} < x <= beta_k` (5 criteria, 6 categories)
- Catch trials: `I = 0`, so `x ~ N(0,1)`

True parameter values used throughout: `A = 1`, `b = 0.8`,
`Icrit = linspace(0.3, 2, 5)`, `pCatch = 0.20`.

---

## Scripts

### 1. `YesNoTSDSimulation(staircaseType)`

**What it does.** Runs a single simulated staircase session, fits the SDT
model by MLE, estimates threshold intensities, and verifies them with a
fixed-intensity ROC simulation.

**How to run.**

```matlab
initSession
YesNoTSDSimulation            % standard staircase (default)
YesNoTSDSimulation('quest')   % QUEST staircase
```

**What to look at.**

- *Figure 1, Panel 1 — Staircase trajectories:* Check that the staircase(s)
  converge to sensible intensity levels. For `'standard'` (nDown = [2, 5, 9])
  the three staircases should settle at different levels. For `'quest'` they
  should converge quickly to their respective P(yes) targets.

- *Figure 1, Panel 2 — Response function:* The fitted curve (red dashed)
  should track the true curve (black solid). The criteria (vertical dotted
  lines) may be noisily estimated, especially at the extremes.

- *Figure 1, Panel 3 — P(yes):* The fitted psychometric function should agree
  with the empirical trial outcomes.

- *Printed table — True vs estimated parameters:* Check `A_hat` and `b_hat`
  relative to their true values (1.0 and 0.8). Criteria estimates are often
  biased, particularly `c1` and `c5`, because the staircase concentrates data
  near one intensity level.

- *Printed table — Target d-prime intensities:* `I_hat` should be near
  `I_true` but will typically show bias (see `checkEstimationBias` for a
  systematic assessment).

- *Figure 2 — ROC curves:* Each panel shows empirical ROC points, the
  theoretical curve, and AUC/MLE d-prime estimates at the fitted threshold
  intensity. Discrepancies between the target d-prime and the "true d-prime
  at I_hat" reflect the threshold estimation bias.

---

### 2. `checkEstimationBias(nReps, b_noise_sd, staircaseType)`

**What it does.** Runs `nReps` independent staircase sessions and quantifies
estimation bias for two fitting approaches:

| Approach | Description |
|---|---|
| **AB** | Fits `A`, `b`, and criteria freely (standard MLE) |
| **Fixed-b** | Fixes `b` at `b_true + N(0, b_noise_sd)`, fits only `I_thresh` |

Also runs a fixed-intensity ROC validation at each estimated threshold to
check whether the estimated intensities actually deliver the target d-prime.

**How to run.**

```matlab
initSession
r_std   = checkEstimationBias(50, 0.1, 'standard');

clear classes; initSession   % needed when switching staircase types
r_quest = checkEstimationBias(50, 0.1, 'quest');
```

**Key parameters.**

| Parameter | Default | Meaning |
|---|---|---|
| `nReps` | 50 | Number of Monte Carlo replications |
| `b_noise_sd` | 0.1 | SD of calibration error on `b` for the Fixed-b fit |
| `staircaseType` | `'standard'` | `'standard'` or `'quest'` |

Both types use **3 interleaved staircases × 50 trials = 150 total** signal trials:

- Standard: `nDown = [2, 5, 9]`, targeting P(yes) ≈ [0.62, 0.76, 0.82]
- QUEST: targets P(yes) = [0.60, 0.75, 0.90]

**What to look at.**

- *Printed tables — AB and Fixed-b parameterisation:* Mean, bias, and SD of
  `A_hat`, `b_hat`, and `I_thresh_hat`. Both approaches typically show positive
  bias in threshold estimates because the staircase concentrates data near one
  intensity level, leaving extreme criteria poorly constrained.

- *Printed tables — Threshold intensity estimates (I) and (log I):* Bias and
  SD of `I_hat` for each target d-prime, for both AB and Fixed-b fits. Fixing
  `b` does not substantially reduce bias — the problem is mainly in the criteria
  estimation, not the A-b parameterisation.

- *Printed table — Fixed-intensity ROC validation:* AUC and MLE d-prime
  estimates when data are actually collected at `I_hat`. These should match the
  target d-prime if `I_hat` is unbiased; positive bias in `I_hat` produces
  inflated d-prime estimates here.

- *Figure — histograms:*
  - Row 1: distributions of `b_hat` and `A_hat` / `I_thresh_hat`
  - Row 2: distributions of `I_hat` for AB and Fixed-b
  - Row 3: distributions of AUC and MLE d-prime from the ROC validation

- *Comparing `'standard'` vs `'quest'`:* Standard staircases typically show
  less bias. QUEST converges quickly and concentrates trials tightly around
  its target intensities, giving less coverage of the intensity range and
  poorer constraint on extreme criteria. The standard staircase's slower
  convergence inadvertently spreads data more broadly, which helps.

---

### 3. `checkLRTPower(nReps, sigma_z_vec, nTrials_vec)`

**What it does.** Estimates the statistical power of a likelihood-ratio test
(LRT) for detecting a trial-wise d-prime covariate.

The experiment is motivated by the idea that the experimenter has a noise-free
trial-by-trial estimate `z_i` of additional variability in the observer's
sensitivity. The question is: how many trials are needed to detect that `z_i`
improves the model?

**Generative model.**

```
x_i = d_prime_true * signal_i + z_i + epsilon_i,   epsilon_i ~ N(0,1)
z_i ~ N(0, sigma_z)   for signal trials   (experimenter measures z_i exactly)
z_i = 0               for catch trials
```

**Models compared.**

| Model | Parameters | Description |
|---|---|---|
| Null | `d_prime`, criteria | `gamma` fixed at 0 |
| Alternative | `d_prime`, criteria, `gamma` | `gamma` free |

Test statistic: `LR = 2 * (nll_null - nll_alt) ~ chi^2(1)` under H0.
Reject H0 if `LR > 3.84` (alpha = 0.05).

**How to run.** (No `initSession` needed.)

```matlab
r = checkLRTPower(200, [0.1 0.2 0.5 1.0], [50 100 200 400 800 1600]);
```

**Key parameters.**

| Parameter | Default | Meaning |
|---|---|---|
| `nReps` | 200 | Monte Carlo replications per condition |
| `sigma_z_vec` | `[0.1 0.2 0.5 1.0]` | Covariate SDs in d-prime units |
| `nTrials_vec` | `[50 100 200 400 800 1600]` | Trial counts to sweep |

Trials are fixed at the true threshold intensity (d-prime = 1), with
`pCatch = 0.20`. The 6-category rating scale is used throughout.

**What to look at.**

- *Power curve:* Power vs number of trials, one curve per `sigma_z`. The
  dashed line marks 0.80 power (a conventional target); the dotted line marks
  alpha = 0.05 (the expected false-positive rate when sigma_z = 0).

- *Effect of sigma_z:* Larger `sigma_z` means more trial-by-trial variability
  and therefore an easier detection problem. Read off the trial count needed
  to reach 0.80 power for each `sigma_z` of interest.

- *Checking type I error:* To verify the test is correctly calibrated, run
  with `sigma_z = 0` included (generates data under the null). The power
  should be near 0.05.

```matlab
r0 = checkLRTPower(500, [0 0.1 0.2 0.5 1.0], [200 400 800]);
```

---

## File overview

```
initSession.m                     Session setup (run once per MATLAB session)
simulations/
  YesNoTSDSimulation.m            Single staircase session + ROC verification
  runYesNoTSDSession.m            Shared session function (called internally)
  checkEstimationBias.m           Monte Carlo bias assessment
  checkLRTPower.m                 LRT power analysis for trial-wise covariate
```

## Collaborators

David Brainard (DHB), HES, ClaudeAI — started 2026-04-08
