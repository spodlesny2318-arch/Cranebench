# cranebench: a reproducible benchmark for the control of underactuated crane systems

**Oleksii Sheremet**^a,\*, **Serhii Podliesnyi**^b

^a Department of Electromechanical Systems of Automation and Electric Drive, Donbas State Engineering Academy, 39 Mashinobudivnykiv Blvd., Kramatorsk, 84313, Ukraine
^b Department of Fundamentals of Machine Design, Donbas State Engineering Academy, Kramatorsk, Ukraine
\* Corresponding author: Oleksii.Sheremet@ddma.edu.ua

---

## Abstract

Crane control results are rarely comparable across papers: each study defines its own plant, manoeuvre, disturbance, uncertainty set and metrics, then reports that the proposed method beats baselines the same authors implemented. `cranebench` fixes the bench instead of the method. It supplies three cross-validated plants, two spectrally exact disturbance models, a paired Latin-hypercube design, a hash-frozen metric module, a provenance ledger and five classical baselines — and deliberately no novel controller. A user subclasses one interface and receives a paired contrast. Six campaigns show that the ranking of the baselines is not preserved between operating points, and that this survives re-tuning every one of them on the operating point where it is measured.

**Keywords:** crane control; underactuated systems; reproducibility; benchmark; Monte Carlo; paired experimental design

---

## Metadata

| Nr | Code metadata description | Metadata |
|----|---------------------------|----------|
| C1 | Current code version | v1.0.0 |
| C2 | Permanent link to code/repository used for this code version | *(mandatory GitHub URL — to be inserted)* |
| C3 | Legal code license | BSD-3-Clause |
| C4 | Code versioning system used | git |
| C5 | Software code languages, tools and services used | Python (≥ 3.10), NumPy, SciPy; SymPy for the model derivations; Matplotlib for the figures |
| C6 | Compilation requirements, operating environments and dependencies | None; pure Python. `pip install -e .` installs NumPy ≥ 1.24 and SciPy ≥ 1.10. Tested on Linux with CPython 3.10 |
| C7 | If available, link to developer documentation/manual | `README.md` and `docs/DESIGN.md` in the repository |
| C8 | Support email for questions | Oleksii.Sheremet@ddma.edu.ua |

## 1. Motivation and significance

Anti-sway control of underactuated cranes is a mature field with a large and still growing literature [1--4], spanning input shaping [5--9], energy- and passivity-based design [10, 11], model predictive control [12, 13], sliding-mode and hierarchical sliding-mode schemes [14--17], and learning-based controllers. It is also a field in which two papers published in the same year are, in practice, almost never comparable.

The reason is structural rather than cultural. A crane control paper must specify a plant (planar or spatial; point-mass or rigid-body payload; rigid or elastic rope), a manoeuvre (distance, ramp time, whether the hoist moves), a disturbance (absent, harmonic, or a turbulence spectrum with its own parameters), an uncertainty set (which parameters vary and over what range), a set of metrics, and an integrator. Each of these is a legitimate modelling choice, and each is normally made afresh. The consequence is that when a new method reports, say, a 4.4-fold reduction in integral sway error against an LQR baseline, a reader cannot tell how much of that factor is the method, how much is the particular LQR weights the authors chose, and how much is a manoeuvre or disturbance that happens to suit the proposed structure.

Two further failure modes are specific to simulation-only work, which most of this literature is. First, the baselines are implemented by the proposers of the new method, who have every incentive — usually unconscious — to spend more tuning effort on their own entry. Second, the metric definitions themselves are rarely frozen: nothing in a published paper allows a reader to verify that the numbers reported for the baseline and for the proposed method were produced by the same scoring code.

`cranebench` addresses these by inverting what is held fixed. It supplies the plants, disturbances, uncertainty design, metrics and statistics, and supplies **no novel controller at all**. Its baselines are five textbook designs. A researcher proposing a new controller subclasses one interface, runs the same design against the same seeds, and reports a paired contrast. What varies is then the controller, and only the controller.

General-purpose multibody engines will of course simulate a crane, and several published crane models are distributed as simulator demonstrations. They are not a substitute for what is proposed here, because the reproducibility problem in this literature is not the equations of motion — it is that the manoeuvre, the disturbance realisation, the uncertainty set, the scoring code and the statistical contrast are reinvented for every paper. A simulation engine fixes none of those; this package fixes all of them and deliberately leaves the controller free.

Related benchmarking efforts exist in adjacent areas — the Tennessee Eastman challenge problem in process control [18], and the reproducibility critique and attendant suites in reinforcement learning [19] — but the crane community has no shared bench, and its distinctive features (underactuation with a physically meaningful safety bound on an unactuated coordinate, strongly non-stationary disturbance, and parameters that vary by design between lifts) are not covered by any of them.

## 2. Software description

### 2.1 Software architecture

The package is organised around the rule that a controller may read the state and return an input, and may do nothing else. Plants, disturbances, the uncertainty design and the metrics are outside the controller's reach.

**Plants.** Three are supplied, following the standard formulations [20, 21]: a planar trolley/hoist/payload crane (6 states, 2 inputs); a three-dimensional crane with a spherical-pendulum suspension and a payload yaw coordinate restrained by the gravitational torsional stiffness of a two-fall suspension and driven by the yaw moment of an eccentric centre of pressure (12 states, 3 inputs), using a parameterisation whose rope-length norm is exact at all swing angles; and a cooperative dual crane carrying a rigid beam on two visco-elastic falls (10 states, 2 inputs), in the lineage of the dual-crane models of [22--24], modelled with spring-damper falls rather than holonomic constraints so that the system remains an ODE and load-sharing dynamics stay visible.

Plants are declared kinematically — where the point masses are as a function of the generalised coordinates, plus an additive generalised inertia, a potential and the generalised forces — and the equations of motion are assembled numerically. Hand-derived spatial crane models are a known source of silent algebra errors; assembling them makes the model checkable, because the same declaration can be handed to an independent derivation and the two compared. Every plant here is checked that way: the planar equations against the assembler, and the spatial and dual equations against symbolic derivations produced by SymPy's `LagrangesMethod`, which shares no code with the assembler (Table 1). The verified symbolic result is then emitted as a closed-form fast path, fifteen times quicker than the assembler and required by the test suite to reproduce it; for the dual plant that path is used only while both falls carry tension, and the assembler takes over the moment one goes slack, because the symbolic derivation assumes the smooth branch of a unilateral contact law. Kinematic Jacobians use the complex-step derivative [25], `J = Im[p(q + ih e_k)]/h`. This is not a refinement: the Coriolis term differentiates the mass matrix, so a central-difference Jacobian is differentiated twice and its noise floor surfaces in the accelerations at 10⁻⁴, which is what agreement between the hand-derived and assembled planar models measured before the change and 6.1·10⁻¹⁰ after it.

**Disturbances.** Kaimal turbulence [26, 27] is synthesised by spectral representation with random phases [28] and rescaled so that the realised variance is exact — without which two controllers on "the same seed" would in fact see disturbances of different strength. Dryden turbulence [29] is realised as a rational shaping filter discretised by Van Loan's method [30], started from its stationary distribution, and has unbounded support, so any never-exceed claim evaluated against it is necessarily probabilistic. Both records live on a fixed 100 Hz grid and are interpolated, **independently of the integrator step**: a record indexed by the solver step changes when the step changes, so a step-refinement study would measure a different disturbance at every step and could never converge.

**Baselines.** PD, LQR on the numerically linearised plant, a ZVD input shaper [5, 6] in cascade with PD tracking, boundary-layer sliding-mode control [14, 31], and hierarchical sliding-mode control [17]. All gains are exposed and documented. The switching function is `tanh(s/φ)` with `φ` reported: with `sign`, the measured effort of a sliding controller is a function of the integrator step rather than of the design.

**Batched execution.** A second path integrates the whole Monte Carlo ensemble at once — the state is an `(N, n_x)` array and one RK4 step advances every sample together — completing 500 runs of one controller in 5.5 s against 0.8 s per single run on the scalar reference path. A second implementation of the same experiment is a liability unless it is pinned to the first, so per-sample setup that could be re-derived incorrectly is obtained by calling the *scalar* code, and `tests/test_batch.py` runs a full paired design through both paths and requires agreement on every metric, observed at 1.4·10⁻¹⁴.

A new controller is added by implementing two methods and nothing else:

```python
from cranebench.controllers import Controller

class MyController(Controller):
    name = "mine"

    def reset(self, plant, manoeuvre):
        super().reset(plant, manoeuvre)     # plant dimensions, nominal params
        ...                                 # per-run setup: gains, filters

    def __call__(self, t, x):
        return u                            # the input vector; x must not be mutated
```

The runner then replays the same design, the same per-sample wind seeds and the same frozen metric module that produced every baseline row in Section 3, and returns a paired contrast.

**Uncertainty design.** A centred Latin hypercube [32] over five multiplicative factors (payload mass ±20 %, rope length ±25 %, swing damping ×0.5–2, drive damping ±30 %, mean wind ±40 %) plus a per-sample wind seed. The sample list is drawn once and replayed for every controller, so contrasts are taken sample by sample and plant variation cancels rather than adding to the noise.

**Metrics.** A single module computes every reported number and hashes its own source at import; the hash is written into every result file. The common failure in simulation-only work is not fabricated data but a metric that quietly changed between the baseline and proposed-method runs, and freezing the module makes that detectable by a reviewer who never executes the code.

*(Figure 1: the harness. Everything shaded blue is fixed by the benchmark; the controller is the only user-supplied component, and it sees the state and the reference and nothing else.)*

**Provenance.** A ledger records the package version, the metric hash, a hash of every source file, the design seed, every per-run wind seed, the integrator and step, and the interpreter and library versions.

### 2.2 Software functionalities

- Eleven metrics per run, defined once and applied identically to every controller: **ISE**, the integral of squared horizontal position error; **settling time**, the first instant after which the position error stays inside a 2 cm band for the rest of the run, reported as right-censored at the horizon when that never happens; **peak** and **RMS swing** over the run and **residual swing**, the RMS swing over the final 5 s; **peak** and **RMS yaw**; **effort**, the integral of u'u over the horizontal channels; **peak input**; **command total variation** ("chatter"), the summed absolute increment of the held horizontal command; and **bound satisfaction**, whether peak swing stayed within a declared limit.
- Paired statistics: percentile bootstrap confidence intervals [33] on the sample-by-sample difference, Wilcoxon signed-rank tests [34], matched-pairs rank-biserial correlation as the effect size consistent with that test, McNemar's test for the binary bound-satisfaction outcome, win rates, and a running-mean convergence diagnostic. Because a campaign produces of order 10² paired tests, and because at *n* = 500 almost any non-zero difference reaches significance, effect sizes are the primary reported quantity and *p*-values are floored at 10⁻¹⁰: the normal approximation to the signed-rank statistic carries no meaning further into the tail, and the point null of an exactly zero shift is known to be false in any case.
- Campaign checkpointing to JSONL after every run, with transparent resumption and an optional wall-clock budget, so an evicted or sliced campaign never restarts from zero.
- A 17-test suite that verifies the models rather than asserting them (Section 3.1).

Effort is integrated over the **horizontal** channels only. The hoist channel carries the static weight — 39.2 kN at the default parameters — so including it makes the effort of every controller equal to (mg)²T to three significant figures and destroys the comparison. Command total variation is reported separately because a boundary-layer sliding controller can buy a small tracking error with a command no drive will accept, and effort alone will not reveal it.

## 3. Illustrative examples

### 3.1 Verification of the bench

| Check | Result |
|---|---|
| Hand-derived planar equations vs. assembled Lagrangian model, 40 random states | max relative difference 6.1·10⁻¹⁰ |
| Spatial plant: assembled model vs. independent SymPy derivation, 25 random states | mass matrix 4.8·10⁻¹⁶, accelerations 1.1·10⁻⁹ |
| Dual plant: assembled model vs. independent SymPy derivation, taut branch, 25 states | mass matrix 2.9·10⁻¹², accelerations 1.6·10⁻⁶ |
| Energy conservation, damping removed, 3 s, planar / spatial / dual | 8.4·10⁻¹⁵ / 7.4·10⁻¹⁵ / 1.8·10⁻¹⁴ relative |
| Mass matrices symmetric positive definite, all plants | passes |
| Spherical-pendulum map preserves rope length | exact to 10⁻¹² at 1.2 rad swing |
| Kaimal realised variance | exact to 10⁻⁹ |
| Kaimal realised spectrum vs. target, 0.01–5 Hz | log-PSD correlation r = 0.998 (Fig. 2a) |
| Dryden stationarity from first sample | within 12 % over 400 seeds |
| Metric convergence, dt from 10⁻³ to 10⁻² s | all metrics within 2.5·10⁻⁴ relative |
| Batched path vs. scalar reference path, all metrics, full design | max relative difference 1.4·10⁻¹⁴ |
| Bitwise reproducibility of a repeated run | exact |
| Independent re-run on a second platform: Windows 11 / CPython 3.14 / NumPy 2.5 / SciPy 1.18 against Linux / CPython 3.10 / NumPy 2.2 / SciPy 1.15 | all 136 reported table cells agree to 6 significant figures |
| Latin hypercube stratification | one point per stratum, all factors |

The step-convergence result is worth isolating. Because the disturbance record is decoupled from the solver, a tenfold change in step moves every reported metric by less than 0.03 %, so the campaign step of 10⁻² s can be justified rather than assumed.

*(Figure 2: (a) realised versus target Kaimal spectrum; (b) swing histories of the five baselines on the nominal wind-free manoeuvre, with the 4.8° bound marked.)*

### 3.2 Four campaigns

Four campaigns were run, each with five baselines over a 500-point paired design — 10 000 closed-loop runs in total, under two minutes of single-core time on the batched path. All use the planar plant and a 20 m transfer. `calm` has no wind; `reference` adds Kaimal turbulence at 12 m/s mean and 14 % intensity over a 20 s quintic ramp; `dryden` replaces the disturbance model; `stress` keeps Kaimal turbulence but halves the ramp to 10 s on a trapezoidal profile.

**Reference campaign** (n = 500, Kaimal, 20 s quintic ramp). Entries marked c are right-censored at the 40 s horizon: PD does not settle on 99 % of samples, ZVD on 100 %, LQR on 60 %.

| controller | ISE [m²s] | peak swing [°] | residual swing [°] | settling [s] | effort [N²s] | chatter [N] | bound met |
|---|---|---|---|---|---|---|---|
| PD    | 0.904   | 3.46 | 1.038 | >40.0 c | 5.79·10⁷ | 1.51·10⁴ | 461/500 |
| LQR   | 0.150   | 3.22 | 1.010 | 38.7 c | 5.41·10⁷ | 1.30·10⁴ | 480/500 |
| ZVD   | 297.9   | 2.95 | 1.069 | >40.0 c | 4.30·10⁷ | 1.15·10⁴ | 481/500 |
| SMC   | 0.0036  | 4.12 | 1.454 | 12.0 | 7.34·10⁷ | 2.33·10⁴ | 391/500 |
| HSMC  | 0.0025  | 4.13 | 1.494 |  5.6 | 7.24·10⁷ | 2.33·10⁴ | 388/500 |

Paired contrasts against PD (95 % percentile bootstrap CI, Wilcoxon signed-rank *p* floored at 10⁻¹⁰, matched-pairs rank-biserial effect size @rrb@):

| controller | residual swing [°] | peak swing [°] |
|---|---|---|
| LQR  | −0.0274 [−0.036, −0.019], *p* = 1.1·10⁻⁸, @rrb@ = −0.295 | −0.240 [−0.255, −0.227], *p* < 10⁻¹⁰, @rrb@ = −0.98 |
| ZVD  | +0.0315 [+0.024, +0.039], *p* < 10⁻¹⁰, @rrb@ = +0.335 | −0.513 [−0.550, −0.475], *p* < 10⁻¹⁰, @rrb@ = −0.90 |
| SMC  | +0.417 [+0.380, +0.453], *p* < 10⁻¹⁰, @rrb@ = +0.91 | +0.657 [+0.626, +0.690], *p* < 10⁻¹⁰, @rrb@ = +1.00 |
| HSMC | +0.457 [+0.417, +0.496], *p* < 10⁻¹⁰, @rrb@ = +0.91 | +0.669 [+0.635, +0.704], *p* < 10⁻¹⁰, @rrb@ = +1.00 |

Bound satisfaction is a paired binary outcome on the same samples, so it is contrasted with McNemar's test rather than by comparing two proportions: against PD, LQR gains 19 samples and loses none (*p* = 3.6·10⁻⁵), ZVD gains 24 and loses 4 (*p* = 3.3·10⁻⁴), SMC loses 70 and gains none and HSMC loses 73 and gains none (both *p* < 10⁻¹⁰).

**Stress campaign** (n = 500, Kaimal, 10 s trapezoidal ramp):

| controller | ISE [m²s] | peak swing [°] | residual swing [°] | effort [N²s] | chatter [N] | bound met |
|---|---|---|---|---|---|---|
| PD    |   2.04 | 11.90 | 2.88 | 4.25·10⁸ | 8.58·10⁴ |   0/500 |
| LQR   |   1.44 | 11.62 | 1.54 | 3.14·10⁸ | 6.85·10⁴ |   0/500 |
| ZVD   | 415.7  |  4.45 | 1.10 | 6.34·10⁷ | 2.20·10⁴ | 348/500 |
| SMC   |   0.128| 15.40 | 8.33 | 1.07·10⁹ | 1.45·10⁵ |   0/500 |
| HSMC  |   0.073| 16.52 | 9.20 | 1.21·10⁹ | 1.55·10⁵ |   0/500 |

Four observations, offered as demonstrations of what the bench makes visible rather than as claims about controller families.

**Single-metric reporting would invert the ranking.** In the reference campaign the two sliding baselines reduce ISE by two to three orders of magnitude relative to PD. Settling times must be read with care: PD fails to settle within the 40 s horizon on 99 % of samples and ZVD on 100 %, so their entries in Table 2 are right-censored and the apparent speed-ups of 3.3 (SMC) and 7.1 (HSMC) are lower bounds rather than measurements. Reported alone — as is common — this reads as a decisive win. On the *same runs* the sliding baselines raise peak swing by 0.66° (@rrb@ = +1.00, i.e. SMC exceeds PD on every one of the 500 paired samples) and drop bound satisfaction from 461/500 to 388/500 (McNemar, 73 discordant pairs all in one direction, *p* < 10⁻¹⁰). Both are true; a paper reporting either without the other is not wrong, it is unfalsifiable.

**The ranking is not preserved between operating points.** Under the aggressive transfer every feedback controller violates the 4.8° bound on every one of 500 samples, and the ZVD shaper — which was unremarkable in the reference campaign, losing to PD on residual swing — becomes the only controller that satisfies it at all (348/500), improving on PD by 7.45° of peak swing (Fig. 4a, b). A benchmark that reports one manoeuvre reports one ranking.

**And that reversal is itself a trade, not a free lunch.** The ZVD advantage is bought by delay: its ISE in the stress campaign is 416 m²s against 2.0 for PD, because a shaper tuned to a 0.9 rad/s pendulum spreads a 10 s command over roughly 17 s. It satisfies the bound largely by declining to perform the aggressive transfer. Figure 4c shows the whole picture in one plane — ZVD is the only marker below the bound and the only one three orders of magnitude to the right. This is precisely the kind of statement that requires several metrics on the same runs to make at all.

**Pairing buys the resolution.** The LQR-versus-PD residual-swing difference is −0.027° against a between-sample spread of about 0.6°, and the paired design still resolves it (*p* = 1.1·10⁻⁸); unpaired, this would need an order of magnitude more samples.

**Confounds, stated plainly.** These are five baselines at one set of gains, tuned once on the benign nominal and held fixed across all four campaigns. That protocol has consequences: a feedforward scheme is not degraded by extrapolation away from its tuning point in the way a fixed-gain feedback law is, so part of the reversal could be an artefact of the protocol rather than a property of the controller families. We therefore tested it: every baseline was re-tuned on the stress nominal under an equal budget — a grid over its two principal gains, 24 evaluations, one frozen objective — and the stress campaign repeated on the identical design.

| controller | peak swing [°], frozen → re-tuned | bound met, frozen → re-tuned |
|---|---|---|
| PD    | 11.90 → 10.04 | 0/500 → 0/500 |
| LQR   | 11.62 → 8.32  | 0/500 → 0/500 |
| ZVD   |  4.45 → 4.52  | 348/500 → 335/500 |
| SMC   | 15.40 → 13.87 | 0/500 → 0/500 |
| HSMC  | 16.52 → 14.07 | 0/500 → 0/500 |

Re-tuning lowers their peak swing by 1.9° to 3.3° and costs them dearly in tracking, since the tuning objective ignores it (PD's ISE rises from 2.04 to 10.05 m²s, LQR's from 1.44 to 46.3). It does not move the conclusion: the four feedback baselines remain at 0/500 at every threshold from 4.0° to 7.0°, while the shaper holds 169/500 to 493/500. The reversal is a property of the manoeuvre, not of the tuning protocol. Two of the five optima sit on the boundary of their grid, so the budget binds; that is a limitation and is reported as one. More generally, the benchmark treats the *tuning budget* — what was tuned, on which operating point, and with how many evaluations — as a mandatory reported field alongside the metric hash, on the grounds that an unstated tuning budget is as much a confound as an unstated metric definition. The comparison here isolates *these implementations* at *this* budget, not the design philosophies, and the package exposes every gain so that a user can move them and re-run.

Across the reference campaign, mean residual swing increases with the total variation of the horizontal command, but the relationship is an association and not a monotone one (Fig. 3b). The two sliding baselines, whose commands have the largest total variation (2.33·10⁴ N), leave the most residual motion (1.45° and 1.49°); LQR combines the second-smoothest command with the lowest residual swing (1.30·10⁴ N, 1.010°). The ZVD shaper is the exception that bounds the trend: it has the smoothest command of the five (1.15·10⁴ N) yet leaves more residual swing than LQR (1.069°), because its residual motion is set by the mismatch between the shaper design frequency and the sampled rope length rather than by command roughness. Rank correlation across the five baselines is ρ = 0.70, suggestive and not conclusive at *n* = 5 controllers. The mechanism is physically unsurprising — a rougher command carries more energy near the pendulum band — but it is exactly the mechanism that a benchmark reporting only tracking error cannot expose, and it is why command total variation is a first-class metric here rather than a diagnostic. Establishing it as more than an association would require a sweep along the smoothness axis at fixed controller structure.

**The bound is a convention, and the conclusion survives moving it.** The 4.8° bound is an operating convention rather than a derived limit, so bound satisfaction was recomputed over a range of thresholds. In the stress campaign the shaper is the only baseline to satisfy it at every threshold from 4.0° to 7.0° (178/500 rising to 495/500, against 0/500 for the four feedback baselines throughout); they begin to pass only at 10°, where PD reaches 179/500 and ZVD is already at 500/500.

The campaign mean of residual swing enters a 1 % band at n ≈ 380–460 across the four campaigns, which is why 500 is the shipped default and why the earlier 120-sample pilot was not enough.

*(Figure 3: reference campaign — (a) distribution of residual swing by controller; (b) mean residual swing against mean command total variation; (c) paired contrasts against PD with 95 % confidence intervals.)*

*(Figure 4: (a) bound satisfaction across the four operating points; (b) mean residual swing, showing that the ranking is not preserved; (c) the stress campaign in the tracking-error / peak-swing plane.)*

### 3.3 A campaign on the spatial plant

A fifth campaign exercises the three-dimensional plant: 150 paired samples over a six-factor design that adds the suspension torsional stiffness to the five factors of Section 3.2, on a 15 m transfer under Kaimal turbulence, smaller than the planar campaigns because this plant is not batched.

Two parameters are conventions. The torsional stiffness of the suspension is gravitational rather than elastic — two falls a distance *S* apart act as a bifilar pendulum, so *k*ψ = *W*(*S*/2)²/*L*, giving 3.3 kN·m/rad at nominal. The wind acts through a centre of pressure offset by 15 % of the payload width, following the torsional wind load case of ASCE 7 [35]; a suspended payload is not a building, so that eccentricity is a parameter rather than an asserted property.

| controller | ISE [m²s] | peak swing [°] | residual swing [°] | peak yaw [°] | effort [N²s] | chatter [N] | bound met |
|---|---|---|---|---|---|---|---|
| PD    | 0.647   | 3.60 | 1.014 | 4.85 | 6.24·10⁷ | 1.65·10⁴ | 143/150 |
| LQR   | 0.181   | 3.45 | 0.851 | 4.85 | 5.92·10⁷ | 1.50·10⁴ | 147/150 |
| ZVD   | 214.8   | 2.47 | 0.747 | 4.85 | 2.83·10⁷ | 8.36·10³ | 149/150 |
| SMC   | 0.00356 | 4.02 | 1.748 | 4.85 | 7.60·10⁷ | 2.33·10⁴ | 133/150 |
| HSMC  | 0.00243 | 4.05 | 1.872 | 4.85 | 7.73·10⁷ | 2.45·10⁴ | 135/150 |

The pattern of Section 3.2 reproduces on a different plant: the sliding baselines again buy tracking accuracy with command roughness and pay in residual swing and bound satisfaction, and the shaper again trades three orders of magnitude of tracking error for the best swing behaviour. Surviving a change of plant is the strongest evidence here that it is not an artefact of the planar model.

The yaw column is the interesting one: it is identical across all five baselines to machine precision on every one of the 150 samples. That coordinate is unactuated and dynamically uncoupled from the drives, so it is a pure disturbance response and no anti-sway law can touch it — a mean peak of 4.85° with a sample standard deviation of 3.05° is what the wind does, whatever the controller. Reporting it identifies what a controller would have to acquire — an actuated hook rotator, or deck-reacting taglines — before payload yaw is a control problem rather than a weather one.

## 4. Impact

The immediate impact is that a controller for an underactuated crane can now be evaluated against a bench its author did not build: the plants, seeds, uncertainty realisations and scoring code are unchanged, and the result is a paired contrast with a confidence interval rather than a table of two numbers.

It also makes new questions askable. Sections 3.2 and 3.3 could ask whether a rank ordering transfers between operating points and between plants — questions needing the manoeuvre and the disturbance held fixed while everything else moves, which no paper can arrange for another's method. The same machinery makes the *tuning budget* measurable rather than rhetorical: re-tuning five baselines under an equal declared budget and re-running an identical design is minutes of work here.

For reviewers, the ledger changes what can be checked without re-running anything: the metric hash establishes that baseline and proposed-method rows were scored identically, and the seed list that they saw identical plants and identical disturbance realisations. For simulation-only work — which most crane control is, because instrumenting a heavy-lift crane for controlled experiments is rarely feasible — this is the closest available substitute for an independently operated testbed.

On adoption we can only report the truth: the package is new, and there is no user base, download count or third-party publication to cite. What we can report is the cost of adoption: two methods to implement, no dependency beyond NumPy and SciPy, no compilation, campaigns that checkpoint and resume, and every reported number reproducible from a clean environment.

The design is not crane-specific: the plant interface assumes only an underactuated mechanical system with a safety-relevant unactuated coordinate, so gantry systems, container spreaders, cable-driven parallel robots and slung-load aircraft are expected to fit without changes to the harness.

## 5. Conclusions

`cranebench` supplies what a crane control paper currently has to reinvent: verified plants, spectrally exact disturbances, a paired uncertainty design, frozen metrics, paired statistics and full provenance. It contains no controller of its own, which is what makes it usable as a neutral bench. All three models are cross-validated against independent derivations and conserve energy to better than 10⁻¹³; every reported metric is converged in the integrator step to better than 0.03 %.

Four 500-sample campaigns — 10 000 closed-loop runs — show that the ranking of five classical baselines is not preserved across operating points, that the controller which alone meets the swing bound under an aggressive transfer meets it by declining to perform the transfer, and that this survives re-tuning every baseline on that operating point under an equal budget. Neither statement is available from any single metric on any single manoeuvre, which is the argument for a shared bench.

Planned extensions: elastic and torsional rope models; actuator dynamics, quantisation and transport delay; batching of the spatial and dual plants; and rare-event estimators for the tails a 500-sample campaign cannot resolve.

## Acknowledgements

*(to be completed)*

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Funding

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## CRediT authorship contribution statement

**Oleksii Sheremet:** Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Visualization, Writing - original draft, Writing - review and editing, Project administration. **Serhii Podliesnyi:** Conceptualization, Methodology, Validation, Supervision, Writing - review and editing.

## Data availability

The source code, the four campaign result files, the provenance ledgers and the scripts that regenerate every table and figure in this article are openly available in the repository given in the code metadata table, archived together with the source under the DOI of reference [36]. No other data were used.

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work the authors used a large language model assistant in order to refine the manuscript text, to implement parts of the accompanying software, and to cross-check the reference list. After using this tool the authors reviewed and edited the content as needed, independently verified every numerical result reported here by re-running the released code, and checked every reference against the publisher record. The authors take full responsibility for the content of the publication.

## References

[1] E. M. Abdel-Rahman, A. H. Nayfeh, Z. N. Masoud, Dynamics and control of cranes: a review, J. Vib. Control 9 (7) (2003) 863-908. doi:10.1177/1077546303009007007.

[2] L. Ramli, Z. Mohamed, A. M. Abdullahi, H. I. Jaafar, I. M. Lazim, Control strategies for crane systems: a comprehensive review, Mech. Syst. Signal Process. 95 (2017) 1-23. doi:10.1016/j.ymssp.2017.03.015.

[3] M. R. Mojallizadeh, B. Brogliato, C. Prieur, Modeling and control of overhead cranes: a tutorial overview and perspectives, Annu. Rev. Control 56 (2023) 100877. doi:10.1016/j.arcontrol.2023.100877.

[4] K.-S. Hong, U. H. Shah, Dynamics and Control of Industrial Cranes, Springer, Singapore, 2019. doi:10.1007/978-981-13-5770-1.

[5] N. C. Singer, W. P. Seering, Preshaping command inputs to reduce system vibration, ASME J. Dyn. Syst. Meas. Control 112 (1) (1990) 76-82. doi:10.1115/1.2894142.

[6] W. Singhose, Command shaping for flexible systems: a review of the first 50 years, Int. J. Precis. Eng. Manuf. 10 (4) (2009) 153-68. doi:10.1007/s12541-009-0084-2.

[7] J. Vaughan, A. Yano, W. Singhose, Comparison of robust input shapers, J. Sound Vib. 315 (4-5) (2008) 797-815. doi:10.1016/j.jsv.2008.02.032.

[8] K. L. Sorensen, W. Singhose, S. Dickerson, A controller enabling precise positioning and sway reduction in bridge and gantry cranes, Control Eng. Pract. 15 (7) (2007) 825-37. doi:10.1016/j.conengprac.2006.03.005.

[9] N. Sun, Y. Fang, H. Chen, A survey on recent developments of input shaping methods for suppressing residual vibration of flexible structures, Mech. Syst. Signal Process. 66-67 (2015) 468-82.

[10] Y. Fang, W. E. Dixon, D. M. Dawson, E. Zergeroglu, Nonlinear control of underactuated overhead crane systems, IEEE Trans. Autom. Control 48 (12) (2003) 2233-8. doi:10.1109/TAC.2003.819287.

[11] M. Fliess, J. Levine, P. Martin, H. Sira-Ramirez, Flatness and defect of non-linear systems: introductory theory and examples, Int. J. Control 61 (6) (1995) 1327-61. doi:10.1080/00207179508921959.

[12] Z. Wu, X. Xia, B. Zhu, Model predictive control for improving operational efficiency of overhead cranes, Nonlinear Dyn. 79 (4) (2015) 2639-57. doi:10.1007/s11071-014-1837-8.

[13] J. Lin, Y. Fang, B. Lu, H. Cao, Y. Hao, Constrained model predictive control for 3-D offshore boom cranes, Control Eng. Pract. 142 (2024) 105741. doi:10.1016/j.conengprac.2023.105741.

[14] V. I. Utkin, Variable structure systems with sliding modes, IEEE Trans. Autom. Control 22 (2) (1977) 212-22. doi:10.1109/TAC.1977.1101446.

[15] X. Shao, et al., Disturbance observer-based robust sliding mode control for an overhead crane system, Mech. Syst. Signal Process. 114 (2018) 63-77. doi:10.1016/j.ymssp.2018.05.012.

[16] J. Han, From PID to active disturbance rejection control, IEEE Trans. Ind. Electron. 56 (3) (2009) 900-6. doi:10.1109/TIE.2008.2011621.

[17] X. Gu, et al., Adaptive hierarchical sliding mode controller for tower cranes based on finite time disturbance observer, Int. J. Adapt. Control Signal Process. 36 (9) (2022).

[18] J. J. Downs, E. F. Vogel, A plant-wide industrial process control problem, Comput. Chem. Eng. 17 (3) (1993) 245-55.

[19] P. Henderson, R. Islam, P. Bachman, J. Pineau, D. Precup, D. Meger, Deep reinforcement learning that matters, in: Proc. AAAI Conf. Artif. Intell., Vol. 32, 2018. doi:10.1609/aaai.v32i1.11694.

[20] H. H. Lee, Modeling and control of a three-dimensional overhead crane, ASME J. Dyn. Syst. Meas. Control 120 (4) (1998) 471-6. doi:10.1115/1.2801488.

[21] D. Chwa, Nonlinear tracking control of 3-D overhead cranes against the initial swing angle and the variation of the payload weight, IEEE Trans. Control Syst. Technol. 17 (4) (2009) 876-83.

[22] B. Lu, Y. Fang, N. Sun, Modeling and nonlinear coordination control for an underactuated dual overhead crane system, Automatica 91 (2018) 244-55. doi:10.1016/j.automatica.2018.01.008.

[23] X. Zhao, J. Huang, Distributed-mass payload dynamics and control of dual cranes undergoing planar motions, Mech. Syst. Signal Process. 126 (2019) 636-48. doi:10.1016/j.ymssp.2019.02.032.

[24] J. Huang, K. Zhu, Dynamics and control of three-dimensional dual cranes transporting a bulky payload, Proc. Inst. Mech. Eng. C: J. Mech. Eng. Sci. 235 (11) (2021) 1956-65. doi:10.1177/0954406220949579.

[25] J. R. R. A. Martins, P. Sturdza, J. J. Alonso, The complex-step derivative approximation, ACM Trans. Math. Softw. 29 (3) (2003) 245-62. doi:10.1145/838250.838251.

[26] J. C. Kaimal, J. C. Wyngaard, Y. Izumi, O. R. Coté, Spectral characteristics of surface-layer turbulence, Q. J. R. Meteorol. Soc. 98 (417) (1972) 563-89. doi:10.1002/qj.49709841707.

[27] International Electrotechnical Commission, IEC 61400-1 Ed.4: Wind energy generation systems - Part 1: Design requirements, IEC, 2019.

[28] M. Shinozuka, G. Deodatis, Simulation of stochastic processes by spectral representation, Appl. Mech. Rev. 44 (4) (1991) 191-204. doi:10.1115/1.3119501.

[29] T. R. Beal, Digital simulation of atmospheric turbulence for Dryden and von Karman models, J. Guid. Control Dyn. 16 (1) (1993) 132-8.

[30] C. F. Van Loan, Computing integrals involving the matrix exponential, IEEE Trans. Autom. Control 23 (3) (1978) 395-404. doi:10.1109/TAC.1978.1101743.

[31] J.-J. E. Slotine, S. S. Sastry, Tracking control of non-linear systems using sliding surfaces, with application to robot manipulators, Int. J. Control 38 (2) (1983) 465-92. doi:10.1080/00207178308933088.

[32] M. D. McKay, R. J. Beckman, W. J. Conover, A comparison of three methods for selecting values of input variables in the analysis of output from a computer code, Technometrics 21 (2) (1979) 239-45. doi:10.2307/1268522.

[33] B. Efron, Bootstrap methods: another look at the jackknife, Ann. Stat. 7 (1) (1979) 1-26. doi:10.1214/aos/1176344552.

[34] F. Wilcoxon, Individual comparisons by ranking methods, Biom. Bull. 1 (6) (1945) 80-3. doi:10.2307/3001968.

[35] American Society of Civil Engineers, ASCE/SEI 7, Minimum Design Loads and Associated Criteria for Buildings and Other Structures; torsional wind load case (Case 2), which pairs 75 % of the design wind load with an eccentricity of 15 % of the width.

[36] O. Sheremet, S. Podliesnyi, cranebench: a reproducible benchmark for underactuated crane control, version 1.0.0 [software], archived at Zenodo, 2026. doi:*(to be inserted on release)*.
