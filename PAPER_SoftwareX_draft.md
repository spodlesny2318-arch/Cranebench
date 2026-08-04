# cranebench: a reproducible benchmark for the control of underactuated crane systems

**Oleksii Sheremet**^a,\*, **Serhii Podliesnyi**^b

^a Department of Electromechanical Systems of Automation and Electric Drive, Donbas State Engineering Academy, 39 Mashinobudivnykiv Blvd., Kramatorsk, 84313, Ukraine
^b Department of Fundamentals of Machine Design, Donbas State Engineering Academy, Kramatorsk, Ukraine
\* Corresponding author: Oleksii.Sheremet@ddma.edu.ua

---

## Abstract

Crane control results are rarely comparable: each paper defines its own plant, manoeuvre, disturbance, uncertainty set and metrics, then reports that the proposed method beats baselines the same authors implemented. `cranebench` fixes the bench instead of the method. It supplies three cross-validated plants, two spectrally exact disturbance models, a paired Latin-hypercube design, hash-frozen metrics, a provenance ledger and five classical baselines — and deliberately no novel controller. A user implements two methods and receives a paired contrast. Six campaigns over three plants show that the ranking of the baselines is largely preserved across operating points while the margin separating them changes by two orders of magnitude.

**Keywords:** crane control; underactuated systems; reproducibility; benchmark; Monte Carlo; paired experimental design

---

## Metadata

| Nr | Code metadata description | Metadata |
|----|---------------------------|----------|
| C1 | Current code version | v1.0.0 |
| C2 | Permanent link to code/repository used for this code version | https://github.com/spodlesny2318-arch/Cranebench |
| C3 | Legal code license | BSD-3-Clause |
| C4 | Code versioning system used | git |
| C5 | Software code languages, tools and services used | Python (≥ 3.10), NumPy, SciPy; SymPy for the model derivations; Matplotlib for the figures |
| C6 | Compilation requirements, operating environments and dependencies | None; pure Python. `pip install -e .` installs NumPy ≥ 1.24 and SciPy ≥ 1.10. Tested on Linux with CPython 3.10 |
| C7 | If available, link to developer documentation/manual | `README.md` and `docs/DESIGN.md` in the repository |
| C8 | Support email for questions | Oleksii.Sheremet@ddma.edu.ua |

## 1. Motivation and significance

Anti-sway control of underactuated cranes is a mature field with a large and still growing literature [1--4], spanning input shaping [5--8], energy- and passivity-based design [9, 10], model predictive control [11, 12], sliding-mode and hierarchical sliding-mode schemes [13--15], and learning-based controllers. It is also a field in which two papers published in the same year are, in practice, almost never comparable.

The reason is structural rather than cultural. A crane control paper must specify a plant (planar or spatial; point-mass or rigid-body payload; rigid or elastic rope), a manoeuvre (distance, ramp time, whether the hoist moves), a disturbance (absent, harmonic, or a turbulence spectrum with its own parameters), an uncertainty set (which parameters vary and over what range), a set of metrics, and an integrator. Each of these is a legitimate modelling choice, and each is normally made afresh. The consequence is that when a new method reports, say, a 4.4-fold reduction in integral sway error against an LQR baseline, a reader cannot tell how much of that factor is the method, how much is the particular LQR weights the authors chose, and how much is a manoeuvre or disturbance that happens to suit the proposed structure.

Two further failure modes are specific to simulation-only work, which most of this literature is. First, the baselines are implemented by the proposers of the new method, who have every incentive — usually unconscious — to spend more tuning effort on their own entry. Second, the metric definitions themselves are rarely frozen: nothing in a published paper allows a reader to verify that the numbers reported for the baseline and for the proposed method were produced by the same scoring code.

`cranebench` addresses these by inverting what is held fixed. It supplies the plants, disturbances, uncertainty design, metrics and statistics, and supplies **no novel controller at all**. Its baselines are five textbook designs. A researcher proposing a new controller subclasses one interface, runs the same design against the same seeds, and reports a paired contrast. What varies is then the controller, and only the controller.

Related benchmarking efforts exist in adjacent areas — the Tennessee Eastman challenge problem in process control [16], and the reproducibility critique and attendant suites in reinforcement learning [17] — but the crane community has no shared bench, and its distinctive features (underactuation with a physically meaningful safety bound on an unactuated coordinate, strongly non-stationary disturbance, and parameters that vary by design between lifts) are not covered by any of them.

## 2. Software description

### 2.1 Software architecture

The package is organised around the rule that a controller may read the state and return an input, and may do nothing else. Plants, disturbances, the uncertainty design and the metrics are outside the controller's reach.

The package is laid out so that each fixed element of the bench is one module and the user's code touches none of them:

```
cranebench/  plants/     base.py planar.py spatial.py dual.py _generated.py
             wind/       kaimal.py dryden.py
             controllers/base.py classical.py sliding.py
             metrics.py  uncertainty.py  stats.py  ledger.py
             integrate.py  runner.py  batch.py  reference.py
tools/       derive_symbolic.py  retune.py  verify_manuscript.py
examples/    run_batch_campaign.py  run_spatial_campaign.py
             run_dual_campaign.py  run_ablation.py  summarise_stats.py
tests/       25 tests: dynamics, symbolic, wind, harness, batch, manuscript
```

`_generated.py` is emitted by `tools/derive_symbolic.py` and committed, so a user needs neither SymPy nor the derivation time; `tools/verify_manuscript.py` checks the tables of this article against the campaign files.

**Comparison with existing software.** A general-purpose engine simulates a crane perfectly well; it does not address the problem this package addresses, which is not the equations of motion but everything around them.

| | Simulink | MuJoCo / Drake | OpenModelica | cranebench |
|---|---|---|---|---|
| Simulates an underactuated crane | yes | yes | yes | yes |
| Fixed manoeuvre and disturbance realisation | user-defined | user-defined | user-defined | **fixed and seeded** |
| Spectrally exact wind with declared variance | no | no | no | **yes** |
| Paired uncertainty design replayed across controllers | no | no | no | **yes** |
| Scoring code frozen and hash-verified | no | no | no | **yes** |
| Paired statistics with effect sizes | no | no | no | **yes** |
| Provenance ledger for every run | no | no | no | **yes** |
| Licence cost to a reviewer reproducing the work | commercial | free | free | **free** |

**Plants.** Three are supplied, following the standard formulations [18, 19]: a planar trolley/hoist/payload crane (6 states, 2 inputs); a three-dimensional crane with a spherical-pendulum suspension and a payload yaw coordinate restrained by the gravitational torsional stiffness of a two-fall suspension and driven by the yaw moment of an eccentric centre of pressure (12 states, 3 inputs), using a parameterisation whose rope-length norm is exact at all swing angles; and a cooperative dual crane carrying a rigid beam on two visco-elastic falls (10 states, 2 inputs), in the lineage of the dual-crane models of [20--22], modelled with spring-damper falls rather than holonomic constraints so that the system remains an ODE and load-sharing dynamics stay visible.

Plants are declared kinematically — where the point masses are as a function of the generalised coordinates, plus an additive generalised inertia, a potential and the generalised forces — and the equations of motion are assembled numerically. Hand-derived spatial crane models are a known source of silent algebra errors; assembling them makes the model checkable, because the same declaration can be handed to an independent derivation and the two compared. Every plant here is checked that way: the planar equations against the assembler, and the spatial and dual equations against symbolic derivations produced by SymPy's `LagrangesMethod`, which shares no code with the assembler (Table 1). The verified symbolic result is then emitted as a closed-form fast path, fifteen times quicker than the assembler and required by the test suite to reproduce it; for the dual plant that path is used only while both falls carry tension, and the assembler takes over the moment one goes slack, because the symbolic derivation assumes the smooth branch of a unilateral contact law. Kinematic Jacobians use the complex-step derivative [23], `J = Im[p(q + ih e_k)]/h`. This is not a refinement: the Coriolis term differentiates the mass matrix, so a central-difference Jacobian is differentiated twice and its noise floor surfaces in the accelerations at 10⁻⁴, which is what agreement between the hand-derived and assembled planar models measured before the change and 6.1·10⁻¹⁰ after it.

**Disturbances.** Kaimal turbulence [24, 25] is synthesised by spectral representation with random phases [26] and rescaled so that the realised variance is exact — without which two controllers on "the same seed" would in fact see disturbances of different strength. Dryden turbulence [27] is realised as a rational shaping filter discretised by Van Loan's method [28], started from its stationary distribution, and has unbounded support, so any never-exceed claim evaluated against it is necessarily probabilistic. Both records live on a fixed 100 Hz grid and are interpolated, **independently of the integrator step**: a record indexed by the solver step changes when the step changes, so a step-refinement study would measure a different disturbance at every step and could never converge.

**Baselines.** PD, LQR on the numerically linearised plant, a ZVD input shaper [5, 6] in cascade with PD tracking, boundary-layer sliding-mode control [13, 29], and hierarchical sliding-mode control [15]. All gains are exposed and documented. The switching function is `tanh(s/φ)` with `φ` reported: with `sign`, the measured effort of a sliding controller is a function of the integrator step rather than of the design.

**Batched execution.** A second path integrates the whole Monte Carlo ensemble at once — the state is an `(N, n_x)` array and one RK4 step advances every sample together — completing 500 runs of one controller in 5.5 s against 0.8 s per single run on the scalar reference path. A second implementation of the same experiment is a liability unless it is pinned to the first, so per-sample setup that could be re-derived incorrectly is obtained by calling the *scalar* code, and `tests/test_batch.py` runs a full paired design through both paths and requires agreement on every metric, observed at 1.4·10⁻¹⁴.

The intended workflow is `pip install` → pick a plant and a disturbance → implement a controller → run a campaign → read the paired contrast, and in the simplest case it is four lines:

```python
from cranebench.runner import Campaign, run_campaign
from cranebench.uncertainty import lhs_design

results = run_campaign(Campaign(controllers={"mine": MyController()}),
                       lhs_design(n=500))
```

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

**Tuning.** Gains are tuned per plant under a budget that is declared and recorded: a grid over each controller's two principal gains, scored on a held-out uncertainty design with its own seed, minimising mean(peak + 4 × residual swing) subject to the load arriving within 1.25 % of the transfer and the command using no more than 5 % of the drive's available slew rate. Every evaluated point is written to disk, not only the winner. Each clause was added because its absence produced a wrong answer: scoring swing alone optimises for not moving, scoring one deterministic run selects gains that diverge over the design, and omitting the slew clause let a shaper win a swing bound with a command that sat on the drive's rate limit for the whole transfer.

**Uncertainty design.** A centred Latin hypercube [30] over five multiplicative factors (payload mass ±20 %, rope length ±25 %, swing damping ×0.5–2, drive damping ±30 %, mean wind ±40 %) plus a per-sample wind seed. The sample list is drawn once and replayed for every controller, so contrasts are taken sample by sample and plant variation cancels rather than adding to the noise.

**Metrics.** A single module computes every reported number and hashes its own source at import; the hash is written into every result file. The common failure in simulation-only work is not fabricated data but a metric that quietly changed between the baseline and proposed-method runs, and freezing the module makes that detectable by a reviewer who never executes the code.

*(Figure 1: The harness. Everything shaded blue is fixed by the benchmark; the controller is the only user-supplied component, and it sees the state and the reference and nothing else.)*

**Development.** The package is versioned semantically and released with a tag; 25 tests cover the models, the disturbances, the harness and the manuscript tables, run in about a minute, and are intended for continuous integration on the public repository.

**Provenance.** A ledger records the package version, the metric hash, a hash of every source file, the design seed, every per-run wind seed, the integrator and step, and the interpreter and library versions.

### 2.2 Software functionalities

- Eleven metrics per run, defined once and applied identically to every controller: **ISE**, the integral of squared horizontal position error; **settling time**, the first instant after which the position error stays inside a 2 cm band for the rest of the run, reported as right-censored at the horizon when that never happens; **peak** and **RMS swing** over the run and **residual swing**, the RMS swing over the final 5 s; **peak** and **RMS yaw**; **effort**, the integral of u'u over the horizontal channels; **peak input**; **command total variation** ("chatter"), the summed absolute increment of the held horizontal command; and **bound satisfaction**, whether peak swing stayed within a declared limit.
- Paired statistics: percentile bootstrap confidence intervals [31] on the sample-by-sample difference, Wilcoxon signed-rank tests [32], matched-pairs rank-biserial correlation as the effect size consistent with that test, McNemar's test for the binary bound-satisfaction outcome, win rates, and a running-mean convergence diagnostic. Because a campaign produces of order 10² paired tests, and because at *n* = 500 almost any non-zero difference reaches significance, effect sizes are the primary reported quantity and *p*-values are floored at 10⁻¹⁰: the normal approximation to the signed-rank statistic carries no meaning further into the tail, and the point null of an exactly zero shift is known to be false in any case.
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
| Kaimal realised variance / spectrum, 0.01–5 Hz | exact to 10⁻⁹ / log-PSD correlation *r* = 0.998 |
| Dryden stationarity from first sample | within 12 % over 400 seeds |
| Metric convergence, dt from 10⁻³ to 10⁻² s | all metrics within 2.5·10⁻⁴ relative |
| Batched path vs. scalar reference path, all metrics, full design | max relative difference 1.4·10⁻¹⁴ |
| Independent re-run on a second platform: Windows 11 / CPython 3.14 / NumPy 2.5 against Linux / CPython 3.10 / NumPy 2.2 | every reported table cell agrees to 6 significant figures |

*(Figure 2: (a) realised versus target Kaimal spectrum; (b) swing histories of the five baselines on the nominal wind-free manoeuvre, with the 4.8° bound marked.)*

### 3.2 Four campaigns on the planar plant

Four operating points, five baselines, 500 paired samples each: `calm` has no wind; `reference` adds Kaimal turbulence at 12 m/s and 14 % intensity over a 20 s quintic ramp; `dryden` changes the disturbance model; `stress` halves the ramp to 10 s on a trapezoidal profile. Gains are tuned once, per plant, under a declared budget (Section 3.4).

**Reference campaign** (n = 500, Kaimal, 20 s quintic ramp):

| controller | ISE [m²s] | peak swing [°] | residual swing [°] | final error [m] | effort [N²s] | chatter [N] | bound met |
|---|---|---|---|---|---|---|---|
| PD    | 0.8552  | 3.239 | 1.022 | 0.1131 | 5.46·10⁷ | 1.40·10⁴ | 482/500 |
| LQR   | 0.1415  | 3.044 | 1.011 | 0.0316 | 5.18·10⁷ | 1.24·10⁴ | 492/500 |
| ZVD   | 300.0   | 2.696 | 1.054 | 0.1162 | 4.00·10⁷ | 1.05·10⁴ | 496/500 |
| SMC   | 0.00319 | 3.771 | 1.241 | 0.00489 | 6.38·10⁷ | 1.99·10⁴ | 443/500 |
| HSMC  | 0.00192 | 3.774 | 1.272 | 0.00462 | 6.30·10⁷ | 1.99·10⁴ | 440/500 |

Paired contrasts against PD use a percentile bootstrap over 10⁴ resamples from a generator seeded at 12345, Wilcoxon signed-rank with matched-pairs rank-biserial effect size @rrb@, and McNemar for the binary bound outcome:

| controller | residual swing [°] | peak swing [°] | bound met (McNemar) |
|---|---|---|---|
| LQR  | −0.0116 [−0.018, −0.006], @rrb@ = −0.147 | −0.195 [−0.206, −0.184], @rrb@ = −0.97 | +10, −0, *p* = 4.4·10⁻³ |
| ZVD  | +0.031 [+0.026, +0.037], @rrb@ = +0.48 | −0.543 [−0.576, −0.508], @rrb@ = −0.93 | +14, −0, *p* = 5.1·10⁻⁴ |
| SMC  | +0.219 [+0.195, +0.242], @rrb@ = +0.79 | +0.532 [+0.509, +0.555], @rrb@ = +1.00 | +0, −39, *p* = 1.2·10⁻⁹ |
| HSMC | +0.250 [+0.225, +0.275], @rrb@ = +0.81 | +0.536 [+0.510, +0.561], @rrb@ = +1.00 | +0, −42, *p* = 2.5·10⁻¹⁰ |

**Stress campaign** (n = 500, Kaimal, 10 s trapezoidal ramp), with exact binomial intervals on the bound:

| controller | ISE [m²s] | peak swing [°] | residual swing [°] | effort [N²s] | chatter [N] | bound met |
|---|---|---|---|---|---|---|
| PD    |   1.918 | 11.44 | 2.357 | 3.75·10⁸ | 7.90·10⁴ | 0/500 |
| LQR   |   1.302 | 11.34 | 1.392 | 2.91·10⁸ | 6.45·10⁴ | 0/500 |
| ZVD   | 418.1   |  4.11 | 1.079 | 6.09·10⁷ | 2.14·10⁴ | 414/500 |
| SMC   |   0.0849| 14.41 | 6.549 | 8.57·10⁸ | 1.31·10⁵ | 0/500 |
| HSMC  |   0.0533| 15.47 | 7.231 | 9.58·10⁸ | 1.39·10⁵ | 0/500 |

Exact binomial intervals on the bound: 414/500 is [79.2, 86.0] %, and 0/500 is [0.0, 0.7] % — not a claim of impossibility, which is why the interval is quoted.

**The ranking is largely preserved; the margin is not.** Kendall's τ between the controller orderings of two operating points is +1.00 for peak swing on every pair among reference, dryden and stress, and +0.40 to +1.00 for residual swing. The one genuine reordering is the shaper, which moves from third to first on residual swing between reference and stress. What changes decisively is not the order but the size of the difference: the same five controllers are separated by 0.03° of residual swing in the reference campaign and by 6.2° in the stress campaign, and a bound that all five satisfy at one operating point is met by exactly one at the other. Reporting "the ranking reverses" would have been the more quotable claim and the wrong one; the rank correlation says so, and it is the reason to compute it rather than argue from examples.

**Single-metric reporting inverts the conclusion.** The two sliding baselines reduce ISE by two to three orders of magnitude relative to PD and arrive to within 5 mm. Reported alone, as is common, this reads as a decisive win. On the *same runs* they raise peak swing by 0.53° (@rrb@ = +1.00, i.e. every one of the 500 paired samples) and lose 39 and 42 samples of bound satisfaction against PD. Both are true; a paper reporting either without the other is unfalsifiable.

**Settling time cannot be summarised by its mean.** It is right-censored whenever the run ends with the error outside the tolerance band, which under wind is most of the time: 99.4 % of PD samples and 100 % of ZVD samples in the reference campaign. A Kaplan–Meier restricted mean over the horizon returns the horizon itself for those two, which is the correct answer and an uninformative one. The metric module therefore reports the censored fraction alongside, and the arrival metric — mean position error over the last 5 s — carries the question the settling time was meant to answer.

### 3.3 The other two plants

The spatial campaign uses 150 paired samples over a six-factor design that adds the suspension torsional stiffness; the dual campaign 100 samples over a design on beam mass, fall stiffness and damping, drive damping, fall length and mean wind.

**Spatial plant** (n = 150):

| controller | ISE [m²s] | peak swing [°] | residual swing [°] | final error [m] | peak yaw [°] | chatter [N] | bound met |
|---|---|---|---|---|---|---|---|
| PD   | 0.6341 | 3.484 | 0.9621 | 0.0757 | 4.665 | 1.60·10⁴ | 148/150 |
| LQR  | 0.1763 | 3.370 | 0.8233 | 0.0205 | 4.669 | 1.47·10⁴ | 149/150 |
| ZVD  | 215.7  | 2.346 | 0.7349 | 0.0916 | 4.622 | 8.07·10³ | 150/150 |
| SMC  | 0.00341| 3.828 | 1.574  | 0.0054 | 4.682 | 2.19·10⁴ | 141/150 |
| HSMC | 0.00210| 3.852 | 1.676  | 0.0070 | 4.688 | 2.29·10⁴ | 138/150 |

**Dual plant** (n = 100):

| controller | ISE [m²s] | peak swing [°] | residual swing [°] | final error [m] | chatter [N] | bound met |
|---|---|---|---|---|---|---|
| PD   | 0.0487 | 7.314 | 2.245 | 0.0200 | 1.32·10⁵ | 0/100 |
| LQR  | 1.723  | 5.162 | 2.220 | 0.1074 | 7.75·10⁴ | 48/100 |
| ZVD  | 107.1  | 5.065 | 2.299 | 0.0561 | 7.85·10⁴ | 52/100 |
| SMC  | 1.500  | 6.529 | 2.258 | 0.0335 | 1.10·10⁵ | 0/100 |
| HSMC | 1.405  | 6.536 | 2.265 | 0.0267 | 1.11·10⁵ | 0/100 |

The pattern of Section 3.2 reproduces on the spatial plant and does not survive intact on the dual one, where the ordering by peak swing is LQR ≈ ZVD < SMC ≈ HSMC < PD and PD, best on tracking by a factor of thirty, misses the bound on every sample. Surviving one change of plant and failing another is more useful than a clean sweep would have been: it locates the claim.

Payload yaw on the spatial plant is identical across all five baselines to machine precision on every sample — 4.85° ± 3.05° of pure disturbance response, because the coordinate is unactuated and uncoupled from the drives. Reporting it identifies what a controller would have to acquire, an actuated hook rotator or deck-reacting taglines, before payload yaw is a control problem rather than a weather one.

### 3.4 What the campaigns found in the bench itself

The most useful output of running the bench was a list of defects in the bench. None was caught by the 25-test suite, because all of them concern experiments that are uninformative rather than equations that are wrong.

- The quasi-steady drag used the absolute wind speed, so the model had no aerodynamic damping at all: 106 N·s/m of dissipation was missing against 0.28 N·s/m modelled, a factor of 380. Correcting it lowers the residual swing of the sliding baselines by 15 % and of PD by 1.5 %, halving the gap between them.
- A drive slew limit was added at the same time. Measured across the stress campaign at 20, 40, 80 and 160 kN/s it changes no reported conclusion: the commands bind on 0.1 % to 0.4 % of control steps at the tightest setting.
- Payload yaw on the spatial plant was inert until the drag model was given the eccentric centre of pressure a bluff body actually has.
- The tuning objective, minimise swing, was degenerate: its global optimum is a controller that does not move, and it duly drove PD to the softest gains on the grid. It is now constrained by arrival and by a slew budget.
- Tuning on a single deterministic run overfits: the LQR gains that won it diverged across the evaluation design, mean peak swing 169°. Gains are now scored on a held-out uncertainty design with its own seed.
- On the dual plant the anti-sway surface was built on the beam pitch, which a synchronised transfer does not excite, so the hierarchical baseline was numerically identical to the flat one; the tracked output was the payload rather than the trolleys, which made the arrival criterion unreachable under a steady wind by construction; the settling window was 1.6 pendulum periods against 2.9 on the planar plant; and the state reference moved the trolleys while holding the beam still, which cost LQR three metres of tracking error. All four are cases of a rule that is correct for a single crane and silently wrong for two.

*(Figure 3: reference campaign — (a) distribution of residual swing by controller; (b) mean residual swing against mean command total variation; (c) paired contrasts against PD with 95 % confidence intervals.)*

*(Figure 4: (a) bound satisfaction across the four operating points; (b) mean residual swing; (c) the stress campaign in the tracking-error / peak-swing plane.)*

## 4. Impact

The immediate impact is that a controller for an underactuated crane can now be evaluated against a bench its author did not build: the plants, seeds, uncertainty realisations and scoring code are unchanged, and the result is a paired contrast with a confidence interval rather than a table of two numbers.

It also makes new questions askable. Sections 3.2 and 3.3 could ask whether a rank ordering transfers between operating points and between plants — questions needing the manoeuvre and the disturbance held fixed while everything else moves, which no paper can arrange for another's method. The same machinery makes the *tuning budget* measurable rather than rhetorical: re-tuning five baselines under an equal declared budget and re-running an identical design is minutes of work here.

For reviewers, the ledger changes what can be checked without re-running anything: the metric hash establishes that baseline and proposed-method rows were scored identically, and the seed list that they saw identical plants and identical disturbance realisations. For simulation-only work — which most crane control is, because instrumenting a heavy-lift crane for controlled experiments is rarely feasible — this is the closest available substitute for an independently operated testbed.

On adoption we can only report the truth: the package is new, and there is no user base, download count or third-party publication to cite. What we can report is the cost of adoption: two methods to implement, no dependency beyond NumPy and SciPy, no compilation, campaigns that checkpoint and resume, and every reported number reproducible from a clean environment.

Three uses follow at no extra cost: a reviewer can replicate a submitted comparison instead of trusting it; a course can use the bench as an exercise in experimental hygiene; and a community can run an open comparison on it, which is how other fields acquired their reference results.

The design is not crane-specific: the plant interface assumes only an underactuated mechanical system with a safety-relevant unactuated coordinate, so gantry systems, container spreaders, cable-driven parallel robots and slung-load aircraft are expected to fit without changes to the harness.

## 5. Conclusions

`cranebench` supplies what a crane control paper currently has to reinvent: verified plants, spectrally exact disturbances, a paired uncertainty design, frozen metrics, paired statistics and full provenance. It contains no controller of its own, which is what makes it usable as a neutral bench. All three models are cross-validated against independent derivations and conserve energy to better than 10⁻¹³; every reported metric is converged in the integrator step to better than 0.03 %.

Four 500-sample campaigns — 10 000 closed-loop runs — show that the ranking of five classical baselines is not preserved across operating points, that the controller which alone meets the swing bound under an aggressive transfer meets it by declining to perform the transfer, and — measured rather than argued — that what changes between operating points is the margin rather than the order. Neither statement is available from any single metric on any single manoeuvre, which is the argument for a shared bench.

Planned extensions: an actuated hook rotator, so that payload yaw becomes a control problem rather than a weather one; elastic and torsional rope models; actuator dynamics, quantisation and transport delay; batching of the spatial and dual plants; and rare-event estimators for the tails a 500-sample campaign cannot resolve. The intent throughout is that `cranebench` fixes the bench instead of the method, and can therefore serve as a reusable community bench on which future crane controllers are compared fairly, reproducibly and by somebody other than their author.

## Acknowledgements

*(to be completed)*

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Funding

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## CRediT authorship contribution statement

**Oleksii Sheremet:** Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Visualization, Writing - original draft, Writing - review and editing, Project administration. **Serhii Podliesnyi:** Conceptualization, Methodology, Validation, Supervision, Writing - review and editing.

## Data availability

The source code, the four campaign result files, the provenance ledgers and the scripts that regenerate every table and figure in this article are openly available in the repository given in the code metadata table, archived together with the source under the DOI of reference [34]; that DOI resolves to the exact release on which every number in this article was computed. No other data were used.

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work the authors used a large language model assistant in order to refine the manuscript text, to implement parts of the accompanying software, and to cross-check the reference list. After using this tool the authors reviewed and edited the content as needed, independently verified every numerical result reported here by re-running the released code, and checked every reference against the publisher record. The authors take full responsibility for the content of the publication.

## References

[1] E. M. Abdel-Rahman, A. H. Nayfeh, Z. N. Masoud, Dynamics and control of cranes: a review, J. Vib. Control 9 (7) (2003) 863-908. doi:10.1177/1077546303009007007.

[2] L. Ramli, Z. Mohamed, A. M. Abdullahi, H. I. Jaafar, I. M. Lazim, Control strategies for crane systems: a comprehensive review, Mech. Syst. Signal Process. 95 (2017) 1-23. doi:10.1016/j.ymssp.2017.03.015.

[3] M. R. Mojallizadeh, B. Brogliato, C. Prieur, Modeling and control of overhead cranes: a tutorial overview and perspectives, Annu. Rev. Control 56 (2023) 100877. doi:10.1016/j.arcontrol.2023.03.002.

[4] K.-S. Hong, U. H. Shah, Dynamics and Control of Industrial Cranes, Springer, Singapore, 2019. doi:10.1007/978-981-13-5770-1.

[5] N. C. Singer, W. P. Seering, Preshaping command inputs to reduce system vibration, ASME J. Dyn. Syst. Meas. Control 112 (1) (1990) 76-82. doi:10.1115/1.2894142.

[6] W. Singhose, Command shaping for flexible systems: a review of the first 50 years, Int. J. Precis. Eng. Manuf. 10 (4) (2009) 153-68. doi:10.1007/s12541-009-0084-2.

[7] J. Vaughan, A. Yano, W. Singhose, Comparison of robust input shapers, J. Sound Vib. 315 (4-5) (2008) 797-815. doi:10.1016/j.jsv.2008.02.032.

[8] K. L. Sorensen, W. Singhose, S. Dickerson, A controller enabling precise positioning and sway reduction in bridge and gantry cranes, Control Eng. Pract. 15 (7) (2007) 825-37. doi:10.1016/j.conengprac.2006.03.005.

[9] Y. Fang, W. E. Dixon, D. M. Dawson, E. Zergeroglu, Nonlinear coupling control laws for an underactuated overhead crane system, IEEE/ASME Trans. Mechatron. 8 (3) (2003) 418-23. doi:10.1109/TMECH.2003.816822.

[10] M. Fliess, J. Lévine, P. Martin, P. Rouchon, Flatness and defect of non-linear systems: introductory theory and examples, Int. J. Control 61 (6) (1995) 1327-61. doi:10.1080/00207179508921959.

[11] Z. Wu, X. Xia, B. Zhu, Model predictive control for improving operational efficiency of overhead cranes, Nonlinear Dyn. 79 (4) (2015) 2639-57. doi:10.1007/s11071-014-1837-8.

[12] J. Lin, Y. Fang, B. Lu, H. Cao, Y. Hao, Constrained model predictive control for 3-D offshore boom cranes, Control Eng. Pract. 142 (2024) 105741. doi:10.1016/j.conengprac.2023.105741.

[13] V. I. Utkin, Variable structure systems with sliding modes, IEEE Trans. Autom. Control 22 (2) (1977) 212-22. doi:10.1109/TAC.1977.1101446.

[14] J. Han, From PID to active disturbance rejection control, IEEE Trans. Ind. Electron. 56 (3) (2009) 900-6. doi:10.1109/TIE.2008.2011621.

[15] X. Gu, H. Zhou, M. Hong, S. Ye, Y. Guo, Adaptive hierarchical sliding mode controller for tower cranes based on finite time disturbance observer, Int. J. Adapt. Control Signal Process. 36 (9) (2022) 2319-40. doi:10.1002/acs.3458.

[16] J. J. Downs, E. F. Vogel, A plant-wide industrial process control problem, Comput. Chem. Eng. 17 (3) (1993) 245-55.

[17] P. Henderson, R. Islam, P. Bachman, J. Pineau, D. Precup, D. Meger, Deep reinforcement learning that matters, in: Proc. AAAI Conf. Artif. Intell., Vol. 32, 2018. doi:10.1609/aaai.v32i1.11694.

[18] H. H. Lee, Modeling and control of a three-dimensional overhead crane, ASME J. Dyn. Syst. Meas. Control 120 (4) (1998) 471-6. doi:10.1115/1.2801488.

[19] D. Chwa, Nonlinear tracking control of 3-D overhead cranes against the initial swing angle and the variation of payload weight, IEEE Trans. Control Syst. Technol. 17 (4) (2009) 876-83. doi:10.1109/TCST.2008.2011367.

[20] B. Lu, Y. Fang, N. Sun, Modeling and nonlinear coordination control for an underactuated dual overhead crane system, Automatica 91 (2018) 244-55. doi:10.1016/j.automatica.2018.01.008.

[21] X. Zhao, J. Huang, Distributed-mass payload dynamics and control of dual cranes undergoing planar motions, Mech. Syst. Signal Process. 126 (2019) 636-48. doi:10.1016/j.ymssp.2019.02.032.

[22] J. Huang, K. Zhu, Dynamics and control of three-dimensional dual cranes transporting a bulky payload, Proc. Inst. Mech. Eng. C: J. Mech. Eng. Sci. 235 (11) (2021) 1956-65. doi:10.1177/0954406220949579.

[23] J. R. R. A. Martins, P. Sturdza, J. J. Alonso, The complex-step derivative approximation, ACM Trans. Math. Softw. 29 (3) (2003) 245-62. doi:10.1145/838250.838251.

[24] J. C. Kaimal, J. C. Wyngaard, Y. Izumi, O. R. Coté, Spectral characteristics of surface-layer turbulence, Q. J. R. Meteorol. Soc. 98 (417) (1972) 563-89. doi:10.1002/qj.49709841707.

[25] International Electrotechnical Commission, IEC 61400-1:2019 Ed. 4.0, Wind energy generation systems - Part 1: Design requirements, IEC, Geneva, 2019, ISBN 978-2-8322-7972-4.

[26] M. Shinozuka, G. Deodatis, Simulation of stochastic processes by spectral representation, Appl. Mech. Rev. 44 (4) (1991) 191-204. doi:10.1115/1.3119501.

[27] T. R. Beal, Digital simulation of atmospheric turbulence for Dryden and von Karman models, J. Guid. Control Dyn. 16 (1) (1993) 132-8. doi:10.2514/3.11437.

[28] C. F. Van Loan, Computing integrals involving the matrix exponential, IEEE Trans. Autom. Control 23 (3) (1978) 395-404. doi:10.1109/TAC.1978.1101743.

[29] J.-J. E. Slotine, S. S. Sastry, Tracking control of non-linear systems using sliding surfaces, with application to robot manipulators, Int. J. Control 38 (2) (1983) 465-92. doi:10.1080/00207178308933088.

[30] M. D. McKay, R. J. Beckman, W. J. Conover, A comparison of three methods for selecting values of input variables in the analysis of output from a computer code, Technometrics 21 (2) (1979) 239-45. doi:10.2307/1268522.

[31] B. Efron, Bootstrap methods: another look at the jackknife, Ann. Stat. 7 (1) (1979) 1-26. doi:10.1214/aos/1176344552.

[32] F. Wilcoxon, Individual comparisons by ranking methods, Biom. Bull. 1 (6) (1945) 80-3. doi:10.2307/3001968.

[33] American Society of Civil Engineers, ASCE/SEI 7-22, Minimum Design Loads and Associated Criteria for Buildings and Other Structures, ASCE, Reston VA, 2022; torsional wind load case (Case 2), which pairs 75 % of the design wind load with an eccentricity of 15 % of the width.

[34] O. Sheremet, S. Podliesnyi, cranebench: a reproducible benchmark for underactuated crane control, version 1.0.0 [software], Zenodo, 2026. doi:10.5281/zenodo.21785505.
