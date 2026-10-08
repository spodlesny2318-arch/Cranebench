# cranebench: a reproducible benchmarking framework for underactuated crane control

**Serhii Podliesnyi**^a,\*, **Oleksii Sheremet**^b, **Bohdan Vorobiov**^b

^a Department of Innovative Technologies and Machine Design, Donbas State Engineering Academy (DSEA), 72 Akademichna St., Kramatorsk, Donetsk Region, 84313, Ukraine
^b Department of Electromechanical Systems of Automation and Electric Drive, Donbas State Engineering Academy (DSEA), 72 Akademichna St., Kramatorsk, Donetsk Region, 84313, Ukraine
\* Corresponding author: Serhii Podliesnyi; Serhii.Podliesnyi@ddma.edu.ua; +380686543568; ORCID 0000-0001-8271-4004

## Abstract

Simulation studies of underactuated cranes are difficult to compare because the plant, manoeuvre, disturbance, uncertainty design, metrics and numerical settings are usually changed together with the controller. `cranebench` fixes these elements and leaves the controller as the experimental variable. The package provides three cross-validated crane plants, reproducible Kaimal and Dryden disturbance models, a paired Latin-hypercube uncertainty design, frozen metrics and paired statistics, checkpointed campaigns and a provenance ledger. Five classical baseline controllers are included, but no novel controller. A new controller implements one public interface and is evaluated on the same realisations as the baselines. Six campaigns covering three plants (11,250 closed-loop runs) illustrate the framework: controller differences persist, but their margins and constraint satisfaction can change substantially with operating conditions. The result is a reusable benchmark rather than another controller-specific simulation study.

**Keywords:** crane control; underactuated systems; benchmarking; reproducibility; Monte Carlo; paired experimental design

## Metadata

| Nr | Code metadata description | Metadata |
|----|---------------------------|----------|
| C1 | Current code version | 0.1.0 |
| C2 | Permanent link to code/repository used for this code version | https://github.com/spodlesny2318-arch/Cranebench |
| C3 | Legal code license | BSD-3-Clause |
| C4 | Code versioning system used | git |
| C5 | Software code languages, tools and services used | Python, NumPy, SciPy; SymPy for symbolic derivation; Matplotlib for figures |
| C6 | Compilation requirements, operating environments and dependencies | No compilation; Python >= 3.10; NumPy >= 1.24 and SciPy >= 1.10. Campaign provenance records CPython 3.14.2 and NumPy 2.3.3 on Windows 10. |
| C7 | If available, link to developer documentation/manual | `README.md` and `docs/DESIGN.md` |
| C8 | Support email for questions | Serhii.Podliesnyi@ddma.edu.ua |

## 1. Motivation and significance

Anti-sway control of underactuated cranes is a mature field [1-4], including input shaping [5-8], nonlinear and flatness-based control [9,10], model predictive control [11,12], sliding-mode and disturbance-rejection approaches [13-15], and learning-based methods. Yet results are often hard to compare because the controller is changed at the same time as the experimental environment. A published comparison can therefore conflate controller quality with the selected plant, manoeuvre, disturbance, uncertainty range, scoring rules or baseline tuning.

This problem is especially acute in simulation-only studies. A new method is usually evaluated by its authors, the baselines are implemented and tuned by the same group, and the metric calculation is embedded in a project script. Even when the numerical model is correct, a reader has no guarantee that two methods saw identical uncertainty realisations or that their performance was scored by exactly the same function. Pairing the experiment and freezing the scorer are therefore not cosmetic reproducibility measures; they define what is being compared.

`cranebench` inverts this workflow. It fixes the plant definitions, reference manoeuvres, disturbance generators, uncertainty design, metrics, statistical procedures and provenance machinery, while deliberately supplying **no novel controller**. A new controller reads the plant state and reference and returns the plant input through one interface. The same uncertainty realisation and wind seed are then replayed for every controller, so the primary comparison is a sample-by-sample contrast rather than a difference between independent Monte Carlo clouds.

The design was motivated by benchmarking practice in other fields, including the Tennessee Eastman challenge problem [16] and reproducibility-oriented evaluation in reinforcement learning [17]. The literature surveyed for this work did not reveal a commonly adopted crane-control benchmark that fixes both the physical test environment and the scoring pipeline. `cranebench` is intended to occupy that gap: it is a benchmark harness with reusable mechanical models, rather than a claim that one control law is universally best.

## 2. Software description

### 2.1 Software architecture

The architecture separates fixed benchmark assets from user-supplied control code. Plants, wind records, uncertainty factors, metrics and statistical routines are imported by the runner; the controller can only access the state and reference through the public controller interface.

The core package contains three plant models (`planar`, `spatial`, `dual`), two stochastic wind generators (`kaimal`, `dryden`), five baseline controllers, an uncertainty module, a frozen metric module, an RK4 integrator, campaign runners, a batch implementation for the planar plant and a provenance ledger. Symbolic model verification is kept outside the execution path: `tools/derive_symbolic.py` produces the committed `_generated.py` fast path, while `tools/verify_manuscript.py` compares the published tables against the stored campaign files.

*(Figure 1: cranebench architecture. Benchmark-controlled elements are fixed and replayed for every controller; the only user-supplied element is the controller implementation.)*

The public API is intentionally small. A minimal controller is an ordinary Python subclass of `Controller`:

```
import numpy as np
from cranebench.controllers import Controller, BASELINES
from cranebench.reference import Manoeuvre
from cranebench.runner import Campaign, run_campaign
from cranebench.uncertainty import lhs_design

class MyController(Controller):
    name = "mine"
    def __call__(self, t, x):
        return np.zeros(self.plant.nu)

controllers = {**{k: v() for k, v in BASELINES.items()}, "mine": MyController()}
camp = Campaign(name="example", plant="planar", wind="kaimal",
                manoeuvre=Manoeuvre(distance=20.0, t_ramp=20.0, t_total=40.0),
                controllers=controllers)
run_campaign(camp, lhs_design(n=100, seed=20260729))
```

This example uses only public objects present in the release. The runner creates the plant and disturbance from the paired design, resets the controller, executes the simulation and writes checkpoint and provenance records.

**Plants and model checks.** The planar model is a trolley/hoist/payload crane with six states and two inputs. The spatial model has a spherical-pendulum suspension and a payload yaw coordinate, giving twelve states and three inputs. The dual model carries a rigid beam on two visco-elastic falls and has ten states and two inputs. The latter uses spring-damper falls rather than holonomic constraints so that load-sharing dynamics remain explicit and the system remains an ODE. The spatial suspension stiffness is a declared gravitational bifilar-pendulum convention, not an elastic-rope measurement. The 15% centre-of-pressure eccentricity used for yaw loading is likewise an explicit benchmark parameter associated with the torsional wind load case of ASCE/SEI 7-22 [33], not a claim about a generic payload.

Plants are declared kinematically and assembled numerically. Independent symbolic derivations using SymPy's `LagrangesMethod` are compared with the numerical assembler for the spatial and dual plants, while the planar model is cross-checked independently. The emitted symbolic fast path is used in the smooth taut branch; for the dual plant, the assembler is used after a fall becomes slack because the symbolic derivation does not represent the unilateral contact transition. Kinematic derivatives use the complex-step approximation [23]. The resulting agreement tests address algebraic consistency; they are not physical validation.

**Disturbances.** Kaimal turbulence [24,25] is generated by spectral representation with random phases [26] and rescaled so that the realised variance matches the target. Dryden turbulence [27] is implemented as a rational shaping filter discretised with Van Loan's method [28] and started from its stationary distribution. Both records are generated on a fixed 100 Hz grid and interpolated independently of the integration step, preventing step-refinement tests from changing the disturbance realisation itself.

**Baselines.** The supplied baselines are PD, LQR, a ZVD input shaper [5,6] in cascade with PD tracking, boundary-layer sliding-mode control [13,29] and hierarchical sliding-mode control [15]. Their gains are exposed in the source and are part of the benchmark definition. The switching function is a `tanh` boundary layer rather than `sign`, avoiding a direct dependence of the measured command roughness on integration step size.

**Tuning and provenance.** Tuning is a separate reproducible functionality rather than a hidden part of the main campaigns. Candidate gains are evaluated on a held-out uncertainty design under declared arrival and actuator-slew constraints, and all evaluated points are saved. The ledger records package version, metric hash, source hashes, design seed, per-sample wind seeds, solver and step, and software-environment information. Checkpointed JSONL output allows campaigns to resume after interruption.

### 2.2 Software functionalities

The metric module computes eleven quantities identically for every controller: position-error ISE; right-censored settling time; peak, RMS and residual swing; peak and RMS yaw; horizontal control effort; peak horizontal input; **command total variation (CTV)**; final position error; and binary swing-bound satisfaction. The stored legacy key is `chatter`, but the reported quantity is explicitly CTV to distinguish it from physical high-frequency chatter.

The statistical layer provides percentile bootstrap intervals [31] for paired differences, Wilcoxon signed-rank tests [32], matched-pairs rank-biserial effect sizes, McNemar's test for the binary bound outcome and running-mean convergence. The paired design uses a centred Latin hypercube [30] over five multiplicative factors for the planar campaigns: payload mass ±20%, rope length ±25%, swing damping ×0.5-2, drive damping ±30% and mean wind ±40%, plus a per-sample wind seed. The spatial and dual campaigns add plant-specific factors.

Horizontal control effort excludes the hoist channel because that channel carries the static payload weight; including it would make the dominant contribution largely common to all controllers. CTV is kept separate from effort because a controller can have low integral effort while producing a command sequence that would be unacceptable to a finite-bandwidth drive.

## 3. Illustrative examples

### 3.1 Verification of the bench

| Check | Result |
|---|---|
| Hand-derived planar equations vs. assembled Lagrangian model, 40 random states | max relative difference 6.1·10^-10 |
| Spatial plant: assembled model vs. independent SymPy derivation, 25 random states | mass matrix 4.8·10^-16, accelerations 1.1·10^-9 |
| Dual plant: assembled model vs. independent SymPy derivation, taut branch, 25 states | mass matrix 2.9·10^-12, accelerations 1.6·10^-6 |
| Energy conservation, damping removed, 3 s, planar / spatial / dual | 8.4·10^-15 / 7.4·10^-15 / 1.8·10^-14 relative |
| Mass matrices symmetric positive definite, all plants | passes |
| Spherical-pendulum map preserves rope length | exact to 10^-12 at 1.2 rad swing |
| Kaimal realised variance / spectrum, 0.01-5 Hz | exact to 10^-9 / log-PSD correlation r = 0.998 |
| Dryden stationarity from first sample | within 12% over 400 seeds |
| Metric convergence, dt from 10^-3 to 10^-2 s | all metrics within 2.5·10^-4 relative |
| Batched path vs. scalar reference path, all metrics, full design | max relative difference 1.4·10^-14 |
| Test suite | 26 tests passed |

*(Figure 2: (a) realised versus target Kaimal spectrum; (b) swing histories of the five baselines for a nominal wind-free manoeuvre, with the 4.8° bound.)*

These checks establish internal consistency, reproducibility of the stochastic generators and agreement between the scalar and batched execution paths. They do not establish agreement with a physical crane experiment, and the release makes that distinction explicit.

### 3.2 Four campaigns on the planar plant

Four 500-sample campaigns use the same controller set and paired Latin-hypercube structure. The `calm` campaign removes wind. The `reference` campaign uses Kaimal turbulence with a 20 s quintic ramp. The `dryden` campaign changes the turbulence model. The `stress` campaign uses a 10 s trapezoidal transfer.

**Reference campaign** (n = 500, Kaimal, 20 s quintic ramp):

| controller | ISE [m²s] | peak swing [°] | residual swing [°] | final error [m] | effort [N²s] | CTV [N] | bound met |
|---|---|---|---|---|---|---|---|
| PD    | 0.8552  | 3.239 | 1.022 | 0.1131 | 5.46·10⁷ | 1.40·10⁴ | 482/500 |
| LQR   | 0.1415  | 3.044 | 1.011 | 0.0316 | 5.18·10⁷ | 1.24·10⁴ | 492/500 |
| ZVD   | 300.0   | 2.696 | 1.054 | 0.1162 | 4.00·10⁷ | 1.05·10⁴ | 496/500 |
| SMC   | 0.00319 | 3.771 | 1.241 | 0.00489 | 6.38·10⁷ | 1.99·10⁴ | 443/500 |
| HSMC  | 0.00192 | 3.774 | 1.272 | 0.00462 | 6.30·10⁷ | 1.99·10⁴ | 440/500 |

Paired contrasts against PD use a 10^4-resample percentile bootstrap, Wilcoxon signed-rank testing, matched-pairs rank-biserial effect size and McNemar testing for the bound outcome:

| controller | residual swing [°] | peak swing [°] | bound met (McNemar) |
|---|---|---|---|
| LQR  | -0.0116 [-0.018, -0.006], @rrb@ = -0.147 | -0.195 [-0.206, -0.184], @rrb@ = -0.97 | +10, -0, p = 4.4·10^-3 |
| ZVD  | +0.031 [+0.026, +0.037], @rrb@ = +0.48 | -0.543 [-0.576, -0.508], @rrb@ = -0.93 | +14, -0, p = 5.1·10^-4 |
| SMC  | +0.219 [+0.195, +0.242], @rrb@ = +0.79 | +0.532 [+0.509, +0.555], @rrb@ = +1.00 | +0, -39, p = 1.2·10^-9 |
| HSMC | +0.250 [+0.225, +0.275], @rrb@ = +0.81 | +0.536 [+0.510, +0.561], @rrb@ = +1.00 | +0, -42, p = 2.5·10^-10 |

**Stress campaign** (n = 500, Kaimal, 10 s trapezoidal ramp):

| controller | ISE [m²s] | peak swing [°] | residual swing [°] | effort [N²s] | CTV [N] | bound met |
|---|---|---|---|---|---|---|
| PD    | 1.918 | 11.44 | 2.357 | 3.75·10⁸ | 7.90·10⁴ | 0/500 |
| LQR   | 1.302 | 11.34 | 1.392 | 2.91·10⁸ | 6.45·10⁴ | 0/500 |
| ZVD   | 418.1 | 4.11 | 1.079 | 6.09·10⁷ | 2.14·10⁴ | 414/500 |
| SMC   | 0.0849 | 14.41 | 6.549 | 8.57·10⁸ | 1.31·10⁵ | 0/500 |
| HSMC  | 0.0533 | 15.47 | 7.231 | 9.58·10⁸ | 1.39·10⁵ | 0/500 |

The stress campaign illustrates why one metric is insufficient. The sliding baselines obtain very low ISE and small final tracking error, but their peak swing and CTV are substantially higher. ZVD achieves the smallest residual swing and is the only baseline to satisfy the 4.8° swing bound on a substantial fraction of the stress realisations, but its large ISE shows that the shaper can trade tracking for sway reduction under the shortened manoeuvre. Exact binomial intervals are available in the stored campaign analysis; 414/500 corresponds to 79.2-86.0%, while 0/500 corresponds to 0.0-0.7%.

The paired results also show that robustness claims depend on the operating point without implying a universal winner. For peak swing, the controller ordering is unchanged across the reference, dryden and stress campaigns (Kendall's tau = 1.00 for every pair of those operating points), whereas residual-swing separations change considerably. The benchmark therefore distinguishes stable ordering from changing margin.

### 3.3 The other two plants

The spatial campaign contains 150 paired samples and the dual campaign 100 paired samples. The smaller sizes keep the illustrative examples computationally practical while retaining identical controller interfaces and provenance mechanisms.

**Spatial plant** (n = 150):

| controller | ISE [m²s] | peak swing [°] | residual swing [°] | final error [m] | peak yaw [°] | CTV [N] | bound met |
|---|---|---|---|---|---|---|---|
| PD   | 0.6341 | 3.484 | 0.9621 | 0.0757 | 4.665 | 1.60·10⁴ | 148/150 |
| LQR  | 0.1763 | 3.370 | 0.8233 | 0.0205 | 4.669 | 1.47·10⁴ | 149/150 |
| ZVD  | 215.7  | 2.346 | 0.7349 | 0.0916 | 4.622 | 8.07·10³ | 150/150 |
| SMC  | 0.00341 | 3.828 | 1.574 | 0.0054 | 4.682 | 2.19·10⁴ | 141/150 |
| HSMC | 0.00210 | 3.852 | 1.676 | 0.0070 | 4.688 | 2.29·10⁴ | 138/150 |

**Dual plant** (n = 100):

| controller | ISE [m²s] | peak swing [°] | residual swing [°] | final error [m] | CTV [N] | bound met |
|---|---|---|---|---|---|---|
| PD   | 0.0487 | 7.314 | 2.245 | 0.0200 | 1.32·10⁵ | 0/100 |
| LQR  | 1.723  | 5.162 | 2.220 | 0.1074 | 7.75·10⁴ | 48/100 |
| ZVD  | 107.1  | 5.065 | 2.299 | 0.0561 | 7.85·10⁴ | 52/100 |
| SMC  | 1.500  | 6.529 | 2.258 | 0.0335 | 1.10·10⁵ | 0/100 |
| HSMC | 1.405  | 6.536 | 2.265 | 0.0267 | 1.11·10⁵ | 0/100 |

The spatial plant reproduces the main trade-off pattern of the planar studies, while the dual plant does not. In the dual campaign, PD has the smallest ISE but misses the swing bound on every sample; LQR and ZVD have the smallest peak swing but substantially larger tracking error. This is precisely the type of dependence that a benchmark is meant to expose rather than conceal.

Payload yaw is effectively a disturbance-only output for these baselines: it is unactuated and weakly coupled to the drive coordinates, so the five controllers produce the same yaw response to numerical precision. The result is informative because it identifies a missing actuation or reaction mechanism rather than falsely attributing yaw robustness to the controller family.

*(Figure 3: reference campaign — (a) distribution of residual swing by controller; (b) mean residual swing against mean command total variation; (c) paired contrasts against PD with 95% confidence intervals.)*

*(Figure 4: (a) bound satisfaction across the four planar operating points; (b) mean residual swing; (c) stress-campaign final tracking error versus peak swing.)*

### 3.4 Development findings that shaped the benchmark

The benchmark was designed iteratively, and several targeted development experiments changed the software definition. The most important lessons are retained in the release documentation because they explain why apparently minor implementation details are fixed rather than configurable. Correcting the quasi-steady drag formulation added the missing aerodynamic damping; introducing a drive slew limit prevented a controller from winning only because it demanded unrealistically abrupt commands; adding an eccentric centre of pressure made payload yaw respond to wind; constraining tuning by arrival and slew rate removed the degenerate "minimise swing" solution that simply avoids motion; evaluating gains on a held-out uncertainty design prevented single-run overfitting; and revising the dual-crane tracking and anti-sway definitions removed single-crane assumptions that silently made the coupled benchmark misleading. These are software-engineering lessons from the benchmark itself, not claims of physical superiority.

## 4. Impact

The primary impact of `cranebench` is methodological. It lets a researcher evaluate a new controller against a fixed experimental environment whose seeds, uncertainty factors, scoring code and provenance are independent of the new controller. The output is therefore a paired statistical comparison rather than a qualitative claim that one implementation "outperformed" another.

The package also changes what can be inspected after a simulation study. A reviewer can check the metric hash, source hashes, seeds and solver settings from the ledger; the manuscript verifier can compare every tabulated result against stored campaign arrays; and the same controller interface can be reused across three mechanically different plants. This makes the benchmark suitable for papers, internal regression tests and teaching exercises in computational experimental design.

The present release is new, so no third-party adoption or citation count is claimed. Its immediate practical benefit is lower setup cost: a new study needs to implement one controller interface rather than reconstruct the plant, uncertainty design, wind generator, metrics and statistics. Campaign checkpointing also makes large paired experiments easier to reproduce in slices on shared computing resources.

The benchmark is intentionally simulation-only. Internal consistency of the equations and reproducibility of the numerical experiments should not be confused with hardware validation. Future extensions can add actuator dynamics, quantisation and delay, elastic and torsional rope models, an actuated payload-yaw mechanism, batched execution for the spatial and dual plants, and rare-event estimators for tail probabilities.

## 5. Conclusions

`cranebench` fixes the benchmark rather than the controller. It supplies three cross-validated crane plants, reproducible stochastic disturbances, a paired uncertainty design, frozen metrics, statistical contrasts, checkpointed execution and a provenance ledger, together with five classical baselines and no novel controller. The release therefore provides a neutral software substrate on which future crane-control methods can be compared using the same physical assumptions and the same realisations.

Six illustrative campaigns across three plants comprise 11,250 closed-loop controller runs. The examples show why benchmark design matters: a controller that is best on one metric can be poor on another, margins can change sharply with manoeuvre and disturbance, and a controller ordering observed on one plant need not transfer to a coupled dual system. These findings are not presented as universal controller rankings; their purpose is to demonstrate that the benchmark can make such changes measurable and reproducible.

The current release is a baseline for future community use. Its central design rule is that benchmark-controlled elements remain fixed while the controller is the experimental variable. That separation is what turns a collection of crane simulations into a reusable benchmarking framework.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Funding

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## CRediT authorship contribution statement

**Serhii Podliesnyi:** Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Visualization, Writing - original draft, Writing - review and editing, Project administration. **Oleksii Sheremet:** Conceptualization, Methodology, Validation, Supervision, Writing - review and editing. **Bohdan Vorobiov:** Software, Validation, Formal analysis, Writing - review and editing.

## Data availability

The source code, campaign result files, provenance ledgers and scripts used to regenerate the tables and figures are openly available in the repository given in C2. The current manuscript intentionally does not cite either of the earlier pre-release Zenodo identifiers, because the repository contained inconsistent DOI metadata. An archival DOI should be inserted after the final three-author release has been deposited; until then, the GitHub repository is the authoritative public location for code and results. A development version, 0.1.2.dev0, has been uploaded to TestPyPI; both distributions were uploaded and installation of the wheel was verified in a clean environment. TestPyPI is a separate test index, so this trial does not constitute a production PyPI release. The final software release and archival DOI remain pending.

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work the authors used a large language model assistant for language refinement, assistance with selected software-development tasks, and reference-list cross-checking. All generated or modified code was reviewed by the authors and subjected to the same automated and numerical verification procedures as the rest of the released software. The authors independently verified the numerical results reported in the manuscript and take full responsibility for the content of the publication.

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

[25] International Electrotechnical Commission, IEC 61400-1:2019 Ed. 4.0, Wind energy generation systems - Part 1: Design requirements, IEC, Geneva, 2019.

[26] M. Shinozuka, G. Deodatis, Simulation of stochastic processes by spectral representation, Appl. Mech. Rev. 44 (4) (1991) 191-204. doi:10.1115/1.3119501.

[27] T. R. Beal, Digital simulation of atmospheric turbulence for Dryden and von Karman models, J. Guid. Control Dyn. 16 (1) (1993) 132-8. doi:10.2514/3.11437.

[28] C. F. Van Loan, Computing integrals involving the matrix exponential, IEEE Trans. Autom. Control 23 (3) (1978) 395-404. doi:10.1109/TAC.1978.1101743.

[29] J.-J. E. Slotine, S. S. Sastry, Tracking control of non-linear systems using sliding surfaces, with application to robot manipulators, Int. J. Control 38 (2) (1983) 465-92. doi:10.1080/00207178308933088.

[30] M. D. McKay, R. J. Beckman, W. J. Conover, A comparison of three methods for selecting values of input variables in the analysis of output from a computer code, Technometrics 21 (2) (1979) 239-45. doi:10.2307/1268522.

[31] B. Efron, Bootstrap methods: another look at the jackknife, Ann. Stat. 7 (1) (1979) 1-26. doi:10.1214/aos/1176344552.

[32] F. Wilcoxon, Individual comparisons by ranking methods, Biom. Bull. 1 (6) (1945) 80-3. doi:10.2307/3001968.

[33] American Society of Civil Engineers, ASCE/SEI 7-22, Minimum Design Loads and Associated Criteria for Buildings and Other Structures, ASCE, Reston, VA, 2022; torsional wind load case (Case 2).

[34] S. Podliesnyi, O. Sheremet, B. Vorobiov, cranebench: a reproducible benchmarking framework for underactuated crane control, version 0.1.0 [software], GitHub repository, 2026. https://github.com/spodlesny2318-arch/Cranebench.
