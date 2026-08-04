# Design notes

This file records the decisions that a user is entitled to disagree with, and
the reason each was made. It is part of the deliverable: a benchmark whose
choices are undocumented is just another private test bed.

## 1. What the benchmark is not

It is not a state of the art. It contains five classical baselines and no novel
controller, because a benchmark that ships its authors' own method has an
obvious conflict of interest: every subsequent tuning decision quietly favours
the house entry. The intended use is that a new controller is written by
somebody else and dropped in.

## 2. Pairing

Plant uncertainty is drawn once on a Latin hypercube and replayed for every
controller, including the wind seed. Two consequences:

* the contrast between controllers is taken sample by sample, so plant
  variation cancels rather than adding to the noise;
* a controller cannot be lucky. If controller A meets a bound on 500/500
  samples and controller B fails on 40 of them, those 40 are identified plants,
  and the user can go and look at them.

Unpaired comparison of two independent 500-sample clouds typically needs an
order of magnitude more samples for the same power.

## 3. The frozen metric module

`metrics.py` hashes its own source at import and the hash goes into every
ledger. This is not ceremony. The common failure mode in simulation-only work
is not fabricated data, it is a metric that quietly changed between the
baseline runs and the proposed-method runs. Freezing the module makes that
detectable by a reviewer who never runs the code.

## 4. Effort, chatter, and the hoist channel

`effort` integrates `u'u` over the **horizontal** channels only. The hoist
channel carries `m g` for the whole run, which for the default parameters is
39.2 kN; including it makes the effort of every controller equal to `(mg)^2 T`
to three significant figures. This is exactly the sort of metric that makes two
very different controllers look identical, and it is why the plant declares
`horizontal_inputs`.

`chatter` (the total variation of the command) is reported separately from
effort because a boundary-layer sliding controller can buy a small tracking
error with a command that no drive will accept, and effort will not reveal it.
Command roughness is also a documented route to exciting modes the controller
does not model, so it is worth a metric of its own.

## 5. Boundary layer instead of sign

Ideal sliding control switches at the sampling rate, so its measured effort and
chatter are functions of the integrator step rather than of the design. Every
sliding baseline here uses `tanh(s/phi)` with `phi` reported. A user who wants
ideal switching can set `phi` small and will then see the step dependence
directly, which is the honest outcome.

## 6. Fixed step, and where the disturbance lives

The solver is a fixed-step RK4. An adaptive solver changes the effective
sampling of a near-discontinuous control law, so two controllers run under an
adaptive solver are not run under the same conditions.

The disturbance record is synthesised on its own 100 Hz grid and interpolated,
independently of the solver step. This was a bug in the first version of the
package: with the record indexed by the solver step, halving the step produced a
different turbulence realisation, and the metrics moved by 20 % for reasons that
had nothing to do with integration accuracy. With the grids separated, every
metric is converged to better than 0.05 % at `dt = 5e-3`.

## 7. Complex-step Jacobians

The generic Lagrangian assembler needs `dM/dq` for the Coriolis term. With
central-difference kinematic Jacobians, that is a finite difference of a finite
difference: the inner noise floor of about 1e-10 is divided by the outer step of
1e-6 and appears in the accelerations at 1e-4. Measured: the hand-derived planar
model agreed with the assembled one to 5.8e-4 before the change and to 6.1e-10
after it, and the energy drift of the 3-D plant fell from 7.4e-6 to 3.7e-13.

Plants whose kinematics are not holomorphic set `holomorphic = False`. The dual
crane does, because its falls carry tension only and `max(0, .)` has no complex
extension.

## 8. Two execution paths

The scalar path in `runner.py` is the reference: readable, general over all
three plants, and the one the verification tests are written against. The
batched path in `batch.py` integrates the whole Monte Carlo ensemble as an
`(N, nx)` array and is about seventy times faster, which is what makes
500-sample campaigns a matter of seconds rather than half an hour.

Two implementations of the same experiment is normally a bad trade. It is
acceptable here only because the batched path is pinned: per-sample setup that
could plausibly be re-derived incorrectly (the LQR linearisation and Riccati
solution, the equilibrium input, the shaper timing) is computed by calling the
*scalar* code on a scalar plant built from that sample's parameters, and
`tests/test_batch.py` requires the two paths to agree on every metric over a
full paired design. Observed agreement is 1.4e-14.

One consequence worth stating: the batched path freezes the disturbance across
the four RK4 stages, because the scalar path does. Re-evaluating the wind
mid-stage would be defensible numerically but would no longer be the same
experiment, and the equivalence test would fail — correctly.

## 9. Encoding

Every text file is read and written with an explicit `encoding="utf-8"`. This
is not pedantry. Python uses the locale encoding by default on Windows, which
turns `5.79·10⁷` in the manuscript into unparseable bytes; the verification
tool then failed to read those cells and reported success on the subset it
could read. The defect was invisible on Linux and was found by an author
re-running the procedure on Windows. `tests/test_manuscript.py` now guards it.

## 10. Scope limits

Deliberately outside the baseline, and listed so that nobody mistakes absence
for oversight:

* elastic and torsional cable models (the rope here is inextensible);
* full payload attitude; yaw is a single coordinate about a vertical axis;
* aeroelastic effects: the drag coupling is quasi-steady, with no vortex
  shedding, no lock-in, no dynamic stall;
* actuator dynamics, measurement noise, quantisation and transport delay;
* hardware. Every number in this package is a simulation result.

Each of these is a natural extension, and the plant interface is small enough
that adding one does not require touching the harness.

## 11. What a good result looks like here

A controller that improves `residual_swing` at equal `effort` **and** equal
`chatter`, with a paired confidence interval that excludes zero, on a design it
was not tuned on. A controller that improves one metric while silently
tripling another is visible in this bench, which is the point of reporting all
of them.
