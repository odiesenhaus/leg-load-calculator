# Humanoid leg load calculator

A first-pass, **quasi-static** design-sizing tool for one humanoid leg. Give it a pose and a foot load and it reports the
actuator torques, the internal loads and stresses in each bracket, joint-bearing reactions and actuator utilisation
against the motor ratings, next to a live drawing of the pose. A *Gait Cycle* tab sweeps a stance phase of walking and
reports the peak / RMS actuator load. Scope: walking, squats and single-stair step-ups (no running, no jump landing).

It is **not a simulator**: no leg-segment inertia, no swing phase, no controller, no optimiser. Every number is a static
equilibrium. Read `REVIEW_NOTES.md` for what to trust and what not to (section 0 is the trust map).

---

## 1. Run it

```bash
python -m venv .venv
.venv/Scripts/activate                  # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements-dev.txt     # runtime deps + pytest + kaleido (requirements.txt alone = runtime only)
streamlit run app.py                    # opens http://localhost:8501
python -m pytest -q                     # 57 tests, ~30 s (the app tests drive the real UI headlessly)
python tools/fingerprint.py save before.json     # ... change something ...
python tools/fingerprint.py compare before.json  # prints IDENTICAL if no number moved
```

Everything runs locally and offline after `pip install`. PNG export needs a local Chrome (kaleido); JSON/CSV export and
everything else do not. A red banner in the UI counts the lines of `leg_config.yaml` still tagged `# PLACEHOLDER`.

**Sharing it online.** The app is hosted on Streamlit Community Cloud from a GitHub repository (Vercel does not fit:
Streamlit needs a long-running Python server). A hosted copy is shared by every visitor, so it runs **read-only**: *Save
Current Pose as Preset* is disabled and *Export* only offers downloads. Read-only mode switches on automatically whenever
the app is not opened on `localhost`, or when the environment variable `LEG_CALC_READ_ONLY=1` is set. Pushing to the
repository's `main` branch redeploys the hosted copy within a minute or so.

## 2. Repository map

```
leg_config.yaml     ALL geometry, masses, sections, actuators, hardware options, gaits and presets (heavily commented)
leg/                physics only, no UI imports
  config.py           dataclasses, YAML load + validation (readable multi-error messages), actuator swap, preset write-back
  kinematics.py       forward kinematics, mass points, beams, foot Jacobian, singularity check          (A)
  statics.py          joint torques                                                                      (B)
  actuators.py        transmission T(q), actuator loads vs ratings, reflected rotor inertia                (C, E)
  linkages.py         pose-dependent transmissions: the lever + rod parallel ankle                         (C)
  rods.py             ankle rod forces (lever balance) for the structure cuts and bearing reactions       (G)
  structure.py        free-body cuts (incl. the ankle rods), link loads, stress / safety factor, bearings  (D)
  dynamics.py         joint-space mass matrix (not used by the quasi-static pipeline yet)                (F)
  gait.py             keyframed stance phases: sampling, sweeps, per-actuator peak / RMS
  analysis.py         Inputs -> Results: runs A-E once and collects warnings
  report.py           JSON / CSV export          names.py   "hip_abduction" -> "Hip Abduction"
app.py              Streamlit UI (rail, stage, spec sheet, cards)   viz.py   Plotly figures (light / dark / auto palettes)
.streamlit/config.toml  theme: fonts and the light + dark colours
tests/              pytest: hand-checkable physics cases, config validation, gait, options, headless app tests
tools/fingerprint.py  numerical fingerprint of the whole model (detects accidental physics changes)
tools/ankle_bx_sweep.py   sweep of the ankle rod pickup position: lever travel, lever-rod angle, available torque / speed
tools/ankle_optimize.py   grid search of lever length, lever neutral angle and pickup x (min peak motor load, lever-rod >= 45 deg)
tools/ankle_ik_opt.py     same problem by SLSQP on the closed-form RSU inverse kinematics (needs SciPy; --check validates it)
REVIEW_NOTES.md     what to double-check, known conflicts in the inputs, what the tests do not cover
```

## 3. Using the app

* **Hardware** (top of the sidebar): choose the ankle motor (X4-36-L default, or X6-50-L). Ratings, rotor inertia and
  weight are swapped; geometry is not (the X6-50-L does not fit the current ankle layout; a caption says so).
* **Pose**: pick a preset and press *Apply preset*, or set the six joint angles with the sliders / number boxes (limits come
  from the config, a warning appears outside them). Pelvis pitch / roll tilt the whole leg frame (ground slope, torso lean).
* **Foot Contact Load**: *fraction x robot weight* (0.5 even double support, 1 single support), a
  *custom* fraction, or a *manual* Fz; plus Fx, Fy, a foot moment and a centre-of-pressure (CoP) shift in the foot frame.
  The Support toggle only changes the default fraction. The weight includes the optional pelvis payload.
* **Walking**: a row of buttons, one per keyframe of the stance phase. Choosing one sets the joint angles, ground
  force and CoP from that keyframe; the inputs they drive are disabled. *Follow Walking Cycle* off returns manual control.
* **Apply preset** sets the pose, support and vertical load and **clears Fx, Fy, the centre-of-pressure shift and the foot
  moment** (a "Load in use" line under *Foot Contact Load* always states what is applied). A *Not applied yet* note appears
  if the selected preset has not been applied.
* **Per-joint table**: *Own Share* is the motor load caused by that joint's torque alone. The two ankle joints share the two
  ankle motors, so each row shows only its own demand; the true combined load of each motor is in the *Actuators* card.
* **Save current pose as preset** appends a one-line preset (with an optional description, and the motion setting) to the
  `presets:` section of the YAML, keeping all comments.
* **Layout** ("drafting desk"): the left rail holds every input. The header has a status pill (green: every motor within
  its continuous rating; amber: over continuous or a safety factor under 2; red: over peak, safety factor under 1 or an
  analysis error). The *stage* shows the leg (Side / Front / 3D) with one readout per motor; the *load summary* beside it
  lists the busiest motor, the lowest safety factor, joint torques, the foot load, the ankle linkage and any flags. Below:
  *Actuators* (utilisation bars, motor table, ankle rod forces), *Per-joint summary*, *Link loads* (N, V, M, T along a
  link, stress, bearing reactions; includes the ankle rods), *Gait cycle* (load vs stance phase, peak / RMS table), *Hand
  check* (reproduces the ground-reaction torques from `Fz x lever arm`), *Export* and *Conventions*.
* **Light / dark**: both themes are defined in `.streamlit/config.toml`; switch in the app's settings menu. The cards'
  CSS uses `light-dark()` and the figures use transparent backgrounds with Streamlit's chart theme, so both follow the
  viewer's theme in the browser.

## 4. The model

**Frames and signs** (full detail at the top of `leg_config.yaml`)

* World / pelvis frame: x forward, y left, z up; origin on the pelvis centre line at the hip-pitch (J1) height. The left leg
  sits at y = +0.1375 m. Gravity is -z. Zero pose: leg straight down, foot flat, toe toward +x.
* Joint i frame = parent link frame, shifted by `origin.xyz`, rotated by `origin.rpy_deg`, then rotated by q about `axis`
  (right-hand rule). **Link i is the body distal to joint i**; its frame is joint i's frame after the rotation.
* Joints (positive direction): J1 hip flexion, J2 hip abduction (foot toward +y), J3 hip rotation, J4 knee flexion,
  J5 ankle dorsiflexion, J6 ankle roll (foot toward +y). Hip flexion, knee flexion and dorsiflexion are positive, so a squat
  has three positive angles (the design drawings use the opposite sign for hip flexion and plantarflexion; limits are converted).
* **Torque sign**: the torque the *actuator* applies to the distal link about the world-resolved joint axis to hold the
  pose. Only magnitudes are directly comparable; the sign gives direction.
* Foot load = the ground reaction ON the foot, in the world frame, at the contact point (sole point under the ankle, shifted by
  the CoP offset along the foot). The pelvis tilt does not rotate the force.
* Masses: a link's `mass` is its structure at `com` (`com: center` = half-way along its beam); every actuator and rod is a lumped
  point mass at its real position. The hip box and the hip-pitch actuator bodies are pelvis-side and not part of the leg.

**Equations** (symbols: `a_j` world joint axis, `p_j` joint origin, `J` foot Jacobian, `m` mass, `g` gravity vector)

| | |
|---|---|
| **A. Kinematics** | `T_i = T_{i-1} [R_o,i, p_o,i] Rot(a_i, q_i)`. A point on link k: `Jv_j = a_j x (p - p_j)`, `Jw_j = a_j` for `j <= k`, else 0. Foot Jacobian `J = [Jv; Jw]` (6xn) at `p_c = p_n + R_n (c_foot + [dx, dy, 0])`. Singularity: `cond(J)` (warn > 1e3, the rows mix m and rad) and reach `|p_c - p_0| / sum of segments` (warn > 99.5 %). |
| **B. Statics** | `tau = -( J^T W_grf + sum_m Jv_m^T (m_m g) )`, `W_grf = [F; M]` the ground wrench on the foot. Reported split: ground reaction / gravity. |
| **C. Transmission** | `qdot_act = T(q) qdot_joint`, so `f_act = T^-T tau_joint`; direct joint ratio N: `T = [N]`; differential `[[N,N],[N,-N]]`. Ratings are output-side, so the geared hip / knee actuators use ratio 1. Rotor inertia in the config is `N^2 x datasheet`; joint-space rotor inertia `= T^T diag(I) T`. |
| **Ankle linkage** | Two levers (75 mm) on the shank pivot about y-axes at `actuator_axis_z`; rod i joins lever tip `A_i + l (cos th, 0, -sin th)` to a foot pickup: `|pickup(q) - tip(th_i)| = rod_length_i`, solved by Newton; `T = d th / d (q_pitch, q_roll)` by central differences. At neutral the levers sit at `neutral_angle` (currently 4.9 deg up); pickups 83.7 mm ahead of J6, rods 274.1 / 204.1 mm, so the pitch ratio is about 1.11 and the roll ratio 0.28; the levers swing about -51 .. +47 deg and the lever-rod angle stays at or above 45 deg over the full range. No lever limits are enforced. |
| **D. Link loads** | Cut a link at station s (point c). Resultant on the proximal face = external wrench of everything distal: `F = F_grf + sum w_m`, `M = M_grf + (p_c - c) x F_grf + sum (p_m - c) x w_m` (masses of later links, plus this link's masses with beam coordinate >= s). In the beam frame: `N = F.x_b`, `Vy, Vz = F.y_b, F.z_b`, `T = M.x_b`, `My = M.y_b`, `Mz = M.z_b`. `sigma = Kt (abs(N)/A + abs(My) c_z/Iy + abs(Mz) c_y/Iz)`, `SF = yield/sigma` (weaker section at a boundary). Shear and torsion are reported, not in `sigma`. Bearing reaction = the s = 0 wrench in the joint frame (`M.a = -tau`). |
| **E. Limits** | `util = abs(f_act)/rating`. Green < 60 % of continuous, amber up to peak, red above peak. |
| **G. Ankle rods** | Each rod is a two-force member. Lever balance about its shaft gives the rod force `F = f_act / arm`, `arm = [(T - A) x u]_y` (lever tip `T`, pivot `A`, `u` = unit vector from the foot pickup to the tip; `F > 0` is tension and pulls the foot toward the lever). The foot receives `F u` at the pickup; by construction this reproduces the J5 / J6 joint torques (tested). A cut sees a rod when its foot end is distal and its lever end is proximal: lower shank below a lever pivot, ankle cross, foot between the ankle and the pickups, J5 / J6 bearings. Rods inside a free body (cuts above both pivots) add nothing, so the hip / knee / upper-shank results are unchanged. The ankle bearing axis moment is 0 (the rods carry the torque). |
| **Gait** | Keyframes over stance phase 0..1 (fraction of stance duration), linear interpolation; joint rates = `d q/d phase / stance_time` (smoothed); RMS over stance = plain mean of squares over the uniform sweep; whole-cycle RMS = stance RMS x sqrt(stance_fraction), swing unloaded. |

## 5. The configuration file

`leg_config.yaml` is validated on load (readable errors listing every bad path). Sections: `robot`, `joints` (axes, origins,
limits, link masses / sections / beams), `foot`, `actuators`, `actuator_options`, `transmissions`, `gaits`, `presets` (must
stay last). Cross-section kinds: `tube`, `rect_tube` (equal or per-direction walls), `parallel_plates`, `i_beam`, `rect`,
`generic`; a link may have several sections along its length. Anything guessed carries a `# PLACEHOLDER` tag on its line.

* `actuator_options` lets the UI swap a group of actuators; its `default` choice must equal the `actuators:` entries (checked).
* `gaits` are one stance phase each; `presets` may carry `motion: {type: gait, gait: <name>, phase: <0..1>}`.

## 6. Verification

`python -m pytest -q` (57 tests). Hand-checkable cases required by the specification: straight-leg torques and a hand-derived
inclined-leg hip torque; planar two-link `tau = J^T F` against an independent hand Jacobian; contact-offset ankle torque
`F d`; cantilever bending `F L`, linear in s; free-body balance (pelvis wrench and link-by-link equilibrium, `tau = -M.a`)
over random poses; joint-space mass matrix symmetric and positive definite; analytic vs numerical
Jacobian. Also: config validation messages, preset write-back, transmission types (ratio, table, differential, hook stub),
section properties by hand (plates, I-beam, hollow rectangle), piecewise sections, the foot beam override, the ankle
linkage (neutral ratio, rod lengths, feasibility over the full ankle range), the ankle rods (forces reproduce the joint torques,
lever balance, equilibrium unchanged outside the rod region, a hand-computed lower-shank cut), gaits (force references, sampling, sweeps,
RMS), the actuator swap, and headless end-to-end runs of the Streamlit app. `tools/fingerprint.py` guards against
accidental physics changes during refactors. Not covered: figure appearance and label placement (checked by eye), PNG export.

## 7. Known limitations

Quasi-static only (no leg inertia, no swing phase; hip and knee loads in gait are therefore under-stated). The ankle rods
are ideal two-force members (no end moments, rod mass not in the load path, the motor mount not resolved: the rod force enters
the shank at the lever pivot axis, so loads inside the motor-braced zone are lumped there; lever, rod and pickup stresses are
not evaluated and rod buckling is not checked).
Stress ignores torsion, shear and buckling; Kt = 1.5 is a guess. The foot plate is a beam toward the toe (a CoP behind the
ankle is only right at the mount). No torso model: the foot load is simply what the foot receives. The bundled gaits are
typical human data scaled to the robot, not robot data, and are heel-to-toe, not flat-foot. One leg only. See `REVIEW_NOTES.md`.

## 8. Extension hooks

* Full inverse dynamics: add `H qdd + C qd` in `statics.joint_torques` and `-m a` to the weights in `structure._wrench`
  (documented in `dynamics.inverse_dynamics_hook`).
* Knee linkage: `linkages.knee_linkage_ratio` is a stub; a `table` transmission needs no code.
* More hardware options: add an entry under `actuator_options:`.
