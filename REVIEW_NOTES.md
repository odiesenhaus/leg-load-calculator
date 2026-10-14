# Review notes: where to look for mistakes

This tool was built iteratively with the user, often changing inputs between steps. The goal of a review is to find
places where we **moved on before the fundamentals were settled**. Items are ordered by how much a wrong answer would
change the conclusions (is the ankle motor big enough, is the bracket strong enough). `README.md` states the model; this
file states what is *uncertain*.

## Scope and trust map (read this first)

**Purpose:** confirm the actuator selection and get a first sense of the forces and moments the structural parts see, for a
walking-down-a-hallway robot that must also squat and do single stair step-ups. The tool is a *foot-wrench -> joint-torque*
calculator: nearly every headline result is `ground force x lever arm`, so it is only as good as the force and where it acts.

**Trust (checkable by hand; error comes from inputs only):** static joint torques for a given pose and wrench; actuator
utilisation in the static presets (the *Actuators* tab has a hand check); the ankle lever ratio T; bending in the thigh and
upper shank; J1 / hip-box reaction.

**Trust to a factor of about 1.5-2, as a trend:** gait peak torques (the CoP range and angles are placeholders); safety
factors (yield only, Kt guessed, no dynamic factor, fatigue not checked).

**Trust with caveats (rods added 2026-10-11):** lower shank / ankle cross / foot / J5-J6 bearing loads now include the ankle
rods as two-force members (verified: the rod forces reproduce the J5 / J6 torques; hip / knee / upper-shank numbers are
unchanged). At walking push-off the J5 bearing load is 688 N (was 379 N), the ankle cross bends 5 N*m (was 45), and the shank
peak moves into the motor zone (39 N*m at the upper pivot; was 45 at the weak lower end). Still ideal: no rod-end moments,
the motor mount is not resolved (force enters at the pivot axis), lever / rod / pickup stresses and rod buckling are not
checked.

**Do not trust:** hip and knee torque in gait (no leg inertia: a rough estimate puts the swing hip near or above its continuous rating).

**Fixed 2026-10-11:** (1) Apply Preset left Fx / Fy / CoP from the last walking keyframe in place, so Stand and Stair Step Up
showed ankle motors at 200-360 % of continuous and the hip over peak; they are now reset. (2) The per-joint table and diagram
labels gave each ankle joint the worse of the two shared ankle motors (so a joint with zero torque showed red); each joint now
shows its own share.

**Removed on 2026-10-09 (out of scope or no real basis; the pre-trim copy is in `../leg_calc_backup_20261009/`):** the Impact
tab and `impact_estimate` (needed an uncertain rotor-inertia reading, fixed base, produced no force); the gait speed / power /
whole-cycle RMS / torque-speed envelope outputs and the joint-velocity input (built on placeholder keyframes); the jump-landing
and stand-speed presets and `motion.py`; the running gait; the Jacobian condition-number warning (rows mix m and rad).
The remaining 526 fingerprint numbers were checked identical to the pre-trim values.

**Planned next (in this order):** combined stress with torsion and shear plus fatigue-based
allowables; a load-case table (stand, single leg, squat, step-up, walking peak) with stated dynamic factors; acceleration
headroom and a mass-budget view.

## 0. How to check quickly

```
python -m pytest -q                       # 57 tests (~30 s)
python tools/fingerprint.py save a.json   # numerical fingerprint of every preset and gait; compare after any change
python tools/fingerprint.py compare a.json
streamlit run app.py
```

The required hand-checkable physics tests are in `tests/test_physics.py` (the impact ones were removed with the impact estimate). The tests prove the *implementation* matches the
stated equations; they cannot prove the equations or the inputs are right. Scratch images in `exports/` are my renders for
visual checks (gitignored, safe to delete).

## 1. Inputs that conflict or are guesses (check against CAD / the user)

| # | Item | What we did | Why it is shaky |
|---|---|---|---|
| 1 | **Ankle rod lengths** | Used 269.85 / 200.67 mm | The user quoted 240 / 170 mm, but with their other facts (levers 75 mm perpendicular to the shin at neutral, lever pivots at the actuator COM heights 102.5 / 172.5 mm below the knee, foot pickups at the roll-axis height, 39.5 mm ahead, +-21 mm) the rods must be ~30 mm longer. We trusted the geometry; the ankle ratio is almost the same either way (~0.56 vs 0.57). Unresolved. |
| 2 | **Lever pivot = actuator COM height** | pivots 250 / 180 mm above J5 | Assumes the output axis is at the motor COM height; `lever_y` (lateral) is assumed equal to the pickup y. |
| 3 | **Hip width / leg lengths** | hip width 275 mm (J2 at y = 137.5 mm); J1->J4 320 mm, J4->J5 352.5 mm | The first drawing said 125 mm from the centre line to J2 and the user's earlier text said calf 0.32 m. They later confirmed 275 mm; the 352.5 mm calf comes from the drawing. |
| 4 | **J1 flange position** | y = 67.5 mm (outer face of the pitch actuator) | Assumed from the drawing; it sets the J1 -> J2 offset (70 mm) and the bearing tilt moment at J1. |
| 5 | **Foot** | plate 220 x 110 x 10 mm, ankle 55 mm from the heel, J6 24 mm above the plate (sole 34 mm below J6) | Conflicting values were given at different times (270 x 50, 230, 120); the latest were adopted. |
| 6 | **Ankle cross 1.333 kg** | structure mass | Suspiciously close to the X8-150 actuator mass (1.33 kg); confirm it is a structure and not an actuator body counted twice or missing. |
| 7 | **Thigh structure split** | 0.445 / 0.555 kg (above / below J3) | The user gave "about 0.555 kg for the bottom section". |
| 8 | **Rotor inertia** | `N^2 x datasheet inertia`, datasheet treated as ROTOR-side | The datasheets disagree in style (X6-60 0.66 kg*cm^2 vs X6-50-L 0.0019 kg*m^2 vs X8-150 1.023e-3 kg*m^2). The user said the numbers are right; the rotor-side reading is ours. Only the impact estimate uses it. |
| 9 | **Kt = 1.5, 6061-T6 276 MPa** | all sections | Guessed Kt; yield quoted is a typical minimum. |
| 10 | **Motor-braced shank zone starts at 75 mm** | solid 68 x 55 block 75-197 mm | Start inferred from the motor diameter; end (197 mm) was given. The bottom 23 mm hollow rectangle's outer size (68 x 55) is assumed. |
| 11 | **X6-50-L swap** | ratings + weight only | 80 mm diameter motors do not fit 70 mm axis spacing / 55 mm plates; geometry not redone. |

## 2. Modelling decisions to challenge

1. **Ground force** is an input (`fraction x total mass x g` per leg, vertical, at the contact point). No torso or pelvis is
   modelled, so nothing checks that the implied pelvis wrench is feasible. A torso-as-projected-load model was proposed and
   shelved. In single support the fraction 1.0 equals the whole robot weight including this leg; the leg's own weights are
   then added as gravity torques (equilibrium is consistent, but check the reasoning in `statics.py`).
2. **Torque sign / transmission convention** (`statics.py`, `actuators.py`): `tau` is the actuator's holding torque,
   `f = T^-T tau`, `T` maps joint rate to actuator OUTPUT rate. Check the sign chain end to end, in particular for the ankle.
3. **Link-load cut** (`structure.py`): pure statics with point masses; the foot GRF is distal only to cuts on the root side of the
   contact point along the beam; the foot beam runs ankle -> toe at the plate mid-plane. Stations are added at masses and
   section changes. Check zero-length links, the `whole_link` rule at s = 0, and the beam-frame orientation (`y_ref`).
4. **Section properties** (`config._section`): `parallel_plates` uses a composite parallel-axis `Iz` (right for a pure moment,
   wrong for lateral shear, where each plate bends alone) and an open-section `J`; torsion and shear are not in the stress.
5. **Ankle rods** (now modelled, `leg/rods.py`): check the sign convention (tension +, force on the foot = F u with u pointing
   from the pickup to the lever tip), that the lever pivot is the right point to apply the rod reaction to the shank, and that
   the actuator torque the rods deliver (f_act from T^-T tau) is the torque about the lever shaft. Not checked: rod buckling,
   end-bearing loads, lever and pickup stresses, rod-end moments.
6. **Hip box** is treated as pelvis-side and not loaded into the leg; the J1 flange reaction is the load for the box.

## 3. Gait and motion presets (newest, least reviewed)

* The bundled gaits are **typical adult human shapes scaled to the robot**, with one given anchor: walking ~1 body weight
  within 150 ms of heel strike. Everything else (angles, fore-aft and lateral
  shares, CoP path, stance time 0.60 s, stance fraction 0.60) is a placeholder. Hip adduction of -4 deg with the
  ankle compensating was chosen so the hip abductor carries a realistic load.
* They are **heel-to-toe**, not flat-foot, gaits: the foot pitches to about -60 deg at toe-off and the CoP travels 165 mm.
  The user plans a flat-foot gait, where the CoP range (hence ankle torque) is the key unknown.
* Keyframe **phase is a fraction of the stance duration**. The 150 ms anchor is only exact at stance_time = 0.60 s. Stance time
  changes only joint speeds, never torques.
* Joint speeds come from smoothed keyframe differences; the torque-speed envelopes are linear approximations from datasheet
  numbers. Whole-cycle RMS assumes an unloaded swing (optimistic).

## 4. Not covered by tests

Figure appearance and label placement, the 3D arrow geometry, the exported PNG, UI layout. The headless app tests only check
that the app runs and that displayed numbers follow the inputs.

## 5. Known results that depend on the above

With the default X4-36-L ankle and the human-like walking gait the ankle actuators run at ~128 % of peak and ~170 % of
continuous RMS; with the X6-50-L they fit. These
conclusions are only as good as items 1-2 of section 1 and the CoP excursion in the gait data.

## 6. Questions still open with the user

Rod lengths vs geometry (1.1); the CoP range / stance timing of the intended flat-foot gait (3); whether the ankle cross is
structure only (1.6); a torso model; real torque-speed curves and thermal data for the actuators; the CAD pickup coordinates.
