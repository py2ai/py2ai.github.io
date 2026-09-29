---
layout: post
title: "openpilot: An Operating System for Driver Assistance - Inside commaai/openpilot"
description: "A source-level tour of commaai/openpilot, the open-source driver assistance system that upgrades 300+ cars with adaptive cruise control and lane centering. We map the real-time process graph, from the manager supervisor and cereal pub/sub messaging bus to the controls loop and the neural driving model."
date: 2026-09-29
header-img: "img/post-bg.jpg"
permalink: /Openpilot-Source-Tour-Real-Time-Process-Graph/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/openpilot/commaai-openpilot-architecture.svg
tags:
  - OpenPilot
  - ADAS
  - Autonomous Driving
  - Architecture
categories: [AI, Open Source]
keywords: "openpilot, comma ai, driver assistance, ADAS, open source self-driving, cereal messaging, controlsd, plannerd, modeld, panda, opendbc, real-time systems, architecture, source code tour"
author: "PyShine"
---

Most people meet openpilot as a product: a comma device bolted behind the windshield, quietly holding a Honda or a Toyota at the center of its lane. Far fewer people have looked at what happens inside that device in the hundred milliseconds after a car ahead brakes. The answer lives in the source of [commaai/openpilot](https://github.com/commaai/openpilot), and it is one of the most instructive real-time codebases in the open-source world.

openpilot describes itself as "an operating system for robotics," and today it upgrades the driver assistance system in 300+ supported cars with adaptive cruise control (ACC) and automated lane centering (ALC), plus a camera-based driver monitoring feature. The project is written primarily in Python, with performance-critical daemons in C and C++, Cap'n Proto message schemas for everything that moves between processes, and the driving policy carried by a single end-to-end neural network. It is MIT-licensed, pinned to Python 3.12, and built with SCons.

Why is the source worth a tour? Because openpilot solves, in public, the exact problem that makes robotics software hard: coordinating dozens of soft-real-time processes that must never block each other, never lose their latest sensor frame, and never disagree about whether the system is engaged. Reading this repository teaches you a complete architecture pattern — a message bus, a supervisor, per-process rate control, and a neural perception path — that transfers to almost any embedded or robotics project.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/openpilot/commaai-openpilot-overview-architecture.svg" alt="Architecture overview of the commaai/openpilot repository" style="max-width:100%;height:auto;" />
</div>

*Overview of openpilot's runtime graph: a supervisor (manager) spawns every daemon onto the cereal pub/sub bus; vehicle I/O feeds neural perception, planning, and control; the UI and logger ride on the same message streams.*

Reading the overview from left to right: the process supervisor (`openpilot/system/manager/manager.py`) reads its process table (`openpilot/system/manager/process_config.py`) and launches every daemon onto the cereal messaging bus, whose service catalog (`openpilot/cereal/services.py`) and socket layer (`openpilot/cereal/messaging/__init__.py`) carry all traffic. On the vehicle side, `pandad` exchanges CAN traffic with the panda safety board and hands raw frames to `card`, the car interface. Perception flows through `modeld`, which turns camera frames into the `modelV2` neural output consumed by `plannerd`, `selfdrived`, and the 100 Hz `controlsd` loop. Finally, the UI and `loggerd` subscribe to the same live streams the controllers use, so what you see on screen and what gets logged is exactly what the system computed.

## Why You Need This

If you have ever tried to build anything that touches a real car, you know the gap between "I can send a CAN frame" and "I have a driver assistance system." openpilot closes that gap end to end. It gives you a fingerprinting and car-port layer (via the opendbc project, wired in at `openpilot/selfdrive/car/card.py`), a pedal-and-steering control stack, a vision model, a supervisor, and a logging pipeline — all designed to run unattended on a small ARM computer in a hot windshield mount. Studying it shows you what the missing fifty percent actually looks like.

The second reason is the messaging design. Most hobby robotics projects degenerate into callback spaghetti; openpilot avoids that with cereal, its pub/sub layer. Every piece of state — `can`, `carState`, `modelV2`, `longitudinalPlan`, `selfdriveState` — is a named service in `openpilot/cereal/services.py` with a declared frequency, log policy, and queue size. Nearly 80 services are catalogued this way, each backed by a Cap'n Proto schema in `openpilot/cereal/log.capnp`. If you want to learn how to make unrelated processes share fresh data safely, this is the cleanest working example you will find.

Third, there is the safety engineering. `docs/SAFETY.md` spells out the two core requirements: the driver must always be able to retake control instantly via the brake pedal or cancel button, and the actuators must never change the vehicle's trajectory faster than a human can react. The enforcing code lives in panda's firmware, written in C, and openpilot itself observes ISO 26262 guidelines. Reading how a software system organizes itself around a hardware safety boundary is a lesson that applies well beyond cars.

Finally, openpilot is simply a great codebase for learning how neural networks earn their keep in production. The driving model is not an appendix — it is the planner's primary input. Following the path from camera frames to steering torque shows you, concretely, how an ONNX network compiled for tinygrad sits inside a soft-real-time loop and drives a physical actuator.

## How It Works

Everything starts with one script: `launch_chffrplus.sh` prepares the environment and then runs `openpilot/system/manager/manager.py`, which becomes the sole authority over what runs on the device.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/openpilot/commaai-openpilot-architecture.svg" alt="Detailed architecture of the commaai/openpilot repository" style="max-width:100%;height:auto;" />
</div>

*Detailed view of the openpilot process graph: the manager and its process table supervise all daemons; pandad and card bridge to the panda board and opendbc car ports; modeld's neural path feeds the planner, the engagement state machine, and the 100 Hz controls loop, while localization daemons refine the state estimate.*

### Understanding the Architecture

**The supervisor and its process table.** `openpilot/system/manager/manager.py` subscribes to `deviceState`, `carParams`, and `pandaStates`, and on every tick calls `ensure_running()` over the daemon list defined in `openpilot/system/manager/process_config.py` — more than 40 processes, each declared as a `PythonProcess`, `NativeProcess`, or `DaemonProcess` with a start condition such as `only_onroad` or `iscar`. This is how openpilot implements ignition-aware lifecycle management: when the device goes on-road, the driving daemons spin up; when it parks, they are stopped and the `updated` and `uploader` processes take over. The manager also clears parameter groups on on-road/off-road transitions using the shared key-value store in `openpilot/common/params.py`.

**The cereal bus.** All daemons talk through cereal. In `openpilot/cereal/services.py`, every service is declared as a tuple of `(should_log, frequency, decimation, queue_size)` — for example `can` runs at 100 Hz with a 10 MB queue for high-rate traffic, while `modelV2` publishes at 20 Hz with a large queue because it carries big AI outputs. The `SubMaster` and `PubMaster` classes in `openpilot/cereal/messaging/__init__.py` wrap the msgq C++ transport: subscribers get conflated (latest-wins) sockets, and a `FrequencyTracker` continuously checks each input for aliveness, correct frequency, and message validity, so a process like `controlsd` can call `sm.all_checks()` and know whether its inputs are trustworthy.

**The vehicle boundary.** `pandad` (`openpilot/selfdrive/pandad/pandad.cc`, C++) owns the connection to the panda board over USB or SPI, publishing `can` frames and `pandaStates`, and transmitting outgoing actuator messages that arrive on the `sendcan` topic. The safety model that hard-limits those messages is compiled into panda's firmware as C code, derived from the per-brand rules in the opendbc project — panda is a hardware backstop that openpilot's own crashes cannot disable. Above it, `card` (`openpilot/selfdrive/car/card.py`) uses `get_car()` from opendbc to fingerprint the vehicle, publishes `carState`, `carParams`, `carOutput`, and `radarTracks`, and translates `carControl` messages back down to `sendcan`.

**The neural driving model path.** `camerad` (`openpilot/system/camerad`) captures the road-facing and driver-facing cameras and ships frames over VisionIPC. `modeld` (`openpilot/selfdrive/modeld/modeld.py`) runs the driving network — the weights ship as `driving_supercombo.onnx` under `openpilot/selfdrive/modeld/models/` and are compiled and executed with tinygrad — and publishes three topics: `modelV2` (the plan: desired curvature and acceleration over the next seconds, plus lane lines and lead vehicles), `drivingModelData` for the UI, and `cameraOdometry` for the pose filter. In parallel, `dmonitoringmodeld` runs `dmonitoring_model.onnx` to produce `driverStateV2`, which `dmonitoringd` and its policy module (`openpilot/selfdrive/monitoring/policy.py`) turn into `driverMonitoringState` — the distraction alerts that gate engagement.

**Engagement, planning, and control.** `selfdrived` (`openpilot/selfdrive/selfdrived/selfdrived.py`) is the engagement authority: it runs the state machine from `openpilot/selfdrive/selfdrived/state.py` (disabled, enabled, soft-disabling, overriding) and evaluates the large on-road event catalog in `openpilot/selfdrive/selfdrived/events.py` against inputs like driver monitoring state and car faults, publishing `selfdriveState` and `onroadEvents`. `plannerd` polls `modelV2`, runs the MPC-based planner in `openpilot/selfdrive/controls/lib/longitudinal_planner.py` (assisted by `radard`'s lead tracking in `openpilot/selfdrive/controls/radard.py`), and publishes `longitudinalPlan`. Then `controlsd` (`openpilot/selfdrive/controls/controlsd.py`) closes the loop at 100 Hz (`DT_CTRL = 0.01` in `openpilot/common/realtime.py`): it combines the model's desired curvature with `carState`, clamps it through `clip_curvature`, computes steering via the lateral controllers (`openpilot/selfdrive/controls/lib/latcontrol.py`, instantiated as angle, curvature, PID, or torque variants depending on the car) and longitudinal accel via `openpilot/selfdrive/controls/lib/longcontrol.py`, and publishes `carControl` back to `card`. Real-time discipline is explicit throughout: processes pin their cores and priorities with `config_realtime_process()`, and `Ratekeeper` flags any loop that misses its cadence.

**The end-to-end flow.** Put it together and a single control cycle reads like this: panda delivers safety-filtered CAN → `card` publishes `carState` → `camerad` feeds `modeld`, which publishes `modelV2` → `plannerd` fuses that with radar lead data into `longitudinalPlan` while `selfdrived` decides whether the system may stay active → `controlsd` merges all of it at 100 Hz into `carControl` → `card` converts that into `sendcan`, which `pandad` hands to the panda, which applies the final, firmware-enforced actuator limits before anything reaches the steering rack. Every arrow in that chain is a cereal topic, every topic has an owner, and every process knows exactly which inputs must be alive for its output to be valid.

## Advantages

- **A real process graph, not a framework.** The entire runtime is a plain Python list in `openpilot/system/manager/process_config.py`; you can read the whole system's topology in one file and start any piece standalone.
- **Declared messaging contracts.** Frequencies, log policies, and queue sizes live in `openpilot/cereal/services.py`, and `SubMaster` enforces them at runtime, so a misbehaving publisher is detected instead of silently tolerated.
- **Safety as an architecture layer.** The two-rule safety model in `docs/SAFETY.md` is enforced in C on independent hardware (panda), demonstrating how to make a software failure unable to cause an actuator-level failure.
- **Neural-first design.** The driving network's output (`modelV2`) is the primary planning input, giving you a rare production example of an end-to-end model wired into hard actuation paths with classical safety checks around it.
- **Cross-platform from the start.** The same code runs on comma hardware and on a PC, with a MetaDrive simulator bridge in `openpilot/tools/sim/`, so you can develop without a car.
- **Data pipeline included.** `loggerd` and `encoderd` record every subscribed service and camera stream into routes, and `uploader` ships them — the full tooling behind a data-driven development loop.

## Benefits

- **Learn soft real-time engineering.** The constants in `openpilot/common/realtime.py` (`DT_CTRL = 0.01`, `DT_MDL = 0.05`), core pinning, and SCHED_FIFO priorities show exactly how to budget a latency-critical stack across cores.
- **A reusable pub/sub pattern.** Cereal's latest-wins subscriptions, Cap'n Proto zero-copy messages, and validity tracking transfer directly to drones, robots, and industrial controllers.
- **Car integration demystified.** Fingerprinting, DBC-driven parsing, per-brand interfaces, and firmware versioning queries in the opendbc layer explain what commercial ADAS integrators actually spend their time on.
- **Driver monitoring done properly.** The `dmonitoringmodeld` → `dmonitoringd` → `selfdrived` chain shows how a vision model's output should feed policy and state machines rather than actuators directly.
- **A trustworthy testing culture.** Software-in-the-loop tests run on every commit, replay tooling lives in `openpilot/tools/replay/`, and the safety-critical panda code has its own dedicated test suite.
- **MIT-licensed freedom to fork.** Every layer, from model weights loading to UI rendering, can be studied, modified, and redeployed under the MIT license.

## Usage

On a comma device, the quick start from the README is a one-liner:

```bash
bash <(curl -fsSL openpilot.comma.ai)
```

On the device's settings screen you point the custom software URL at a branch — the release channel or the bleeding edge:

```text
openpilot.comma.ai          # release branch
openpilot-nightly.comma.ai  # nightly development branch
```

To try openpilot without a car, use the built-in MetaDrive simulator bridge from the repository:

```bash
# terminal 1: launch openpilot against the simulator
./openpilot/tools/sim/launch_openpilot.sh

# terminal 2: run the bridge (keys: 1 accel, 2 decel, S brake, q quit)
cd openpilot/tools/sim
./run_bridge.py --joystick
```

For development on a PC, clone the repo, install dependencies with the pinned `uv` environment (Python 3.12), and build the native components with SCons (`scons` at the repository root) before launching the same `launch_openpilot.sh` entry point.

## Conclusion

openpilot is more than an ADAS tweak — it is a complete, working answer to the question "how do you organize a real-time robotics system around a neural network and a safety boundary?" The manager gives it a nervous system, cereal gives it a language, pandad and opendbc give it hands on the wheel, modeld gives it eyes, and controlsd gives it reflexes at 100 Hz. Whether your interest is autonomous driving, embedded architecture, or production ML systems, there is a tour-worthy lesson waiting in almost every directory.

Links:

- GitHub repository: https://github.com/commaai/openpilot
- Documentation: https://docs.comma.ai
- Supported cars: https://github.com/commaai/openpilot/blob/master/docs/CARS.md
- Safety model: https://github.com/commaai/openpilot/blob/master/docs/SAFETY.md
