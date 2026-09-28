---
layout: post
title: "PLFM_RADAR: An Open-Source 10.5 GHz Phased-Array Radar You Can Build - Inside NawfalMotii79/PLFM_RADAR"
description: "PLFM_RADAR (AERIS-10) is an open-source 10.5 GHz pulse-LFM phased array radar published with complete schematics, Verilog FPGA firmware, STM32 supervisor code, and a PyQt6 ground station. We tour the repository's real architecture: chirp generation, the FPGA receive chain, the 64x32 range-Doppler pipeline, and the cross-layer contract tests that hold it all together."
date: 2026-09-28
header-img: "img/post-bg.jpg"
permalink: /PLFM-RADAR-AERIS-10-Open-Source-Phased-Array-Radar/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/plfm-radar/nawfalmotii79-plfm-radar-architecture.svg
tags:
  - Radar
  - FPGA
  - Open Source Hardware
  - Embedded Systems
categories: [AI, Open Source]
keywords: "open source radar, phased array radar, AERIS-10, PLFM_RADAR, 10.5 GHz radar, FPGA signal processing, Verilog radar, STM32 firmware, pulse compression, Doppler FFT, CFAR detection, CERN-OHL-P, PyQt6 ground station, FT2232H, radar bring-up"
author: "PyShine"
---

Radar has always been one of those fields where the barrier to entry is not intelligence but access. The theory is in every textbook; the hardware is locked behind defense suppliers and six-figure price tags. So when a repository appears that ships the schematics, PCB layouts, Verilog, C firmware, and desktop application for a genuine 10.5 GHz phased array radar, it deserves attention. That repository is [NawfalMotii79/PLFM_RADAR](https://github.com/NawfalMotii79/PLFM_RADAR), home of the AERIS-10 project by Nawfal Motii.

AERIS-10 is an open-source pulse Linear Frequency Modulated (PLFM) phased array radar operating at 10.5 GHz. The repository documents two build variants: a 3 km-range "Nexus" version with an 8x16 patch antenna array, and a 20 km-range Extended version using a 32x16 dielectric-filled slotted waveguide array with sixteen QPA2962 GaN power amplifier boards. Both steer their beam electronically to ±45 degrees in elevation and azimuth, and scan 360 degrees mechanically with a stepper motor and slip ring.

What makes the source worth a tour is that it is a complete, layered engineering artifact rather than a proof-of-concept dump. The repo is organized into numbered top-level directories — from `1_Project_Description` through `9_Firmware` — covering power management spreadsheets, board schematics, RF simulations, component datasheets, and a full firmware tree of roughly eighty C files, seventy-seven Verilog modules, and fifty-five Python files, all talking to each other through documented, testable contracts.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/plfm-radar/nawfalmotii79-plfm-radar-overview-architecture.svg" alt="Architecture overview of the NawfalMotii79/PLFM_RADAR repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the PLFM_RADAR repository: hardware design documents feed the MCU and FPGA firmware layers, which stream data into the Python host software, with a dedicated verification layer watching every interface.*

Reading the overview from left to right: the hardware design group defines the physical system — the main board schematic (`4_Schematics and Boards Layout/4_6_Schematics/MainBoard/RADAR_Main_Board.sch`) and the power sequencing workbook (`3_Power Management/Power Management V6.xlsx`). The MCU layer, anchored by `9_Firmware/9_1_Microcontroller/9_1_3_C_Cpp_Code/main.cpp`, supervises the analog world and hands chirp and beam timing to the FPGA. The FPGA layer is the DSP workhorse: `9_Firmware/9_2_FPGA/radar_system_top.v` instantiates the receiver chain and streams frames out through USB. The host layer turns that byte stream into a product — `9_Firmware/9_3_GUI/radar_protocol.py` owns the wire format, and the PyQt6 dashboard in `9_Firmware/9_3_GUI/v7/` turns frames into maps and tracks. Finally, the verification group binds every interface: an iverilog regression runner in the FPGA folder and cross-layer contract tests under `9_Firmware/tests/cross_layer/` that check the Python, Verilog, and C layers against each other.

## Why You Need This

If you are teaching or learning radar signal processing, this repository is the missing bridge between block diagrams and silicon. Most courses stop at "and then the pulse compression happens." Here you can read `9_Firmware/9_2_FPGA/matched_filter_processing_chain.v` and see exactly how a matched filter is pipelined in hardware, how the Doppler FFT is organized, and how `cfar_ca.v` turns a magnitude map into detection flags — all with committed golden test vectors to check your understanding against.

If you are a drone developer or an SDR hobbyist, AERIS-10 solves the "where do I even start" problem. The bill of materials and production outputs are co-located under `4_Schematics and Boards Layout/4_7_Production Files`, the antenna designs ship as board files, and the docs folder carries a bring-up guide served via GitHub Pages. You do not have to reverse-engineer a closed box to know which voltage rail comes up first — the power management workbook sequences it, and the STM32 firmware enforces it.

If you are a firmware or verification engineer, the project is a case study in cross-layer discipline. A radar system has three independent codebases — Verilog, C, and Python — that must agree on opcodes, bit widths, and packet layouts. Most hobby projects let those constants drift apart; PLFM_RADAR enforces them with a three-tier contract test suite and an FPGA regression runner, the kind of infrastructure usually seen in commercial teams.

Finally, if you simply want a hackable sensing platform, the design is deliberately modular. The FPGA exposes a full register map — chirp timing, CFAR parameters, MTI, DC-notch width, AGC tuning — so you can reconfigure the radar at runtime, or replay recorded data through a bit-accurate software replica of the signal chain without touching hardware at all.

## How It Works

The system splits cleanly into four cooperating layers — hardware design, MCU supervision, FPGA signal processing, and host software — stitched together by an unusual depth of test infrastructure.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/plfm-radar/nawfalmotii79-plfm-radar-architecture.svg" alt="Detailed architecture of the NawfalMotii79/PLFM_RADAR repository" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture of PLFM_RADAR: the STM32 supervisor configures clocks, synthesizers, and beamformers; the FPGA runs the transmit and receive DSP chain; the Python host renders and tracks; verification tools bind the layers together.*

### Understanding the Architecture

**The FPGA is the real-time core.** `9_Firmware/9_2_FPGA/radar_system_top.v` integrates the transmitter and receiver with a dual-mode USB interface selected by a `USB_MODE` parameter — FT601 USB 3.0 for the 32-bit premium board, FT2232H USB 2.0 for the 8-bit production board. On transmit, `radar_transmitter.v` drives the DAC through `plfm_chirp_controller.v`, which plays PLFM chirp waveforms from committed `.mem` files. On receive, `radar_receiver_final.v` orchestrates the chain: the AD9484 ADC interface feeds `ddc_400m.v` with its NCO and CIC decimator, then the matched filter chain, range bin decimation, the Doppler processor with its FFT engine, the MTI canceller, and finally `cfar_ca.v` for constant-false-alarm-rate detection. `rx_gain_control.v` closes the AGC loop; `radar_mode_controller.v` sequences operating modes.

**The MCU is the supervisor, not the DSP.** `9_Firmware/9_1_Microcontroller/9_1_3_C_Cpp_Code/main.cpp` on the STM32F746 handles everything that must be sequenced and calibrated rather than computed at line rate. It brings up power in the order defined by the power management workbook, configures the AD9523-1 clock generator (`9_1_1_C_Cpp_Libraries/ad9523.c`), the ADF4382 synthesizers (`adf4382a_manager.c`), and the four ADAR1000 beamformer chips (`ADAR1000_Manager.h`) that phase-steer all sixteen elements. It also closes a per-channel calibration loop: two ADS7830 I2C ADCs sense the current of each PA channel through shunt resistors, while two DAC5578s adjust gate voltages until every channel matches its target at boot. GPS arrives from a UM982 module (`9_1_3_C_Cpp_Code/um982_gps.c`), attitude from a GY-85 IMU.

**The protocol layer is a single source of truth.** `9_Firmware/9_3_GUI/radar_protocol.py` documents the exact wire format shared with `usb_data_interface_ft2232h.v`: eleven-byte `0xAA`-framed data packets carrying one 64x32 range-Doppler frame's worth of I/Q and detection data, twenty-six-byte `0xBB` status packets, and four-byte host commands of the form `{opcode, addr, value_hi, value_lo}`. Its `Opcode` enum mirrors the FPGA's command table — radar mode, chirp and listen cycle counts, CFAR parameters, AGC tuning, self-test triggers — and the module is deliberately dependency-free so tests can import it without a GUI.

**The host software is a real ground station.** `9_Firmware/9_3_GUI/GUI_V7_PyQt.py` launches `v7/dashboard.py`, a six-tab PyQt6 application: a live range-Doppler canvas, an embedded Leaflet map (`v7/map_widget.py`) that centers on GPS fixes and draws target trails, a full FPGA register control panel, an AGC monitor fed by `v7/agc_sim.py` (a bit-accurate Python replica of the RTL gain logic), diagnostics, and settings. Workers in `v7/workers.py` hand frames to `v7/processing.py`, which fuses consecutive CPIs, unwraps velocity across PRFs, clusters detections with DBSCAN, and runs Kalman tracking. Best of all, `v7/software_fpga.py` implements the entire signal chain in NumPy, so `v7/replay.py` can re-process raw IQ captures or HDF5 recordings through identical settings without hardware.

**The verification stack is the hidden gem.** `9_Firmware/9_2_FPGA/run_regression.sh` lints the production RTL with Vivado-class checks, then runs the iverilog testbench suite in phases — including exact-match comparisons against committed golden vectors generated from real data, and an end-to-end system test. `9_Firmware/tests/cross_layer/test_cross_layer_contract.py` goes further with three tiers: statically parsing the Python, Verilog, and C sources to catch opcode and bit-width drift, co-simulating the FT2232H interface in Verilog, and executing a compiled C stub to confirm the STM32 settings parser agrees with Python-generated packets. A GitHub Actions workflow covers the Python side on every pull request.

Follow one detection end to end: the STM32 signals a new elevation step, `radar_system_top.v` triggers `plfm_chirp_controller.v` to play a chirp through the DAC while the ADAR1000s apply the programmed phases; the echo returns through the AD9484 interface, is down-converted and decimated in `ddc_400m.v`, compressed by the matched filter chain, Doppler-processed by `doppler_processor.v` and `fft_engine.v`, clutter-cancelled by `mti_canceller.v`, and thresholded by `cfar_ca.v`. The frame is packed into an `0xAA` packet by the USB interface, parsed by `radar_protocol.py`, assembled by `v7/workers.py`, tracked by `v7/processing.py`, and plotted on the range-Doppler canvas and Leaflet map — with the STM32 streaming GPS so every detection can be geolocated.

## Advantages

- **Genuinely complete design package.** Schematics, stack-ups, gerbers, BOMs, mechanical drawings, antenna designs, and RF simulations ship in one repository — not just a block diagram and a promise.
- **Hardware/software co-design done visibly.** The FPGA register map, the STM32 sequencing logic, and the GUI control panel all agree because shared protocol code and contract tests pin them together.
- **Bit-accurate software twin.** `v7/software_fpga.py` and `v7/agc_sim.py` replicate the RTL numerically, enabling offline development, replay, and register tuning without hardware.
- **Verification-first culture.** Golden-vector co-simulation with real captured data, an end-to-end system testbench, three-tier cross-layer tests, and CI linting — rare rigor for open hardware.
- **Dual-variant scalability.** The same firmware tree targets the 3 km patch-array Nexus build and the 20 km waveguide-array Extended build, with both USB 2.0 and USB 3.0 data paths.
- **Readable, self-documenting source.** Module headers state clock domains, packet formats, and design intent; the numbered directory scheme maps physical to logical cleanly.

## Benefits

- **Learn radar by reading real RTL.** Pulse compression, Doppler FFT, MTI, CFAR, and AGC all have concrete, tested implementations you can study, modify, and re-simulate.
- **Lower the cost of experimentation.** Open schematics and BOMs mean you can fabricate, modify, or extend the platform instead of buying a closed unit or starting from a blank page.
- **Bring up hardware with confidence.** Power sequencing workbooks, a Tk bring-up dashboard, an FPGA self-test triggered by one opcode, and a UART diagnostic capture tool are all aimed at board-day.
- **Reuse the patterns elsewhere.** The cross-layer contract test approach and the software-FPGA replay architecture transfer directly to any multi-language embedded project.
- **Build applications, not just demos.** GPS/IMU-corrected, geolocated target tracking with map integration is a starting point for real sensing applications.
- **Contribute to a growing ecosystem.** The project openly requests RF, FPGA, and software contributors, and its documentation is improvable via GitHub Pages.

## Usage

The host software requires Python 3.8+ (the repository's own tooling targets 3.12); the FPGA flow needs Xilinx Vivado. Install the V7 ground station dependencies and launch it:

```bash
pip install -r 9_Firmware/9_3_GUI/requirements_v7.txt
python 9_Firmware/9_3_GUI/GUI_V7_PyQt.py
```

For the lightweight Tkinter bring-up dashboard and its dependencies:

```bash
pip install -r 9_Firmware/9_3_GUI/requirements_dashboard.txt
python 9_Firmware/9_3_GUI/GUI_V65_Tk.py
```

Run the host-side board bring-up smoke test, which triggers the FPGA self-test via opcode `0x30` and reads results back via `0x31` (use `--live` against real FT2232H hardware, otherwise it runs in mock mode):

```bash
python 9_Firmware/9_3_GUI/smoke_test.py
python 9_Firmware/9_3_GUI/smoke_test.py --live --adc-dump adc_raw.npy
```

Run the full FPGA regression suite (lint plus iverilog testbenches) from the FPGA directory, and the cross-layer contract tests:

```bash
cd 9_Firmware/9_2_FPGA
./run_regression.sh [--quick] [--skip-lint]
```

```bash
pytest 9_Firmware/tests/cross_layer/test_cross_layer_contract.py
```

For a hardware build, the README directs you to order PCBs from the production outputs under `4_Schematics and Boards Layout/4_7_Production Files`, source components from the co-located BOM/CPL files, and assemble using the schematics in `4_6_Schematics`.

## Conclusion

PLFM_RADAR is a reminder of what open hardware can be when someone finishes the whole job. The AERIS-10 repository does not just describe a phased array radar — it contains the schematics, the RTL, the firmware, the protocol, the ground station, and the tests that prove they all agree. Whether you want to learn radar DSP from real Verilog, build a 10.5 GHz sensing platform, or borrow its cross-layer verification discipline, the source is approachable, honest about its alpha status, and actively developed. Check the issues page for known limitations before planning a build — and if you have RF or FPGA experience, the project is explicitly looking for reviewers.

Links:

- GitHub repository: [NawfalMotii79/PLFM_RADAR](https://github.com/NawfalMotii79/PLFM_RADAR)
- Documentation (GitHub Pages): [NawfalMotii79.github.io/PLFM_RADAR/docs/](https://NawfalMotii79.github.io/PLFM_RADAR/docs/)
- Contributing guidelines: [CONTRIBUTING.md](https://github.com/NawfalMotii79/PLFM_RADAR/blob/main/CONTRIBUTING.md)
