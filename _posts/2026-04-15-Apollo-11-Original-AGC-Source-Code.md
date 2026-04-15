---
layout: post
title: "Apollo 11: Original AGC Source Code - A Historic Software Heritage"
description: "Explore the original Apollo 11 Guidance Computer (AGC) source code that powered humanity's first Moon landing. Learn about the Command Module (Comanche 055) and Lunar Module (Luminary 099) software architecture."
date: 2026-04-15
header-img: "img/post-bg.jpg"
permalink: /Apollo-11-Original-AGC-Source-Code/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - Open Source
  - History
  - Software Heritage
  - Space
  - AGC
author: "PyShine"
---

# Apollo 11: Original AGC Source Code - A Historic Software Heritage

The Apollo 11 Guidance Computer (AGC) source code represents one of the most significant achievements in software engineering history. This repository contains the original source code that guided humanity's first Moon landing on July 20, 1969. With over 66,000 stars on GitHub, it stands as a testament to the pioneering work of Margaret Hamilton and the MIT Instrumentation Laboratory team.

## Overview

The Apollo-11 repository preserves the original AGC source code for both the Command Module (Comanche 055) and the Lunar Module (Luminary 099). This code was written in AGC Assembly Language, a specialized low-level language designed specifically for the Apollo Guidance Computer's unique architecture.

![AGC Architecture](/assets/img/diagrams/apollo-11/apollo-agc-architecture.svg)

### Understanding the AGC Architecture

The architecture diagram above illustrates the layered structure of the Apollo Guidance Computer system. This design represents one of the earliest examples of real-time embedded systems programming.

**Hardware Layer: Apollo Guidance Computer**

The AGC was a revolutionary piece of hardware for its time, featuring:

- **16-bit CPU**: Running at approximately 2 MHz, the processor executed instructions at a rate that seems primitive by modern standards but was sufficient for the complex navigation tasks required.

- **Memory Architecture**: The AGC had 36K words of read-only memory (ROM) for program storage and 2K words of erasable memory (RAM) for variable data. This extreme constraint drove innovative programming techniques that remain instructive today.

- **I/O Channels**: The computer interfaced with numerous spacecraft systems including the DSKY (Display and Keyboard), IMU (Inertial Measurement Unit), radar systems, and engine controls.

**Software Layer: Executive and Interpreter**

The AGC software architecture was remarkably sophisticated:

- **Executive**: Functioning as an operating system, the Executive managed task scheduling, interrupt handling, and resource allocation. It implemented a priority-based scheduling system that ensured critical tasks always received CPU time.

- **Interpreter**: The AGC Interpreter was essentially a virtual machine that executed a higher-level instruction set. This allowed programmers to write more complex mathematical operations without directly managing the limited hardware resources.

**Program Modules**

The software was divided into two main program packages:

- **Comanche 055**: The Command Module software, responsible for orbital navigation, reentry calculations, and crew interface during the journey to and from the Moon.

- **Luminary 099**: The Lunar Module software, handling descent guidance, landing radar processing, ascent guidance, and rendezvous navigation.

**Interface Modules**

The AGC communicated with spacecraft systems through specialized interface routines:

- **DSKY Interface**: The Display and Keyboard interface provided the primary crew interaction point, displaying navigation data and accepting program inputs.

- **IMU Interface**: The Inertial Measurement Unit interface processed gyroscopic and accelerometer data to maintain spacecraft orientation knowledge.

- **Radar Interface**: Landing and rendezvous radar data processing for navigation during critical mission phases.

- **Engine Control**: Thrust vector control and engine timing for orbital maneuvers and landing.

## Command Module Software (Comanche 055)

![Command Module Software](/assets/img/diagrams/apollo-11/apollo-command-module.svg)

### Understanding the Command Module Software

The Command Module software (Comanche 055) managed the spacecraft during Earth orbit, trans-lunar coast, lunar orbit, and Earth return phases. This diagram shows the major software components and their relationships.

**Core Systems**

The Executive and Interpreter formed the foundation of the AGC software stack:

- **Executive (Task Scheduler)**: The Executive managed multiple concurrent tasks through a priority-based scheduling system. It handled task queuing, interrupt processing, and resource allocation. The "WAITLIST" system allowed tasks to be scheduled for future execution, critical for time-sensitive navigation calculations.

- **Interpreter (VM Engine)**: The Interpreter executed a higher-level instruction set that included complex mathematical operations. This virtual machine approach allowed more efficient use of limited memory while maintaining acceptable performance.

- **Restart (Recovery)**: The restart system provided fault tolerance by enabling the computer to recover from transient errors. If the AGC experienced a hardware fault, the restart system would restore the computer to a known state and resume critical operations.

**Navigation Programs**

The Command Module contained numerous navigation programs (P-numbers):

- **P11 (Entry Program)**: Controlled the spacecraft during atmospheric reentry, calculating the optimal trajectory for splashdown in the Pacific Ocean.

- **P20-P25 (Rendezvous)**: A suite of programs for orbital navigation and rendezvous operations, essential for docking with the Lunar Module.

- **P30-P37 (Orbital Maneuvers)**: Calculated and executed orbital transfer burns, including trans-lunar injection and trans-Earth injection.

- **P51-P53 (Alignment)**: Star sighting programs for aligning the Inertial Measurement Unit, ensuring accurate navigation reference.

**Guidance Systems**

The guidance subsystems provided mathematical foundations for navigation:

- **Conic Subroutines**: Implemented orbital mechanics calculations assuming Keplerian (conic) trajectories. These routines calculated position and velocity based on orbital elements.

- **Orbital Integration**: Numerical integration routines that propagated spacecraft state vectors over time, accounting for gravitational perturbations.

- **KALCMANU (Steering)**: Kalman filter-based steering algorithms that combined navigation measurements with predicted states for optimal guidance.

**Utility Modules**

Supporting routines provided essential services:

- **Single Precision Math**: Arithmetic routines optimized for the AGC's limited word size, handling overflow and precision issues.

- **Inter-Bank Communication**: Memory management routines that allowed code and data to span multiple memory banks, overcoming the 36K word limit.

- **Alarm and Abort**: Error handling routines that detected and responded to anomalous conditions, ensuring mission safety.

## Lunar Module Software (Luminary 099)

![Lunar Module Software](/assets/img/diagrams/apollo-11/apollo-lunar-module.svg)

### Understanding the Lunar Module Software

The Lunar Module software (Luminary 099) was specifically designed for the unique challenges of lunar landing and ascent. This diagram illustrates the specialized programs and guidance systems.

**Core Systems**

Like the Command Module, the Lunar Module software was built on the Executive and Interpreter foundation, but with additional restart tables optimized for the landing mission profile.

**Landing Programs**

The landing phase was divided into three distinct program phases:

- **P63 (Braking Phase)**: The initial descent phase that used the descent engine to reduce orbital velocity. This phase consumed the majority of the descent fuel budget.

- **P64 (Approach Phase)**: The intermediate phase where the spacecraft pitched over to allow the commander to view the landing site. This phase included the famous "program alarms" that occurred during Apollo 11's landing.

- **P66 (Final Descent)**: The final landing phase with precise control of descent rate and horizontal velocity. The commander could take manual control while the computer continued to handle guidance calculations.

**Ascent Programs**

After surface operations, the ascent programs guided the return to orbit:

- **P12 (Ascent Program)**: Controlled the ascent stage launch, calculating the optimal trajectory to reach the designated orbit for rendezvous.

- **P70-P71 (Abort Modes)**: Emergency programs that could be invoked if the landing needed to be aborted. These programs calculated safe return trajectories from any point in the landing sequence.

**Guidance Systems**

The Lunar Module guidance systems were specialized for landing operations:

- **Lunar Landing Guidance Equations**: The mathematical heart of the landing system, calculating thrust vectors and timing to achieve a soft landing at the designated site.

- **Ascent Guidance**: Calculated the ascent trajectory to reach the proper orbit for rendezvous with the Command Module.

- **Radar Routines**: Processed data from the landing radar, which provided altitude and velocity measurements during descent.

- **Throttle Control**: Managed the descent engine throttle settings, implementing the first throttleable rocket engine used in spaceflight.

## Mission Phases

![Mission Phases](/assets/img/diagrams/apollo-11/apollo-mission-phases.svg)

### Understanding the Mission Phases

This diagram illustrates the complete Apollo 11 mission profile, showing how the AGC software guided the spacecraft through each critical phase.

**Launch and Earth Orbit (P11)**

The mission began with launch and insertion into Earth orbit. The Command Module's P11 program monitored the ascent trajectory and prepared for trans-lunar injection.

**Trans-Lunar Injection**

The TLI burn accelerated the spacecraft from Earth orbit to a trajectory that would intercept the Moon. This critical maneuver required precise timing and delta-v calculations.

**Coast to Moon**

During the coast phase, the AGC performed navigation updates based on star sightings and ground tracking data. The crew monitored systems and prepared for lunar orbit insertion.

**Lunar Orbit Insertion**

The LOI burn placed the spacecraft into lunar orbit. This maneuver required precise timing to achieve the desired orbit altitude and inclination.

**Descent (P63-P66)**

The landing sequence represented the most critical phase of the mission. The Lunar Module's descent guidance software executed a complex series of calculations to achieve a soft landing at the designated site.

**Surface Operations**

While on the lunar surface, the AGC maintained navigation state and prepared for ascent. The crew performed extravehicular activities while the computer monitored systems.

**Ascent (P12)**

The ascent stage launched from the Moon's surface, guided by the P12 program to reach the proper orbit for rendezvous.

**Rendezvous (P20-P25)**

The Command Module and Lunar Module executed a carefully choreographed rendezvous sequence, with both computers sharing navigation data to achieve docking.

**Trans-Earth Injection**

The TEI burn accelerated the spacecraft from lunar orbit onto a trajectory returning to Earth.

**Entry (P61-P67)**

The Command Module's entry programs guided the spacecraft through atmospheric reentry, calculating the optimal trajectory for splashdown in the Pacific Ocean.

## Software Heritage and Preservation

![Software Heritage](/assets/img/diagrams/apollo-11/apollo-software-heritage.svg)

### Understanding the Preservation Process

This diagram shows how the original AGC source code was preserved and made available to the public.

**Original Sources**

The preservation effort began with the original hardcopy documents held at the MIT Museum. These paper printouts contained the actual assembly listings created by the MIT Instrumentation Laboratory team.

**Digitization Process**

Paul Fjeld performed the painstaking digitization work, photographing and transcribing thousands of pages of assembly code. Deborah Douglas of the MIT Museum arranged access to the historical documents.

**Transcription**

The Virtual AGC project, led by Ron Burkey, created tools to assemble and simulate the AGC code. The yaYUL assembler translates the original AGC assembly language into a format that can be executed by the Virtual AGC emulator.

**GitHub Repository**

The chrislgarry/Apollo-11 repository on GitHub serves as the primary public archive for the source code. The repository accepts pull requests for corrections and improvements, allowing the community to help preserve this historic software.

**Software Heritage Archive**

The Software Heritage project has archived the code, ensuring its long-term preservation as part of humanity's software legacy.

**Virtual AGC Emulator**

The Virtual AGC project provides a complete emulation environment, allowing anyone to run the original Apollo software on modern computers.

## Key Files and Modules

### Command Module (Comanche055)

| File | Description |
|------|-------------|
| MAIN.agc | Main program entry point |
| EXECUTIVE.agc | Task scheduler and interrupt handler |
| INTERPRETER.agc | Virtual machine for high-level operations |
| P11.agc | Entry program for atmospheric reentry |
| P20-P25.agc | Rendezvous and navigation programs |
| P30-P37.agc | Orbital maneuver programs |
| P51-P53.agc | IMU alignment programs |
| CONIC_SUBROUTINES.agc | Orbital mechanics calculations |
| ORBITAL_INTEGRATION.agc | Trajectory propagation |

### Lunar Module (Luminary099)

| File | Description |
|------|-------------|
| MAIN.agc | Main program entry point |
| THE_LUNAR_LANDING.agc | Landing guidance equations |
| ASCENT_GUIDANCE.agc | Ascent trajectory calculations |
| P12.agc | Ascent program |
| P63-P66.agc | Landing phase programs |
| P70-P71.agc | Abort mode programs |
| RADAR_LEADIN_ROUTINES.agc | Landing radar processing |
| THROTTLE_CONTROL_ROUTINES.agc | Engine throttle management |

## Technical Highlights

### Memory Constraints

The AGC's 2K words of erasable memory (approximately 4KB) forced programmers to be extremely efficient:

- **Bank Switching**: Code and data were organized into banks that could be swapped in and out of addressable memory.

- **Interpretive Mode**: The Interpreter allowed more compact representation of complex operations, trading execution time for memory savings.

- **Restart Protection**: Critical data was stored in protected memory locations to survive computer restarts.

### Real-Time Operations

The AGC was one of the first computers designed for real-time operation:

- **Priority Scheduling**: Tasks were assigned priorities, with higher-priority tasks preempting lower-priority ones.

- **Interrupt Handling**: Hardware interrupts from timers and I/O devices triggered immediate response routines.

- **Time-Sharing**: The Executive allowed multiple tasks to share CPU time while maintaining responsiveness.

### Error Handling

The famous "1202" and "1201" program alarms during Apollo 11's landing demonstrated the robustness of the AGC software:

- **Executive Overflow**: The computer was receiving more radar data than it could process in the available time.

- **Restart Recovery**: The Executive detected the overflow condition and restarted, preserving critical navigation data.

- **Mission Continuation**: Because the restart system preserved essential state, the landing continued successfully.

## Historical Significance

### Margaret Hamilton and Team

Margaret Hamilton led the software engineering team at MIT's Instrumentation Laboratory. Her work on the Apollo program pioneered many concepts now standard in software engineering:

- **Software Engineering**: Hamilton is credited with coining the term "software engineering" to describe the rigorous development process.

- **Error Handling**: The robust error handling and recovery systems she championed saved the Apollo 11 landing.

- **Human-Rated Software**: The concept of software reliable enough for human life was revolutionary at the time.

### Contract and Approvals

The source code includes the original contract and approval signatures:

- **Contract NAS 9-4065**: The NASA contract that funded the AGC development.

- **Margaret H. Hamilton**: Colossus Programming Leader, Apollo Guidance and Navigation.

- **Daniel J. Lickly**: Director, Mission Program Development.

- **Fred H. Martin**: Colossus Project Manager.

- **David G. Hoag**: Director, Apollo Guidance and Navigation Program.

## Installation and Usage

### Viewing the Source

The source code can be viewed directly on GitHub:

```bash
# Clone the repository
git clone https://github.com/chrislgarry/Apollo-11.git
cd Apollo-11

# View Command Module source
ls Comanche055/

# View Lunar Module source
ls Luminary099/
```

### Running with Virtual AGC

To actually run the AGC software:

1. Download Virtual AGC from [virtualagc.org](https://www.ibiblio.org/apollo/)
2. Build the yaYUL assembler and AGC emulator
3. Assemble the source files
4. Run the emulation

```bash
# Assemble Luminary
yaYUL Luminary099/MAIN.agc

# Run the emulator
yaAGC --cfg=LM.ini
```

## Conclusion

The Apollo 11 AGC source code represents a remarkable achievement in software engineering. Written under extreme constraints of memory and processing power, it successfully guided humanity's first Moon landing. The code demonstrates principles of real-time systems, fault tolerance, and efficient resource utilization that remain relevant today.

The preservation of this code on GitHub and in the Software Heritage archive ensures that future generations can study and learn from this pioneering work. Whether you're interested in space history, software engineering, or embedded systems, the Apollo 11 source code offers invaluable insights into one of humanity's greatest technological achievements.

## Related Posts

- [Pascal Editor: 3D Architectural Building Tool](/Pascal-Editor-3D-Architectural-Building-Tool/)
- [Software Heritage: Preserving Open Source History](/)
- [Real-Time Systems: Lessons from Apollo](/)

## Links

- [GitHub Repository](https://github.com/chrislgarry/Apollo-11)
- [Virtual AGC Project](http://www.ibiblio.org/apollo/)
- [MIT Museum](http://web.mit.edu/museum/)
- [NASA Apollo 11 Mission](https://www.nasa.gov/mission_pages/apollo/missions/apollo11.html)
- [Software Heritage Archive](https://archive.softwareheritage.org/browse/origin/https://github.com/chrislgarry/Apollo-11/)