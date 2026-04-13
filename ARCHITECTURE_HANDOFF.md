# Buddy Architecture Handoff

## Project Goal

Buddy is a physical robot built around a Raspberry Pi.

The current development setup is split across two environments:

- Raspberry Pi:
  Handles robot-side hardware and real-world operation.
  This includes camera input, microphones, speakers, servo/motor/Arduino control, and the final always-on robot runtime.

- Windows laptop / PC:
  Used as a development and testing machine.
  It is also used as a local LLM server and a convenient place to test the full interaction flow before pushing things to the Pi.

## Most Important File Right Now

The most important behavior reference is:

- [`pi_services/buddy_windows_test.py`](./pi_services/buddy_windows_test.py)

This file is the current best example of how Buddy is supposed to behave at a high level:

- camera loop
- face detection
- face recognition
- object detection
- voice interaction
- registration of new faces
- conversation flow with the LLM

Important: this file is mainly a Windows testing harness.

It is meant to verify the desired Buddy interaction flow using:

- laptop webcam
- Windows audio environment
- local desktop dependencies
- Visual Studio build tools / Windows-compatible Python packages

It is **not** the final Raspberry Pi runtime.

## Core Architecture Intent

The intended final Buddy architecture is:

1. Raspberry Pi runs the robot in the real environment.
2. The Pi handles hardware-facing responsibilities:
   - camera
   - microphone / speaker
   - servo and motor movement
   - Arduino communication
   - wake/sleep robot behavior
3. The PC or local server side handles higher-cost model work when needed:
   - LLM response generation
   - memory-related logic
   - possibly some remote service endpoints
4. The final Pi startup file should integrate all of this cleanly into one runnable robot entrypoint.

## Current Repo Reality

There are multiple runtime-style files inside `pi_services`, including:

- `buddy_windows_test.py`
- `buddy_main.py`
- `buddy_pi.py`
- `Buddy_new.py`
- `Buddy_move.py`

These represent overlapping experiments / versions.

Right now:

- `buddy_windows_test.py` is the clearest reference for conversation behavior.
- The Pi-oriented files contain important hardware, movement, audio, and robot logic.
- The repo still needs one clean final Raspberry Pi entrypoint that combines the correct pieces.

## Recent Organization Work

A non-destructive folder organization was added under `pi_services`.

New folders:

- `core`
- `vision`
- `audio`
- `hardware`
- `memory`
- `miscellaneous`

These contain copied/organized versions of related files so the codebase is easier to understand.

Examples:

- `vision` contains face recognition, object recognition, and vision models
- `audio` contains STT/TTS/audio diagnostics and audio models
- `hardware` contains servo, motor, and camera helper code
- `memory` contains database-related face memory scripts
- `core` contains shared config/state/tracking utilities

Important: old files were **not removed**.

This means the repo currently has:

- original files still present in their old locations
- organized copies in the new folders

This was done on purpose to avoid destructive changes.

## Key Design Guidance For Next Session

When continuing this project, treat `buddy_windows_test.py` as the behavior reference, not the final deployment target.

The next important engineering task is:

- build one proper Raspberry Pi startup/runtime file

That Pi runtime should preserve the behavior pattern proven in `buddy_windows_test.py`:

- recognize known people
- ask unknown people for their name
- register face data
- use object detection context
- talk conversationally through the LLM

But it must be adapted for the Raspberry Pi environment:

- Pi-compatible camera handling
- Pi-compatible audio capture/output
- Pi hardware control
- Arduino/movement integration
- Pi-safe dependencies and performance constraints

## Practical Rule For Future Work

If there is ever a conflict between:

- "what behavior Buddy should have"
- and "which file is the final deployment file"

then use this rule:

- `buddy_windows_test.py` defines the intended interaction behavior
- the future Pi runtime should implement that same behavior correctly for Raspberry Pi

## Notes About Dependencies

Windows testing and Raspberry Pi deployment are not the same environment.

Some packages or flows that work on Windows may need different handling on Raspberry Pi, especially for:

- InsightFace / RetinaFace setup
- audio libraries
- camera integration
- hardware GPIO / serial control
- model performance constraints

So future work should focus on behavioral parity, not blindly copying Windows-specific implementation details.

## Recommended Next Step

Next session, read this file first, then inspect:

1. `pi_services/buddy_windows_test.py`
2. `pi_services/buddy_main.py`
3. `pi_services/buddy_pi.py`
4. `pi_services/Buddy_new.py`
5. `pi_services/Buddy_move.py`

Then design a single clean Raspberry Pi startup file that:

- uses the correct Pi-side modules
- matches the conversation/perception flow from `buddy_windows_test.py`
- becomes the main file used to start Buddy on the robot
