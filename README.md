# Debris Tracker CubeSat
**3U CubeSat Engineering Model for Space Debris Optical Tracking**

Developed by [Matteo De Masi] — Politecnico di Milano, 2025

## Mission
Detect and characterize space debris in LEO using an optical payload 
mounted on a 3U CubeSat Engineering Model. The system uses a miniaturized 
telescope and a custom detection algorithm based on streak analysis.

## Project Status
🔴 Phase 1 — Ground Prototype Development (in progress)

## Repository Structure
- `payload/` — Camera control and image acquisition
- `algorithm/` — Debris detection pipeline
- `docs/` — Mission Design Report and technical documents
- `data/` — Acquired images and results
- `structure/` — 3D CAD files

## Hardware
- Raspberry Pi 4 (4GB) — On-Board Computer
- Raspberry Pi HQ Camera (Sony IMX477) — Optical Detector
- 16mm f/1.4 C-mount lens — Optical system

## Tech Stack
Python 3.11 · OpenCV · NumPy · Astropy · scikit-image