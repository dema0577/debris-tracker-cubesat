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

## Software Modules
- `payload/camera.py` — Live view dalla camera, salvataggio singoli frame, telemetria base (min/max/mean/std dei pixel).
- `payload/acquisition.py` — Acquisizione di sequenze (sessioni) con metadata, salvataggio frame e immagine mediana di fondo.
- `algorithm/detector.py` — Algoritmo di detection (background mediano, sottrazione, sogliatura robusta, classificazione stelle/detriti, test simulato).
- `algorithm/live_detector.py` — Pipeline live camera + algoritmo con overlay HUD e logging dei rilevamenti.

## Setup & Installazione
Si consiglia un ambiente virtuale Python:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
# .venv\Scripts\activate   # Windows (PowerShell)

pip install -r requirements.txt
```

## Come Eseguire i Moduli Principali

### Test Camera Live
```bash
cd payload
python camera.py
```

### Acquisizione di una Sequenza
```bash
cd payload
python acquisition.py
```

### Test Algoritmo con Dati Simulati
```bash
cd algorithm
python detector.py
```

### Live Detection (Camera + Algoritmo)
```bash
cd algorithm
python live_detector.py
```

I risultati (frame, metadata, detections) vengono salvati nella cartella `data/` nelle rispettive sottocartelle (`raw`, `sessions`, `test`, `live_sessions`). 