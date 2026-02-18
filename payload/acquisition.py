"""Debris Tracker CubeSat
Sequence Acquisition Module
Politecnico di Milano — 2025
"""

import json
import os
from datetime import datetime

import cv2
import numpy as np

from camera import acquisisci_frame, init_camera


# ─────────────────────────────────────────
# CONFIGURAZIONE SESSIONE
# ─────────────────────────────────────────
DURATA_SECONDI = 30  # durata di ogni sessione di acquisizione
FPS_TARGET = 10  # frame al secondo
N_FRAME_TOTALI = DURATA_SECONDI * FPS_TARGET  # 300 frame per sessione
SAVE_DIR = "../data/sessions"


# ─────────────────────────────────────────
# FUNZIONI
# ─────────────────────────────────────────
def crea_sessione():
    """
    Crea una cartella per la sessione corrente.
    Ogni sessione ha un ID univoco basato sul timestamp.
    Struttura:
        data/sessions/
            20250301_214500/
                frames/
                metadata.json
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(SAVE_DIR, timestamp)
    frames_dir = os.path.join(session_dir, "frames")

    os.makedirs(frames_dir, exist_ok=True)

    metadata = {
        "session_id": timestamp,
        "data_ora_utc": datetime.utcnow().isoformat(),
        "fps_target": FPS_TARGET,
        "durata_secondi": DURATA_SECONDI,
        "n_frame_target": N_FRAME_TOTALI,
        "risoluzione": "1920x1080",
        "note": "",
    }

    with open(os.path.join(session_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Sessione creata: {session_dir}")
    return session_dir, frames_dir, metadata


def acquisisci_sequenza(cap, frames_dir, n_frame=N_FRAME_TOTALI):
    """
    Acquisisce una sequenza di n_frame frame consecutivi.
    Salva ogni frame come PNG con timestamp preciso.
    Restituisce la lista dei frame in memoria (array numpy).
    """
    frames = []
    timestamps = []

    print(f"\nAcquisizione in corso — {n_frame} frame @ {FPS_TARGET}fps")
    print("Premi Ctrl+C per interrompere\n")

    for i in range(n_frame):
        t_start = datetime.utcnow()
        frame, gray = acquisisci_frame(cap)

        # Salva il frame grezzo in scala di grigi
        ts_str = t_start.strftime("%H%M%S_%f")
        filename = os.path.join(frames_dir, f"frame_{i:04d}_{ts_str}.png")
        cv2.imwrite(filename, gray)

        # Tieni in memoria per processing immediato
        frames.append(gray)
        timestamps.append(t_start)

        # Progress bar semplice
        if i % 50 == 0:
            pct = int(i / n_frame * 100)
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            print(f"  [{bar}] {pct}%  Frame {i}/{n_frame}", end="\r")

    print(f"\n\nSequenza completata: {len(frames)} frame salvati in {frames_dir}")
    return frames, timestamps


def aggiorna_metadata(session_dir, metadata, n_frame_acquisiti, timestamps):
    """Aggiorna il metadata con i dati reali della sessione."""
    metadata["n_frame_acquisiti"] = n_frame_acquisiti
    if len(timestamps) > 1:
        durata = (timestamps[-1] - timestamps[0]).total_seconds()
        metadata["fps_reale"] = round(n_frame_acquisiti / durata, 2) if durata > 0 else 0
        metadata["durata_reale_sec"] = round(durata, 3)
    else:
        metadata["fps_reale"] = 0
        metadata["durata_reale_sec"] = 0

    with open(os.path.join(session_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"FPS reale: {metadata['fps_reale']}")
    print(f"Durata reale: {metadata['durata_reale_sec']}s")


def genera_preview(frames, session_dir):
    """
    Genera un'immagine di preview: la mediana di tutti i frame.
    La mediana elimina gli oggetti che si muovono (detriti, satelliti)
    e lascia solo il fondo stabile (stelle, cielo).
    Questa sarà la tua baseline per la detection.
    """
    stack = np.stack(frames, axis=0).astype(np.float32)
    mediana = np.median(stack, axis=0).astype(np.uint8)

    preview_path = os.path.join(session_dir, "preview_mediana.png")
    cv2.imwrite(preview_path, mediana)
    print(f"Preview mediana salvata: {preview_path}")
    return mediana


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    print("=== Debris Tracker CubeSat — Acquisizione Sequenza ===\n")

    cap = init_camera()
    session_dir, frames_dir, metadata = crea_sessione()

    try:
        frames, timestamps = acquisisci_sequenza(cap, frames_dir)
    except KeyboardInterrupt:
        print("\n\nAcquisizione interrotta dall'utente.")
        frames = []
        timestamps = []

    if len(frames) > 1:
        aggiorna_metadata(session_dir, metadata, len(frames), timestamps)
        _ = genera_preview(frames, session_dir)
        print(f"\nSessione completata. Dati in: {session_dir}")
    else:
        print("Nessun frame acquisito.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

