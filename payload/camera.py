"""Debris Tracker CubeSat
Payload Camera Controller
Politecnico di Milano — 2025
"""

import os
from datetime import datetime

import cv2


# ─────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────
CAMERA_INDEX = 0  # 0 = prima camera disponibile
FRAME_WIDTH = 1920  # risoluzione orizzontale
FRAME_HEIGHT = 1080  # risoluzione verticale
FPS_TARGET = 10  # frame al secondo
SAVE_DIR = "../data/raw"  # dove salvi le immagini


# ─────────────────────────────────────────
# FUNZIONI
# ─────────────────────────────────────────
def init_camera():
    """Inizializza e configura la camera."""
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError("Errore: camera non trovata. Controlla la connessione.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)

    # Modalità manuale — fondamentale per astronomia
    # L'auto-exposure rovinerebbe le immagini del cielo notturno
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = manuale su molte camere
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)  # esposizione manuale (valore base)

    print(f"Camera inizializzata: {FRAME_WIDTH}x{FRAME_HEIGHT} @ {FPS_TARGET}fps")
    return cap


def acquisisci_frame(cap):
    """Cattura un singolo frame e lo restituisce in scala di grigi."""
    ret, frame = cap.read()

    if not ret:
        raise RuntimeError("Errore: impossibile leggere il frame dalla camera.")

    # Converti in scala di grigi — più efficiente per la detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame, gray


def salva_frame(frame, cartella=SAVE_DIR):
    """Salva il frame con timestamp nel nome del file."""
    os.makedirs(cartella, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{cartella}/frame_{timestamp}.fits"

    # Salva come PNG per ora (FITS quando integriamo Astropy)
    filename_png = filename.replace(".fits", ".png")
    cv2.imwrite(filename_png, frame)
    return filename_png


def mostra_info_frame(gray):
    """Stampa statistiche base del frame — utile per debug."""
    print(
        f"  Min pixel: {int(gray.min()):4d}  "
        f"Max pixel: {int(gray.max()):4d}  "
        f"Media: {float(gray.mean()):.1f}  "
        f"Std dev: {float(gray.std()):.2f}"
    )


# ─────────────────────────────────────────
# MAIN — Acquisizione Live
# ─────────────────────────────────────────
def main():
    print("=== Debris Tracker CubeSat — Camera Controller ===")
    print("Premi 'q' per uscire, 's' per salvare un frame\n")

    cap = init_camera()
    n_frame = 0

    try:
        while True:
            frame, gray = acquisisci_frame(cap)
            n_frame += 1

            # Mostra statistiche ogni 30 frame
            if n_frame % 30 == 0:
                print(f"Frame #{n_frame}:", end="")
                mostra_info_frame(gray)

            # Aggiungi overlay informativo sull'immagine
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            cv2.putText(
                frame,
                f"Debris Tracker CubeSat — {timestamp}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Frame: {n_frame}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Payload Camera — Live View", frame)

            tasto = cv2.waitKey(1) & 0xFF

            if tasto == ord("q"):
                print("\nChiusura camera.")
                break
            elif tasto == ord("s"):
                path = salva_frame(frame)
                print(f"Frame salvato: {path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

