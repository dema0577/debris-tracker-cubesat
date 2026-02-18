"""Debris Tracker CubeSat
Live Detection — Camera + Algorithm Pipeline
Politecnico di Milano — 2025
"""

import os
import sys
from collections import deque
from datetime import datetime

import cv2

# Importa i moduli che hai già scritto
sys.path.append("../payload")
from camera import acquisisci_frame, init_camera
from detector import (
    calcola_fondo_mediano,
    classifica_oggetti,
    salva_risultati,
    sogliatura,
    sottrai_fondo,
)


# ─────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────
N_FRAME_CALIBRAZIONE = 50  # frame usati per costruire il fondo iniziale
BUFFER_SIZE = 100  # frame tenuti in memoria per aggiornare il fondo
SAVE_DIR = "../data/live_sessions"


# ─────────────────────────────────────────
# CALIBRAZIONE — Costruisce il fondo iniziale
# ─────────────────────────────────────────
def calibra(cap):
    """
    Acquisisce N frame per costruire il fondo mediano iniziale.
    Durante la calibrazione il sistema NON fa detection —
    sta solo imparando come appare il cielo senza oggetti mobili.
    """
    print(f"Calibrazione in corso — acquisisco {N_FRAME_CALIBRAZIONE} frame...")
    print("Tieni la camera ferma e puntata verso il cielo.\n")

    frames_cal = []
    for i in range(N_FRAME_CALIBRAZIONE):
        _, gray = acquisisci_frame(cap)
        frames_cal.append(gray)

        # Progress
        pct = int((i + 1) / N_FRAME_CALIBRAZIONE * 100)
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"  [{bar}] {pct}%", end="\r")

    fondo = calcola_fondo_mediano(frames_cal)
    print("\nCalibrazione completata.\n")
    return fondo, deque(frames_cal, maxlen=BUFFER_SIZE)


# ─────────────────────────────────────────
# OVERLAY — Disegna i risultati sul frame
# ─────────────────────────────────────────
def disegna_overlay(frame_bgr, detriti, stelle, n_frame, n_rilevamenti_totali):
    """
    Disegna in tempo reale sul frame live:
    - Riquadro rosso sui detriti rilevati
    - Punto blu sulle stelle
    - HUD con statistiche
    """
    overlay = frame_bgr.copy()

    # Stelle — punto blu
    for s in stelle:
        y, x = int(s["centroid"][0]), int(s["centroid"][1])
        cv2.circle(overlay, (x, y), 3, (255, 100, 0), -1)

    # Detriti — riquadro rosso + info
    for d in detriti:
        y0, x0, y1, x1 = [int(v) for v in d["bbox"]]
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.putText(
            overlay,
            f"DEBRIS e={d['eccentricita']:.2f}",
            (x0, y0 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
        )

    # HUD — angolo in alto a sinistra
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    hud = [
        "Debris Tracker CubeSat — PoliMi 2025",
        f"Frame: {n_frame:05d}",
        f"Timestamp: {ts}",
        f"Stelle rilevate: {len(stelle)}",
        f"Detriti questo frame: {len(detriti)}",
        f"Totale rilevamenti sessione: {n_rilevamenti_totali}",
    ]

    for i, testo in enumerate(hud):
        cv2.putText(
            overlay,
            testo,
            (10, 25 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
        )

    # Indicatore stato — verde = ok, rosso = detrito rilevato
    colore_stato = (0, 0, 255) if detriti else (0, 255, 0)
    stato_testo = "⚠ DEBRIS DETECTED" if detriti else "● SCANNING"
    cv2.putText(
        overlay,
        stato_testo,
        (10, frame_bgr.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        colore_stato,
        2,
    )

    return overlay


# ─────────────────────────────────────────
# MAIN — Loop di detection live
# ─────────────────────────────────────────
def main():
    print("=== Debris Tracker CubeSat — Live Detection ===\n")

    # Setup
    cap = init_camera()
    session_ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(SAVE_DIR, session_ts)
    os.makedirs(session_dir, exist_ok=True)

    # Calibrazione
    fondo, buffer = calibra(cap)

    # Stato sessione
    n_frame = 0
    tutti_rilevamenti = []
    aggiorna_fondo_ogni = 30  # ricalcola fondo ogni N frame

    print("Sistema attivo. Comandi:")
    print("  'q' → esci")
    print("  's' → salva frame corrente")
    print("  'r' → ricalibrа fondo\n")

    while True:
        frame_bgr, gray = acquisisci_frame(cap)
        n_frame += 1

        # ── Detection ──────────────────────────────
        residuo = sottrai_fondo(gray, fondo)
        maschera, _, _ = sogliatura(residuo)
        detriti, stelle = classifica_oggetti(maschera)

        # ── Se trova un detrito — logga e salva ────
        if detriti:
            ts = datetime.utcnow()
            for d in detriti:
                rilevamento = {
                    "frame": n_frame,
                    "timestamp": ts.isoformat(),
                    "centroid_x": round(d["centroid"][1], 2),
                    "centroid_y": round(d["centroid"][0], 2),
                    "eccentricita": d["eccentricita"],
                    "orientazione": d["orientazione"],
                    "lunghezza_px": d["lunghezza"],
                    "area_px": d["area"],
                }
                tutti_rilevamenti.append(rilevamento)
                print(
                    f"  ⚠  Frame {n_frame:05d} | "
                    f"DETRITO pos=({d['centroid'][1]:.0f},"
                    f"{d['centroid'][0]:.0f}) | "
                    f"e={d['eccentricita']:.3f} | "
                    f"L={d['lunghezza']:.0f}px"
                )

            # Salva frame con detection
            fname = os.path.join(session_dir, f"detection_{n_frame:05d}.png")
            cv2.imwrite(fname, frame_bgr)

        # ── Aggiorna fondo periodicamente ──────────
        buffer.append(gray)
        if n_frame % aggiorna_fondo_ogni == 0:
            fondo = calcola_fondo_mediano(list(buffer))

        # ── Visualizzazione ────────────────────────
        overlay = disegna_overlay(
            frame_bgr, detriti, stelle, n_frame, len(tutti_rilevamenti)
        )
        cv2.imshow("Debris Tracker — Live", overlay)

        # ── Input utente ───────────────────────────
        tasto = cv2.waitKey(1) & 0xFF

        if tasto == ord("q"):
            break
        elif tasto == ord("s"):
            fname = os.path.join(session_dir, f"manual_{n_frame:05d}.png")
            cv2.imwrite(fname, frame_bgr)
            print(f"  Frame salvato manualmente: {fname}")
        elif tasto == ord("r"):
            print("\nRicalibrazione fondo...")
            fondo, buffer = calibra(cap)

    # ── Fine sessione ──────────────────────────────
    cap.release()
    cv2.destroyAllWindows()

    if tutti_rilevamenti:
        salva_risultati(tutti_rilevamenti, session_dir)
        print(f"\nSessione terminata — {len(tutti_rilevamenti)} rilevamenti totali")
    else:
        print("\nSessione terminata — nessun rilevamento.")

    print(f"Dati salvati in: {session_dir}")


if __name__ == "__main__":
    main()

