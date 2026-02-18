"""Debris Tracker CubeSat
Core Detection Algorithm
Politecnico di Milano — 2025
"""

import json
import os
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops


# ─────────────────────────────────────────
# CONFIGURAZIONE DETECTION
# ─────────────────────────────────────────
SIGMA_SOGLIA = 4.0  # soglia in unità di sigma (4σ = molto selettivo)
ECCENTRICITA_MIN = 0.90  # sotto questo valore = stella (oggetto compatto)
AREA_MIN_STREAK = 15  # pixel minimi per essere un detrito (filtra rumore)
AREA_MIN_STELLA = 3  # pixel minimi per essere una stella


# ─────────────────────────────────────────
# STEP 1 — COSTRUISCI IL FONDO
# ─────────────────────────────────────────
def calcola_fondo_mediano(frames):
    """
    Il fondo è la MEDIANA di tutti i frame della sequenza.

    Perché la mediana e non la media?
    Se un detrito attraversa il campo visivo, occupa ogni pixel
    solo per 2-3 frame su 300. La mediana ignora questi outlier
    e restituisce il valore "stabile" — il cielo senza oggetti mobili.
    La media invece verrebbe contaminata dai pixel del detrito.
    """
    print("Calcolo fondo mediano...", end=" ")
    stack = np.stack(frames, axis=0).astype(np.float32)
    fondo = np.median(stack, axis=0)
    print(f"fatto. Range fondo: [{fondo.min():.0f}, {fondo.max():.0f}]")
    return fondo


# ─────────────────────────────────────────
# STEP 2 — SOTTRAI IL FONDO DA OGNI FRAME
# ─────────────────────────────────────────
def sottrai_fondo(frame, fondo):
    """
    Sottrae il fondo dal frame corrente.
    Il risultato è un'immagine dove:
    - I pixel vicini a 0 = sfondo stabile (stelle, cielo)
    - I pixel molto positivi = qualcosa che si è mosso

    Usiamo float32 per evitare overflow con valori negativi.
    """
    frame_f = frame.astype(np.float32)
    residuo = frame_f - fondo

    # Normalizza: porta i valori negativi a zero
    # (non ci interessano zone diventate più scure)
    residuo = np.clip(residuo, 0, None)
    return residuo


# ─────────────────────────────────────────
# STEP 3 — SOGLIATURA STATISTICA
# ─────────────────────────────────────────
def sogliatura(residuo, sigma_threshold=SIGMA_SOGLIA):
    """
    Isola i pixel statisticamente significativi.

    Usiamo MAD (Median Absolute Deviation) invece della deviazione
    standard classica perché è robusta agli outlier.
    Se usassimo la std, le stelle brillanti la gonfierebbero
    e perderemmo segnali deboli.

    La soglia a 4σ significa: accettiamo solo pixel la cui
    probabilità di essere rumore casuale è < 0.003%.
    """
    mediana = np.median(residuo)
    mad = np.median(np.abs(residuo - mediana))
    sigma = mad * 1.4826  # conversione MAD → sigma gaussiano

    soglia = mediana + sigma_threshold * sigma
    maschera = (residuo > soglia).astype(np.uint8)

    return maschera, soglia, sigma


# ─────────────────────────────────────────
# STEP 4 — CLASSIFICA GLI OGGETTI
# ─────────────────────────────────────────
def classifica_oggetti(maschera):
    """
    Analizza la forma geometrica di ogni gruppo di pixel connessi.

    Metrica chiave: ECCENTRICITÀ
    - Eccentricità = 0 → cerchio perfetto (stella puntiforme)
    - Eccentricità = 1 → linea perfetta (streak di detrito)

    Un detrito in moto produce una traccia allungata → alta eccentricità.
    Una stella è un punto → bassa eccentricità.

    Restituisce due liste: detriti e stelle trovati nel frame.
    """
    # Etichetta componenti connessi (ogni gruppo di pixel = un oggetto)
    etichette = label(maschera, connectivity=2)
    proprieta = regionprops(etichette)

    detriti = []
    stelle = []

    for obj in proprieta:
        area = obj.area
        eccentricita = obj.eccentricity

        if area < AREA_MIN_STELLA:
            continue  # troppo piccolo, è rumore

        if eccentricita >= ECCENTRICITA_MIN and area >= AREA_MIN_STREAK:
            detriti.append(
                {
                    "centroid": obj.centroid,
                    "bbox": obj.bbox,
                    "area": area,
                    "eccentricita": round(eccentricita, 4),
                    "orientazione": round(np.degrees(obj.orientation), 2),
                    "lunghezza": round(obj.major_axis_length, 2),
                    "larghezza": round(obj.minor_axis_length, 2),
                }
            )
        else:
            stelle.append(
                {
                    "centroid": obj.centroid,
                    "area": area,
                    "eccentricita": round(eccentricita, 4),
                }
            )

    return detriti, stelle


# ─────────────────────────────────────────
# STEP 5 — PROCESSA INTERA SEQUENZA
# ─────────────────────────────────────────
def processa_sequenza(frames, timestamps=None):
    """
    Applica la pipeline completa all'intera sequenza di frame.
    Restituisce tutti i rilevamenti con i loro timestamp.
    """
    print(f"\n=== Detection Pipeline — {len(frames)} frame ===\n")

    fondo = calcola_fondo_mediano(frames)
    rilevamenti = []

    for i, frame in enumerate(frames):
        residuo = sottrai_fondo(frame, fondo)
        maschera, soglia, sigma = sogliatura(residuo)
        detriti, stelle = classifica_oggetti(maschera)

        if detriti:
            ts = timestamps[i].isoformat() if timestamps else f"frame_{i:04d}"
            for d in detriti:
                rilevamento = {
                    "frame_index": i,
                    "timestamp": ts,
                    "centroid_y": round(d["centroid"][0], 2),
                    "centroid_x": round(d["centroid"][1], 2),
                    "area_px": d["area"],
                    "eccentricita": d["eccentricita"],
                    "orientazione": d["orientazione"],
                    "lunghezza_px": d["lunghezza"],
                    "sigma_soglia": round(sigma, 3),
                }
                rilevamenti.append(rilevamento)
                print(
                    f"  ⚠ DETRITO frame {i:04d} | "
                    f"pos=({d['centroid'][1]:.0f}, {d['centroid'][0]:.0f}) | "
                    f"e={d['eccentricita']:.3f} | "
                    f"L={d['lunghezza']:.0f}px | "
                    f"θ={d['orientazione']:.1f}°"
                )

    print(f"\nTotale rilevamenti: {len(rilevamenti)}")
    return rilevamenti, fondo


# ─────────────────────────────────────────
# VISUALIZZAZIONE
# ─────────────────────────────────────────
def visualizza_detection(frame, detriti, stelle, n_frame=0):
    """Visualizza il frame con le detection marcate."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.imshow(frame, cmap="gray", vmin=0, vmax=255)

    for s in stelle:
        y, x = s["centroid"]
        ax.plot(x, y, "b+", markersize=6, markeredgewidth=1)

    for d in detriti:
        y0, x0, y1, x1 = d["bbox"]
        rect = mpatches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            x0,
            y0 - 8,
            f"e={d['eccentricita']:.2f} L={d['lunghezza']:.0f}px",
            color="red",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_title(
        f"Frame {n_frame} — "
        f"Detriti: {len(detriti)} (rosso) | Stelle: {len(stelle)} (blu)",
        fontsize=11,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def salva_risultati(rilevamenti, session_dir):
    """Salva i rilevamenti in un file JSON."""
    path = os.path.join(session_dir, "detections.json")
    with open(path, "w") as f:
        json.dump(rilevamenti, f, indent=4, default=str)
    print(f"Risultati salvati: {path}")
    return path


# ─────────────────────────────────────────
# MAIN — Test con immagine simulata
# ─────────────────────────────────────────
def main():
    """
    Test della pipeline con dati simulati.
    Quando arriva l'hardware reale sostituiamo
    questi frame simulati con quelli della camera.
    """
    print("=== Debris Tracker — Detection Test (simulato) ===\n")

    np.random.seed(42)
    N = 50  # frame simulati

    frames = []
    for i in range(N):
        # Fondo: rumore gaussiano + gradiente
        img = np.random.normal(100, 8, (480, 640)).astype(np.float32)
        for r in range(480):
            img[r, :] += r * 0.03

        # Stelle fisse (stesse posizioni in ogni frame)
        for (sy, sx) in [(100, 200), (300, 450), (240, 80), (400, 600)]:
            img[sy, sx] += np.random.uniform(60, 120)

        # Detrito: streak orizzontale che si sposta di 8px per frame
        x_start = 50 + i * 8
        x_end = 120 + i * 8
        if x_end < 640:
            img[220, x_start:x_end] += 90

        frames.append(np.clip(img, 0, 255).astype(np.uint8))

    # Esegui pipeline
    rilevamenti, fondo = processa_sequenza(frames)

    # Visualizza un frame con detection
    if rilevamenti:
        idx = rilevamenti[0]["frame_index"]
        frame = frames[idx]
        residuo = sottrai_fondo(frame, fondo)
        maschera, _, _ = sogliatura(residuo)
        detriti, stelle = classifica_oggetti(maschera)
        visualizza_detection(frame, detriti, stelle, n_frame=idx)

    # Salva risultati
    os.makedirs("../data/test", exist_ok=True)
    salva_risultati(rilevamenti, "../data/test")


if __name__ == "__main__":
    main()

