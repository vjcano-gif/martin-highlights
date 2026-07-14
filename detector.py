"""
detector.py
------------
Detección de jugadores y balón con YOLOv8 + seguimiento (ByteTrack).
Clasifica el equipo por el color de la camiseta (color configurable).

Este es el "ojo" del sistema: mira el video y devuelve, frame por frame,
dónde está cada jugador (con un ID que lo sigue) y dónde está el balón.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

# --- Clases del modelo COCO que nos interesan -------------------------------
PERSON_CLASS = 0      # persona (jugadores, árbitro, etc.)
BALL_CLASS = 32       # 'sports ball' (el balón)

# --- Rangos de color en HSV (OpenCV: H de 0 a 179) --------------------------
# Cada color es una lista de rangos (el rojo necesita dos porque "da la vuelta").
COLOR_RANGES: dict[str, list[tuple[tuple, tuple]]] = {
    "amarillo": [((20, 80, 80), (35, 255, 255))],
    "naranja":  [((10, 120, 100), (20, 255, 255))],
    "rojo":     [((0, 100, 80), (10, 255, 255)), ((170, 100, 80), (179, 255, 255))],
    "rosa":     [((140, 50, 120), (170, 255, 255))],
    "morado":   [((130, 60, 60), (160, 255, 255))],
    "azul":     [((95, 80, 60), (130, 255, 255))],
    "celeste":  [((85, 40, 120), (105, 255, 255))],
    "verde":    [((40, 60, 60), (85, 255, 255))],
    "blanco":   [((0, 0, 185), (179, 40, 255))],
    "negro":    [((0, 0, 0), (179, 255, 55))],
}


@dataclass
class Detection:
    """Una detección en un frame concreto."""
    frame_idx: int
    time_s: float
    track_id: int | None            # ID que sigue al mismo objeto entre frames
    cls: int                        # PERSON_CLASS o BALL_CLASS
    bbox: tuple[float, float, float, float]   # x1, y1, x2, y2
    conf: float
    is_target: bool = False         # True si la camiseta coincide con el color buscado


def _fraction_color(crop: np.ndarray, ranges) -> float:
    """Fracción (0-1) de píxeles del color buscado en un recorte."""
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = None
    for lo, hi in ranges:
        m = cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    return float(mask.mean()) / 255.0


def matches_color(frame: np.ndarray, bbox, ranges, torso_ratio: float = 0.5,
                  threshold: float = 0.15) -> bool:
    """
    Decide si el jugador lleva el color buscado mirando su torso
    (la mitad superior del recuadro, para evitar el short/piernas).
    """
    x1, y1, x2, y2 = (int(v) for v in bbox)
    h = y2 - y1
    ty2 = int(y1 + h * torso_ratio)
    crop = frame[max(y1, 0):max(ty2, 0), max(x1, 0):max(x2, 0)]
    return _fraction_color(crop, ranges) >= threshold


def analyze_video(video_path, model_name: str = "yolov8n.pt", conf: float = 0.3,
                  team_color: str | None = "amarillo", color_threshold: float = 0.15,
                  progress=None):
    """
    Corre YOLO + seguimiento sobre el video.

    Parámetros
    ----------
    model_name    : "yolov8n.pt" rápido; "yolov8s/m.pt" más precisos y lentos.
    team_color    : nombre de color en COLOR_RANGES para marcar el equipo de
                    Martin. Si es None, NO filtra por color y marca a todos
                    (útil si Martin usa un uniforme distinto, ej. arquero).
    color_threshold : cuánta camiseta debe ser de ese color (0-1). Bájalo si no
                    detecta el uniforme; súbelo si marca a rivales por error.
    progress      : función opcional progress(frame_idx, total) para la barra.

    Devuelve
    --------
    (detections, fps, total_frames)
    """
    model = YOLO(model_name)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    ranges = COLOR_RANGES.get(team_color) if team_color else None

    detections: list[Detection] = []

    # stream=True procesa frame a frame sin cargar todo en RAM.
    results = model.track(
        source=str(video_path),
        classes=[PERSON_CLASS, BALL_CLASS],
        conf=conf,
        persist=True,
        tracker="bytetrack.yaml",
        stream=True,
        verbose=False,
    )

    for frame_idx, r in enumerate(results):
        frame = r.orig_img
        time_s = frame_idx / fps

        if r.boxes is not None:
            for b in r.boxes:
                cls = int(b.cls[0])
                xyxy = tuple(float(v) for v in b.xyxy[0])
                conf_v = float(b.conf[0])
                tid = int(b.id[0]) if b.id is not None else None

                is_target = False
                if cls == PERSON_CLASS:
                    # Sin filtro de color -> todos son candidatos a ser Martin.
                    is_target = (ranges is None) or matches_color(
                        frame, xyxy, ranges, threshold=color_threshold
                    )

                detections.append(
                    Detection(frame_idx, time_s, tid, cls, xyxy, conf_v, is_target)
                )

        if progress:
            progress(frame_idx, total)

    return detections, fps, total


def sample_crops(video_path, detections, track_ids):
    """
    Saca un recorte de ejemplo por cada track_id pedido, para que en la
    interfaz puedas ver a cada jugador y decir cuál es Martin.

    Devuelve {track_id: imagen_recorte_BGR}.
    """
    wanted: dict[int, Detection] = {}
    for d in detections:
        if d.cls == PERSON_CLASS and d.track_id in track_ids:
            # nos quedamos con la aparición más nítida (mayor confianza)
            if d.track_id not in wanted or d.conf > wanted[d.track_id].conf:
                wanted[d.track_id] = d

    cap = cv2.VideoCapture(str(video_path))
    crops: dict[int, np.ndarray] = {}
    for tid, d in wanted.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, d.frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        x1, y1, x2, y2 = (int(v) for v in d.bbox)
        crop = frame[max(y1, 0):y2, max(x1, 0):x2]
        if crop.size:
            crops[tid] = crop.copy()
    cap.release()
    return crops
