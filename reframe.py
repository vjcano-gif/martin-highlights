"""
reframe.py
----------
Recorte VERTICAL dinámico: la ventana se mueve para mantener a Martin
centrado durante toda la jugada (ideal para Stories / TikTok / Reels).

Usa las posiciones que YOLO ya detectó de Martin, las suaviza (para que la
cámara no "tiemble") y recorta frame por frame con OpenCV.
"""
from __future__ import annotations

import os
import subprocess
import tempfile

import cv2
import numpy as np

from clipper import concat_clips
from detector import PERSON_CLASS


def martin_centers(detections, martin_ids) -> dict[int, tuple[float, float]]:
    """Centro (x, y) de Martin en cada frame donde aparece."""
    ids = set(martin_ids)
    out: dict[int, tuple[float, float]] = {}
    for d in detections:
        if d.cls == PERSON_CLASS and d.track_id in ids:
            x1, y1, x2, y2 = d.bbox
            out[d.frame_idx] = ((x1 + x2) / 2, (y1 + y2) / 2)
    return out


def _series(centers, f0, f1):
    """Arrays cx, cy para los frames f0..f1, interpolando los huecos."""
    known = sorted(k for k in centers if f0 <= k <= f1)
    if not known:
        return None
    frames = list(range(f0, f1 + 1))
    kx = [centers[k][0] for k in known]
    ky = [centers[k][1] for k in known]
    return np.interp(frames, known, kx), np.interp(frames, known, ky)


def _smooth(a: np.ndarray, win: int) -> np.ndarray:
    """Media móvil para que el encuadre no dé saltos bruscos."""
    if win <= 1 or len(a) < win:
        return a
    win = min(win, len(a))
    left = win // 2
    right = win - 1 - left
    padded = np.pad(a, (left, right), mode="edge")
    return np.convolve(padded, np.ones(win) / win, mode="valid")


def follow_crop_clip(src, start_s, end_s, centers, fps, size, out_path,
                     smooth_win: int = 15):
    """
    Genera un clip recortado que sigue a Martin.

    size : (ancho, alto) de salida. La ventana mantiene esa proporción y se
           mueve horizontal/verticalmente para seguir a Martin.
    Devuelve out_path, o None si Martin no aparece en ese tramo.
    """
    tw, th = size
    ar = tw / th

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened() or not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError("No se pudo abrir el video o su FPS no es válido.")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if W <= 0 or H <= 0 or total_frames <= 0:
        cap.release()
        raise ValueError("El video no informa dimensiones o fotogramas válidos.")
    # Los clips usan el intervalo semiabierto [inicio, fin), como ffmpeg -t.
    f0 = max(int(np.floor(start_s * fps)), 0)
    f1 = min(int(np.ceil(end_s * fps)), total_frames)
    if f1 <= f0:
        cap.release()
        raise ValueError("El rango no contiene fotogramas dentro del video.")

    # Ventana de recorte: la más grande posible con la proporción de salida.
    cw, ch = round(H * ar), H
    if cw > W:
        cw, ch = W, round(W / ar)
    cw -= cw % 2
    ch -= ch % 2

    ser = _series(centers, f0, f1 - 1)
    if ser is None:
        cap.release()
        return None
    cx, cy = _smooth(ser[0], smooth_win), _smooth(ser[1], smooth_win)

    fd, tmp = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (tw, th))
    if not writer.isOpened():
        cap.release()
        os.remove(tmp)
        raise RuntimeError("No se pudo crear el video temporal.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    expected = f1 - f0
    written = 0
    for i in range(expected):
        ok, frame = cap.read()
        if not ok:
            break
        x = int(np.clip(cx[i] - cw / 2, 0, max(W - cw, 0)))
        y = int(np.clip(cy[i] - ch / 2, 0, max(H - ch, 0)))
        crop = frame[y:y + ch, x:x + cw]
        writer.write(cv2.resize(crop, (tw, th)))
        written += 1
    writer.release()
    cap.release()
    if written != expected:
        os.remove(tmp)
        raise ValueError(f"Se esperaban {expected} fotogramas y se leyeron {written}.")

    # Reincorporar el audio original del tramo (el crop de OpenCV no lleva sonido).
    dur = end_s - start_s
    cmd = ["ffmpeg", "-y", "-i", tmp,
           "-ss", f"{start_s:.2f}", "-t", f"{dur:.2f}", "-i", str(src),
           "-map", "0:v:0", "-map", "1:a:0?",
           "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac",
           "-shortest", out_path]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return out_path


def build_following_highlights(src, clips, centers, fps, size,
                               workdir: str = "clips",
                               out_path: str = "highlights_martin.mp4",
                               smooth_win: int = 15, progress=None) -> str:
    """Recorta cada jugada siguiendo a Martin y las une."""
    os.makedirs(workdir, exist_ok=True)
    paths = []
    for i, (s, e) in enumerate(clips):
        p = os.path.join(workdir, f"clip_{i:03d}.mp4")
        if follow_crop_clip(src, s, e, centers, fps, size, p, smooth_win):
            paths.append(p)
        if progress:
            progress(i + 1, len(clips))
    if not paths:
        raise ValueError("No se pudo recortar siguiendo a Martin en ningún clip.")
    return concat_clips(paths, out_path)
