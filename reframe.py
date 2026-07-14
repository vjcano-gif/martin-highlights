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
    kernel = np.ones(win) / win
    return np.convolve(a, kernel, mode="same")


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
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f0, f1 = int(start_s * fps), int(end_s * fps)

    # Ventana de recorte: la más grande posible con la proporción de salida.
    cw, ch = round(H * ar), H
    if cw > W:
        cw, ch = W, round(W / ar)
    cw -= cw % 2
    ch -= ch % 2

    ser = _series(centers, f0, f1)
    if ser is None:
        cap.release()
        return None
    cx, cy = _smooth(ser[0], smooth_win), _smooth(ser[1], smooth_win)

    tmp = tempfile.mktemp(suffix=".mp4")
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (tw, th))

    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    for i in range(f1 - f0 + 1):
        ok, frame = cap.read()
        if not ok:
            break
        x = int(np.clip(cx[i] - cw / 2, 0, max(W - cw, 0)))
        y = int(np.clip(cy[i] - ch / 2, 0, max(H - ch, 0)))
        crop = frame[y:y + ch, x:x + cw]
        writer.write(cv2.resize(crop, (tw, th)))
    writer.release()
    cap.release()

    # Reincorporar el audio original del tramo (el crop de OpenCV no lleva sonido).
    dur = end_s - start_s
    cmd = ["ffmpeg", "-y", "-i", tmp,
           "-ss", f"{start_s:.2f}", "-t", f"{dur:.2f}", "-i", str(src),
           "-map", "0:v:0", "-map", "1:a:0?",
           "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac",
           "-shortest", out_path]
    subprocess.run(cmd, check=True, capture_output=True)
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
