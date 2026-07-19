"""
possession.py
-------------
A partir de las detecciones, decide en qué momentos Martin está con el balón
(o muy cerca de él) y agrupa esos instantes en clips.

Nota honesta: "tener el balón" se aproxima por cercanía balón-jugador. No es
perfecto (el balón es chico y rápido), pero para un resumen casero funciona
bien y tú siempre revisas el resultado final.
"""
from __future__ import annotations

from detector import BALL_CLASS, PERSON_CLASS, Detection


def _center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def _dist_point_to_bbox(px, py, bbox) -> float:
    """Distancia de un punto (el balón) al recuadro del jugador."""
    x1, y1, x2, y2 = bbox
    dx = max(x1 - px, 0, px - x2)
    dy = max(y1 - py, 0, py - y2)
    return (dx * dx + dy * dy) ** 0.5


def possession_times(detections: list[Detection], martin_track_ids,
                     dist_ratio: float = 0.06, interpolate_frames: int = 4) -> list[float]:
    """
    Devuelve los tiempos (en segundos) en los que el balón está cerca de Martin.

    dist_ratio : distancia máxima balón-Martin como fracción del ancho del
                 frame. Súbelo si se te escapan jugadas, bájalo si hay ruido.
    """
    martin_ids = set(martin_track_ids)

    by_frame: dict[int, list[Detection]] = {}
    for d in detections:
        by_frame.setdefault(d.frame_idx, []).append(d)

    # Interpola únicamente huecos breves entre dos observaciones reales del balón.
    balls = sorted((d for d in detections if d.cls == BALL_CLASS),
                   key=lambda d: d.frame_idx)
    for left, right in zip(balls, balls[1:]):
        gap = right.frame_idx - left.frame_idx
        if 1 < gap <= interpolate_frames + 1:
            for frame in range(left.frame_idx + 1, right.frame_idx):
                alpha = (frame - left.frame_idx) / gap
                bbox = tuple(a + (b - a) * alpha
                             for a, b in zip(left.bbox, right.bbox))
                by_frame.setdefault(frame, []).append(Detection(
                    frame, left.time_s + (right.time_s-left.time_s)*alpha,
                    None, BALL_CLASS, bbox, min(left.conf, right.conf), False,
                    left.frame_width or right.frame_width,
                    left.frame_height or right.frame_height))

    times: list[float] = []
    for dets in by_frame.values():
        balls = [d for d in dets if d.cls == BALL_CLASS]
        martins = [d for d in dets
                   if d.cls == PERSON_CLASS and d.track_id in martin_ids]
        if not balls or not martins:
            continue

        frame_w = next((d.frame_width for d in dets if d.frame_width > 0), 0)
        if frame_w <= 0:
            raise ValueError("Las detecciones no incluyen el ancho real del cuadro.")
        thresh = frame_w * dist_ratio

        for ball in balls:
            bx, by = _center(ball.bbox)
            if any(_dist_point_to_bbox(bx, by, m.bbox) <= thresh for m in martins):
                times.append(dets[0].time_s)
                break

    return sorted(times)


def merge_into_clips(times: list[float], gap: float = 2.0, pad: float = 1.5,
                     min_len: float = 1.0, min_detections: int = 2) -> list[tuple[float, float]]:
    """
    Agrupa instantes cercanos en clips [inicio, fin].

    gap     : si dos instantes están a <gap segundos, van al mismo clip.
    pad     : segundos extra de contexto antes y después de cada jugada.
    min_len : descarta clips más cortos que esto (probablemente ruido).
    """
    # Una misma detección no puede contar dos veces por entradas duplicadas.
    times = sorted(set(times))
    if not times:
        return []

    clips: list[tuple[float, float, int]] = []
    start = prev = times[0]
    count = 1
    for t in times[1:]:
        if t - prev <= gap:
            prev = t
            count += 1
        else:
            clips.append((start, prev, count))
            start = prev = t
            count = 1
    clips.append((start, prev, count))

    out: list[tuple[float, float]] = []
    for s, e, count in clips:
        if count < min_detections:
            continue
        s2 = max(s - pad, 0.0)
        e2 = e + pad
        if e2 - s2 >= min_len:
            out.append((s2, e2))
    return out
