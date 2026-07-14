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
                     dist_ratio: float = 0.06) -> list[float]:
    """
    Devuelve los tiempos (en segundos) en los que el balón está cerca de Martin.

    dist_ratio : distancia máxima balón-Martin como fracción del ancho del
                 frame. Súbelo si se te escapan jugadas, bájalo si hay ruido.
    """
    martin_ids = set(martin_track_ids)

    by_frame: dict[int, list[Detection]] = {}
    for d in detections:
        by_frame.setdefault(d.frame_idx, []).append(d)

    times: list[float] = []
    for dets in by_frame.values():
        balls = [d for d in dets if d.cls == BALL_CLASS]
        martins = [d for d in dets
                   if d.cls == PERSON_CLASS and d.track_id in martin_ids]
        if not balls or not martins:
            continue

        frame_w = max(d.bbox[2] for d in dets)   # ancho aproximado del frame
        thresh = frame_w * dist_ratio

        for ball in balls:
            bx, by = _center(ball.bbox)
            if any(_dist_point_to_bbox(bx, by, m.bbox) <= thresh for m in martins):
                times.append(dets[0].time_s)
                break

    return sorted(times)


def merge_into_clips(times: list[float], gap: float = 2.0, pad: float = 1.5,
                     min_len: float = 1.0) -> list[tuple[float, float]]:
    """
    Agrupa instantes cercanos en clips [inicio, fin].

    gap     : si dos instantes están a <gap segundos, van al mismo clip.
    pad     : segundos extra de contexto antes y después de cada jugada.
    min_len : descarta clips más cortos que esto (probablemente ruido).
    """
    if not times:
        return []

    clips: list[tuple[float, float]] = []
    start = prev = times[0]
    for t in times[1:]:
        if t - prev <= gap:
            prev = t
        else:
            clips.append((start, prev))
            start = prev = t
    clips.append((start, prev))

    out: list[tuple[float, float]] = []
    for s, e in clips:
        s2 = max(s - pad, 0.0)
        e2 = e + pad
        if e2 - s2 >= min_len:
            out.append((s2, e2))
    return out
