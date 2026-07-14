"""
clipper.py
----------
Trae el video de YouTube (yt-dlp, sin que tú lo descargues a mano) y corta /
une los clips con ffmpeg para producir el video final de highlights.
"""
from __future__ import annotations

import os
import subprocess
import tempfile


def download_youtube(url: str, out_path: str = "match.mp4",
                     cookies: str | None = None, max_height: int = 720) -> str:
    """
    Descarga el video con yt-dlp.

    cookies : ruta a un cookies.txt exportado de tu navegador. Ayuda cuando
              YouTube bloquea al servidor con "confirma que no eres un robot"
              (típico desde IPs de la nube como Colab).
    """
    fmt = f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]"
    cmd = ["yt-dlp", "-f", fmt, "--merge-output-format", "mp4", "-o", out_path]
    if cookies:
        cmd += ["--cookies", cookies]
    cmd.append(url)
    subprocess.run(cmd, check=True)
    return out_path


def trim_video(src: str, start: float, end: float,
               out_path: str = "segment.mp4") -> str:
    """
    Recorta el tramo [start, end] (en segundos) para analizar solo esa parte
    (ej. saltar la previa). Copia sin recodificar => rapidísimo.
    """
    dur = end - start
    cmd = ["ffmpeg", "-y", "-ss", f"{start:.2f}", "-i", src,
           "-t", f"{dur:.2f}", "-c", "copy", out_path]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


# --- Formatos de salida para redes -----------------------------------------
# short = lado corto en píxeles (la calidad). El otro lado se calcula solo.
ASPECTS = {
    "Vertical (9:16) — Stories/TikTok/Reels": (9, 16),
    "Horizontal (16:9) — YouTube/normal": (16, 9),
    "Cuadrado (1:1) — feed": (1, 1),
}


def target_size(aspect: tuple[int, int], quality_short: int) -> tuple[int, int]:
    """Calcula (ancho, alto) en píxeles, siempre pares (requisito de H.264)."""
    aw, ah = aspect
    if aw <= ah:                       # vertical o cuadrado -> el ancho es el lado corto
        w, h = quality_short, round(quality_short * ah / aw)
    else:                              # horizontal -> el alto es el lado corto
        h, w = quality_short, round(quality_short * aw / ah)
    return (w - w % 2, h - h % 2)


def _format_filter(size: tuple[int, int], fit: str) -> str:
    """Filtro ffmpeg para llevar el clip al tamaño pedido."""
    w, h = size
    if fit == "pad":                   # encaja todo el cuadro y rellena con negro
        return (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
                f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black")
    # 'crop' (por defecto): llena la pantalla recortando los bordes (centrado)
    return (f"scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"crop={w}:{h}")


def cut_clip(src: str, start: float, end: float, out_path: str,
             size: tuple[int, int] | None = None, fit: str = "crop") -> str:
    """
    Corta un fragmento [start, end] re-codificando (cortes precisos).
    size : (ancho, alto) de salida. Si es None, mantiene el original.
    fit  : 'crop' (llena, recorta bordes) o 'pad' (encaja, barras negras).
    """
    dur = end - start
    cmd = ["ffmpeg", "-y", "-ss", f"{start:.2f}", "-i", src, "-t", f"{dur:.2f}"]
    if size:
        cmd += ["-vf", _format_filter(size, fit)]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac",
            "-avoid_negative_ts", "make_zero", out_path]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


def make_preview(src: str, out_path: str = "preview.mp4", seconds: int = 8,
                 height: int = 480,
                 watermark: str = "MUESTRA - paga para ver completo") -> str:
    """
    Genera una PREVIA corta con marca de agua para el marketplace.
    Baja resolución + sin audio + texto encima: engancha pero no sirve como
    producto final (para eso hay que pagar).
    """
    wm = watermark.replace(":", r"\:").replace("'", "")
    vf = (f"scale=-2:{height},"
          f"drawtext=text='{wm}':x=(w-text_w)/2:y=h-h/8:"
          f"fontsize={height//18}:fontcolor=white:"
          f"box=1:boxcolor=black@0.5:boxborderw=10")
    cmd = ["ffmpeg", "-y", "-i", src, "-t", str(seconds), "-vf", vf,
           "-c:v", "libx264", "-preset", "veryfast", "-an", out_path]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


def concat_clips(clip_paths: list[str],
                 out_path: str = "highlights_martin.mp4") -> str:
    """Une varios clips en un solo video."""
    if not clip_paths:
        raise ValueError("No hay clips para unir.")

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     encoding="utf-8") as f:
        for p in clip_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
        list_file = f.name

    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
           "-i", list_file, "-c", "copy", out_path]
    subprocess.run(cmd, check=True, capture_output=True)
    os.unlink(list_file)
    return out_path


def build_highlights(src_video: str, clips: list[tuple[float, float]],
                     workdir: str = "clips",
                     out_path: str = "highlights_martin.mp4",
                     size: tuple[int, int] | None = None, fit: str = "crop",
                     progress=None) -> str:
    """Corta todos los clips (con el formato pedido) y los une en el video final."""
    os.makedirs(workdir, exist_ok=True)
    paths = []
    for i, (s, e) in enumerate(clips):
        p = os.path.join(workdir, f"clip_{i:03d}.mp4")
        cut_clip(src_video, s, e, p, size=size, fit=fit)
        paths.append(p)
        if progress:
            progress(i + 1, len(clips))
    return concat_clips(paths, out_path)
