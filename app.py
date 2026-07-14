"""
app.py
------
Interfaz web (Streamlit) para generar los highlights de Martin.

Flujo:
  1) Pegas el link de YouTube  ->  se descarga y analiza con YOLO
  2) Ves a los jugadores amarillos y marcas cuál es Martin
  3) Generas y descargas el video con sus mejores jugadas

IMPORTANTE: el paso 1 (IA) necesita GPU. Corre esta app en Google Colab
(con GPU) usando el notebook incluido, no en el Streamlit Cloud gratuito.
"""
from __future__ import annotations

import os
import tempfile
import time

import cv2
import streamlit as st

from clipper import (ASPECTS, build_highlights, download_youtube, target_size,
                     trim_video)
from detector import COLOR_RANGES, analyze_video, sample_crops
from possession import merge_into_clips, possession_times


def _fmt_time(seg: float) -> str:
    """Segundos -> 'm:ss' legible."""
    seg = int(max(seg, 0))
    return f"{seg // 60}:{seg % 60:02d}"

st.set_page_config(page_title="Highlights de Martin", page_icon="⚽")
st.title("⚽ Highlights de Martin")
st.caption("Arma un video con las mejores jugadas de Martin (nº 11, equipo amarillo).")

st.warning(
    "El análisis con IA necesita GPU. Ejecuta esta app desde Google Colab "
    "(con GPU activada) usando el notebook del repo, no en Streamlit Cloud gratis."
)

ss = st.session_state

# --- Paso 1: entrada --------------------------------------------------------
url = st.text_input("🔗 Link del partido en YouTube")

# --- Tramo del video a analizar (para saltar la previa, etc.) --------------
st.markdown("**¿Qué parte del video analizo?**")
modo = st.radio("Tramo", ["Video completo", "Elegir minutos"],
                horizontal=True, label_visibility="collapsed")
t_ini = t_fin = None
if modo == "Elegir minutos":
    c1, c2 = st.columns(2)
    t_ini = c1.number_input("Desde (min)", min_value=0.0, value=0.0, step=0.5)
    t_fin = c2.number_input("Hasta (min)", min_value=0.0, value=5.0, step=0.5)

# --- Color del uniforme de Martin ------------------------------------------
st.markdown("**¿De qué color juega Martin?**")
color_opt = st.selectbox(
    "Color del uniforme",
    list(COLOR_RANGES.keys()) + ["Sin filtro (elegir entre todos)"],
    index=0,
    help="Si Martin usa un uniforme distinto (ej. arquero), elige su color "
         "o 'Sin filtro' para marcarlo tú entre todos los jugadores.",
    label_visibility="collapsed",
)
team_color = None if color_opt.startswith("Sin filtro") else color_opt

with st.expander("Opciones avanzadas"):
    cookies_file = st.file_uploader(
        "cookies.txt (opcional: si YouTube bloquea la descarga)", type=["txt"]
    )
    max_h = st.select_slider("Resolución de descarga",
                             options=[360, 480, 720, 1080], value=720)
    model_name = st.selectbox(
        "Modelo YOLO",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
        help="n = rápido, m = más preciso pero más lento.",
    )
    color_th = st.slider("Sensibilidad del color", 0.05, 0.40, 0.15, 0.01,
                         help="Bájalo si no detecta el uniforme de Martin.")

if st.button("① Descargar y analizar", disabled=not url, type="primary"):
    cookies_path = None
    if cookies_file:
        cookies_path = os.path.join(tempfile.gettempdir(), "cookies.txt")
        with open(cookies_path, "wb") as f:
            f.write(cookies_file.read())

    with st.spinner("Descargando el video de YouTube..."):
        video_path = download_youtube(url, cookies=cookies_path, max_height=max_h)

    # Recortar al tramo elegido (salta la previa, etc.)
    if modo == "Elegir minutos" and t_fin and t_fin > (t_ini or 0):
        with st.spinner(f"Recortando del minuto {t_ini} al {t_fin}..."):
            video_path = trim_video(video_path, t_ini * 60, t_fin * 60)
    ss.video_path = video_path

    prog = st.progress(0.0, "Analizando con YOLO...")
    _t0 = time.time()
    _last = [0.0]   # para no refrescar la UI en cada frame

    def _cb(i, total):
        if not total:
            return
        ahora = time.time()
        if ahora - _last[0] < 0.5 and i < total:   # refresca ~2 veces/seg
            return
        _last[0] = ahora
        frac = min(i / total, 1.0)
        transcurrido = ahora - _t0
        if frac > 0.02:
            restante = transcurrido / frac - transcurrido
            txt = (f"{frac*100:.0f}%  ·  frame {i}/{total}  ·  "
                   f"faltan ~{_fmt_time(restante)}  "
                   f"(lleva {_fmt_time(transcurrido)})")
        else:
            txt = f"Frame {i}/{total} · calculando tiempo restante..."
        prog.progress(frac, txt)

    dets, fps, total = analyze_video(
        video_path,
        model_name=model_name,
        team_color=team_color,
        color_threshold=color_th,
        progress=_cb,
    )
    ss.detections = dets
    ss.fps = fps

    tids = sorted({d.track_id for d in dets
                   if d.is_target and d.track_id is not None})
    ss.target_ids = tids
    ss.crops = sample_crops(video_path, dets, set(tids))
    etiqueta = "jugadores" if team_color is None else f"jugadores de {team_color}"
    st.success(f"Análisis listo. Detecté {len(tids)} {etiqueta}.")

# --- Paso 2: identificar a Martin ------------------------------------------
if ss.get("target_ids"):
    st.subheader("② ¿Cuál es Martin?")
    st.caption("Marca la casilla del/los recortes que sean Martin (el nº 11).")

    crops = ss.get("crops", {})
    chosen: list[int] = []
    cols = st.columns(4)
    for i, tid in enumerate(ss.target_ids):
        c = cols[i % 4]
        if tid in crops:
            c.image(cv2.cvtColor(crops[tid], cv2.COLOR_BGR2RGB),
                    caption=f"ID {tid}", width=130)
        if c.checkbox(f"Es Martin", key=f"martin_{tid}"):
            chosen.append(tid)
    ss.martin_ids = chosen

    # --- Paso 3: generar highlights ----------------------------------------
    st.subheader("③ Generar el video")
    dist = st.slider("Sensibilidad (cercanía al balón)", 0.02, 0.15, 0.06, 0.01,
                     help="Más alto = más jugadas capturadas (y más ruido).")

    st.markdown("**Formato para redes**")
    fc1, fc2, fc3 = st.columns(3)
    aspect_label = fc1.selectbox("Orientación", list(ASPECTS.keys()))
    quality = fc2.selectbox("Calidad", [1080, 720, 480], index=0,
                            format_func=lambda q: f"{q}p")
    fit_label = fc3.selectbox("Ajuste", ["Rellenar (recorta bordes)",
                                         "Encajar (barras negras)"])
    fit = "crop" if fit_label.startswith("Rellenar") else "pad"
    size = target_size(ASPECTS[aspect_label], quality)
    st.caption(f"Salida: {size[0]}×{size[1]} px · MP4 (H.264) — compatible con "
               "Instagram, WhatsApp, TikTok y YouTube.")

    seguir = st.checkbox(
        "🎯 Que el encuadre siga a Martin (recorte dinámico)",
        value=size[0] < size[1],   # activado por defecto en vertical
        help="La ventana se mueve para mantener a Martin centrado. Ideal para "
             "formato vertical; evita que se salga del cuadro.",
    )

    if st.button("⚡ Generar highlights de Martin", disabled=not chosen):
        with st.spinner("Buscando las jugadas de Martin..."):
            times = possession_times(ss.detections, ss.martin_ids, dist_ratio=dist)
            clips = merge_into_clips(times)

        if not clips:
            st.error("No encontré jugadas. Sube la sensibilidad o revisa el ID.")
        else:
            cut_prog = st.progress(0.0, f"Cortando {len(clips)} jugadas...")

            def _cut_cb(i, total):
                cut_prog.progress(i / total, f"Clip {i}/{total} listo")

            if seguir:
                from reframe import build_following_highlights, martin_centers
                centers = martin_centers(ss.detections, ss.martin_ids)
                out = build_following_highlights(
                    ss.video_path, clips, centers, ss.fps, size,
                    progress=_cut_cb)
            else:
                out = build_highlights(ss.video_path, clips, size=size, fit=fit,
                                       progress=_cut_cb)
            st.success(f"¡Listo! {len(clips)} jugadas de Martin en "
                       f"{size[0]}×{size[1]}. 🎬")
            st.video(out)
            with open(out, "rb") as f:
                st.download_button("⬇️ Descargar highlights_martin.mp4", f,
                                   file_name="highlights_martin.mp4",
                                   mime="video/mp4")
