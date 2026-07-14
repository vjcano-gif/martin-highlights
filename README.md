# ⚽ Highlights de Martin

Genera automáticamente un video con las mejores jugadas de **Martin** (nº 11,
equipo amarillo) a partir del link de YouTube de un partido — sin descargar el
video a mano y sin instalar nada en tu PC.

## ¿Cómo funciona?

```
Link de YouTube ─▶ Google Colab (GPU) ─▶ video de highlights ⚽
                    yt-dlp + YOLOv8 + ffmpeg
   (interfaz web Streamlit corriendo dentro de Colab)
```

1. **yt-dlp** trae el video de YouTube.
2. **YOLOv8 + ByteTrack** detecta y sigue a jugadores y balón.
3. Un filtro de **color** separa al equipo amarillo.
4. Tú marcas **cuál es Martin** (una vez).
5. Se detecta cuándo **el balón está cerca de Martin** → jugadas candidatas.
6. **ffmpeg** corta y une → `highlights_martin.mp4`.

## Archivos

| Archivo | Qué hace |
|---|---|
| `detector.py` | YOLO: detecta jugadores + balón y clasifica el equipo amarillo |
| `possession.py` | Calcula cuándo Martin tiene el balón y agrupa en clips |
| `clipper.py` | Descarga de YouTube + corte/unión con ffmpeg |
| `app.py` | Interfaz web (Streamlit) |
| `colab_martin.ipynb` | Motor: corre la app con la GPU de Colab |
| `requirements.txt` / `packages.txt` | Dependencias |

## Uso (recomendado: Google Colab)

1. Sube este repo a **GitHub**.
2. Abre `colab_martin.ipynb` en **Google Colab**.
3. Activa la **GPU**: *Entorno de ejecución → Cambiar tipo de entorno → GPU*.
4. Cambia `TU_USUARIO` por tu usuario de GitHub y ejecuta las celdas.
5. Abre el enlace `loca.lt` que aparece, usa la IP como contraseña.
6. En la app: pega el link → analiza → marca a Martin → descarga el video.

> **Consejo:** la primera vez pon *"Analizar solo los primeros N minutos = 2"*
> para probar rápido. Cuando funcione, ponlo en `0` para el partido completo.

## Límites honestos (para que no te frustres)

- **"Las mejores jugadas"** se aproxima por *cercanía al balón*, no por talento.
  Es un buen filtro, pero tú das el visto bueno final.
- El **seguimiento del mismo jugador** puede cortarse a lo largo de un partido
  largo (el ID cambia). Si pasa, marca varios IDs que sean Martin, o procesa
  por tiempos más cortos.
- **YouTube** a veces bloquea a los servidores de la nube. Solución: subir un
  `cookies.txt` en *Opciones avanzadas*.
- Reconocer el **número "11"** automáticamente es poco fiable (borroso/lejano),
  por eso usamos el método de "márcalo tú una vez".

## Próximas mejoras posibles

- Re-identificación por apariencia para no perder a Martin (deep re-ID).
- Detección de eventos (disparos a puerta, regates) para priorizar jugadas.
- Música y transiciones automáticas en el resumen.
