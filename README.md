# ⚽ Highlights de Martin

## Abrir la aplicación (sin instalar nada)

[![Abrir en Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vjcano-gif/martin-highlights/blob/main/colab_martin.ipynb)

1. Pulsa el botón **Abrir en Google Colab**.
2. En Colab, pulsa **Entorno de ejecución → Ejecutar todas**.
3. Si Colab pregunta si confías en el notebook, pulsa **Ejecutar de todos modos**.
4. Al final, pulsa el enlace grande **Abrir Highlights de Martin**. La dirección correcta termina en `trycloudflare.com`; cierra cualquier pestaña `loca.lt`.

> La primera ejecución puede tardar varios minutos mientras Colab instala las dependencias.

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

1. Abre `colab_martin.ipynb` de `vjcano-gif/martin-highlights` en **Google Colab**.
2. Ejecuta las celdas (se puede repetir la ejecución sin volver a clonar).
3. Activa la **GPU**: *Entorno de ejecución → Cambiar tipo de entorno → GPU*.
4. Abre el enlace `trycloudflare.com` que aparece en una pestaña nueva. El túnel
   es público: no compartas su dirección.
5. En la app: pega el link → analiza → marca a Martin → descarga el video.

La última celda comprueba que Streamlit responde y termina en cuanto muestra la URL; la app y el túnel siguen activos en segundo plano. No reutilices enlaces antiguos de `loca.lt`: abre únicamente la nueva URL `trycloudflare.com`. Si Colab se queda en **Conectando** antes de ejecutar, selecciona *Entorno de ejecución → Desconectar y eliminar entorno de ejecución*, vuelve a conectar y repite las celdas.

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

## Precisión, estabilidad y privacidad

- Cada detección conserva las dimensiones reales del cuadro; la cercanía al balón
  se calcula con ese ancho y se interpolan pérdidas breves. Los eventos aislados
  se descartan automáticamente.
- El análisis usa `vid_stride=2` de forma predeterminada para acelerar YOLO sin
  modificar los tiempos originales. La interfaz permite ajustarlo.
- La descarga predeterminada es 1080p y la salida 720p. La app avisa cuando un
  formato vertical exige reescalado.
- Solo se aceptan URLs de videos individuales de YouTube. Cada sesión usa un
  directorio temporal independiente y sus cookies temporales se borran al acabar
  la descarga.

## Desarrollo y pruebas

Requiere Python 3.10 o posterior y `ffmpeg` en el `PATH`:

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m compileall -q .
python -m json.tool colab_martin.ipynb >/dev/null
```

GitHub Actions ejecuta estas comprobaciones en cada push y pull request.
