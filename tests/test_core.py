import ast
import pathlib
import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from clipper import cut_clip, target_size
from detector import BALL_CLASS, PERSON_CLASS, Detection, analyze_video
from possession import merge_into_clips, possession_times
from reframe import _series, _smooth


def det(frame, cls, box, track=None, width=1000, time=None):
    return Detection(frame, frame / 25 if time is None else time, track, cls,
                     box, .9, False, width, 500)


class PossessionTests(unittest.TestCase):
    def test_uses_real_frame_width(self):
        data = [det(0, PERSON_CLASS, (400, 100, 450, 200), 7),
                det(0, BALL_CLASS, (500, 150, 510, 160))]
        self.assertEqual(possession_times(data, [7], .06), [0.0])

    def test_interpolates_brief_ball_loss(self):
        data = [det(0, BALL_CLASS, (0, 0, 10, 10)),
                det(2, PERSON_CLASS, (45, 0, 60, 20), 7, time=.08),
                det(4, BALL_CLASS, (90, 0, 100, 10))]
        self.assertEqual(possession_times(data, [7], .01), [.08])

    def test_events_need_two_detections(self):
        self.assertEqual(merge_into_clips([3.0]), [])
        self.assertEqual(merge_into_clips([3.0, 3.0]), [])
        self.assertEqual(merge_into_clips([3.0, 3.1]), [(1.5, 4.6)])

    def test_missing_dimensions_rejected(self):
        data = [det(0, PERSON_CLASS, (0, 0, 10, 10), 7, width=0),
                det(0, BALL_CLASS, (0, 0, 2, 2), width=0)]
        with self.assertRaises(ValueError):
            possession_times(data, [7])


class DetectorTests(unittest.TestCase):
    class Capture:
        def isOpened(self): return True
        def get(self, prop):
            import cv2
            return {cv2.CAP_PROP_FPS: 25, cv2.CAP_PROP_FRAME_COUNT: 5}.get(prop, 0)
        def release(self): pass

    def test_stride_preserves_timestamps_dimensions_and_progress(self):
        box = SimpleNamespace(cls=np.array([BALL_CLASS]),
                              xyxy=np.array([[1, 2, 3, 4]]),
                              conf=np.array([.8]), id=None)
        results = [SimpleNamespace(orig_img=np.zeros((480, 640, 3), np.uint8),
                                   boxes=[box]) for _ in range(3)]
        model = SimpleNamespace(track=lambda **kwargs: results)
        progress = []
        with patch('detector.cv2.VideoCapture', return_value=self.Capture()), \
             patch('detector.YOLO', return_value=model):
            detections, fps, total = analyze_video('video.mp4', vid_stride=2,
                                                    progress=lambda i, n: progress.append((i, n)))
        self.assertEqual([d.frame_idx for d in detections], [0, 2, 4])
        self.assertEqual([d.time_s for d in detections], [0, .08, .16])
        self.assertTrue(all((d.frame_width, d.frame_height) == (640, 480)
                            for d in detections))
        self.assertEqual((fps, total), (25, 5))
        self.assertEqual(progress[-1], (5, 5))


class ReframeTests(unittest.TestCase):
    def test_smoothing_preserves_edges(self):
        values = np.full(20, 100.0)
        np.testing.assert_allclose(_smooth(values, 5), values)

    def test_series_interpolates_and_extends_edges(self):
        x, y = _series({2: (10, 20), 4: (30, 40)}, 1, 5)
        np.testing.assert_allclose(x, [10, 10, 20, 30, 30])
        np.testing.assert_allclose(y, [20, 20, 30, 40, 40])


class ClipperTests(unittest.TestCase):
    def test_target_sizes_are_even(self):
        self.assertEqual(target_size((16, 9), 720), (1280, 720))
        self.assertEqual(target_size((9, 16), 720), (720, 1280))

    def test_failed_cut_preserves_previous_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, 'result.mp4')
            pathlib.Path(output).write_bytes(b'previous-good-video')
            error = subprocess.CalledProcessError(1, ['ffmpeg'], stderr='fallo controlado')
            with patch('clipper.subprocess.run', side_effect=error):
                with self.assertRaisesRegex(RuntimeError, 'fallo controlado'):
                    cut_clip('source.mp4', 0, 1, output)
            self.assertEqual(pathlib.Path(output).read_bytes(), b'previous-good-video')

    def test_invalid_cut_range_is_rejected(self):
        with self.assertRaises(ValueError):
            cut_clip('source.mp4', 2, 1, 'output.mp4')


class NotebookTests(unittest.TestCase):
    def test_colab_uses_cloudflare_tunnel(self):
        import json
        notebook = json.loads(
            (pathlib.Path(__file__).parents[1] / "colab_martin.ipynb").read_text(encoding="utf8")
        )
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        self.assertIn("trycloudflare.com", source)
        self.assertIn("/_stcore/health", source)
        self.assertNotIn("npx localtunnel", source)


class SyntaxTests(unittest.TestCase):
    def test_python_sources_parse(self):
        root = pathlib.Path(__file__).parents[1]
        for path in root.glob('*.py'):
            with self.subTest(path=path):
                ast.parse(path.read_text(encoding='utf8'))
