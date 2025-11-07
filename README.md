

# Multi-Camera Face Anonymization Tool

This project is based on the original [deface repository by ORB-HD](https://github.com/ORB-HD/deface).
This project provides tools to anonymize faces in thermal videos and images by projecting detections from a paired RGB camera using stereo calibration. It includes an interactive calibration UI and a projection-based anonymizer.

## Features

- Anonymize faces in thermal videos or single images using RGB detections.
- Interactive calibration UI with epipolar line visualization.
- Multiple anonymization modes: blur, mosaic, solid color, or image replacement.
- Ready-to-run examples with included test media.

## Repository layout

```text
camera_conf/                  # Calibration and configuration
  config.yaml                 # Main config linking intrinsics/extrinsics and model params
  rgb_intrinsics_basler.yaml
  thermal_intrinsics_flir_boson.yaml
deface/deface/
  deface.py                   # Detector & anonymizer (RGB-side detection, optional anonymization)
  calibrate_depth_on_line.py  # Interactive projection calibration UI
  blur_thermal_video.py       # Final thermal anonymization using calibrated projection
  stereo_projector.py         # Projection core
pic/
  image.png                   # Test RGB image
videos/
  rgb_rect_synced.mp4         # Test RGB video (LFS)
  thermal_rect.mp4            # Test thermal video (LFS)
requirements.txt
````

## Prerequisites

* Python 3.8–3.11 recommended.
* Git LFS installed for large media.

## Installation

```bash
git clone https://github.com/DaniilPart/deface_multi_cam.git
cd deface_multi_cam

# Git LFS for test media
git lfs install
git lfs pull

# Virtual environment and dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you do not have CUDA, replace onnxruntime-gpu with CPU runtime:

```bash
pip uninstall -y onnxruntime-gpu && pip install onnxruntime
```

## How the detector works (deface.py)

`deface.py` can:

* Detect faces and immediately anonymize the input (image or video).
* Detect faces and save bounding boxes to JSON (used by the projection pipeline).

Key arguments:

* `input`: path(s) to image/video or `cam` for webcam.
* `--detections-output`: path to save JSON with bounding boxes and timestamps.
* `--thresh`: detection threshold (e.g., 0.49).
* `--replacewith`: anonymization mode: `blur`, `mosaic`, `solid`, `img`, `none`.
* `--mask-scale`: face mask scale factor (default 1.3).
* `--preview`: show a live preview window.
* `--keep-audio`: copy audio when writing video outputs.
* `--backend/--ep`: execution backend/provider for ONNX runtime.

Typical detector outputs JSON entries like:

```json
{
  "source_video": "videos/rgb_rect_synced.mp4",
  "fps": 30,
  "detections": [
    {
      "timestamp_sec": 0.200,
      "bbox": [x1, y1, x2, y2],
      "height": 120,
      "width": 98,
      "score": 0.91
    }
  ]
}
```

## Configuration (camera_conf/config.yaml)

Ensure these paths and parameters are correct:

```yaml
pylon_camera:
  intrinsics_path: "camera_conf/rgb_intrinsics_basler.yaml"

thermal_camera:
  intrinsics_path: "camera_conf/thermal_intrinsics_flir_boson.yaml"
  extrinsics:
    translation: [-0.127, 0.043, 0.031]
    quaternion: [0.000, 0.000, -1.000, 0.000]

model_parameters:
  real_head_height_m: 0.25
  depth_correction_multiplier: 1.0

anonymization:
  replacewith: "blur"      # blur | mosaic | solid | img
  size_multiplier: 1.3
  offset_x: 0
  offset_y: 0
  mosaicsize: 20
  ellipse: true
  replaceimg_path: "replace_img.png"
```

You will update `model_parameters` and `anonymization` after calibration.

## Quickstart (with included test media)

### 1) Detect on RGB video (produce JSON)

```bash
deface \
  videos/rgb_rect_synced.mp4 \
  --thresh 0.49 \
  --detections-output detections_video.json \
  --disable-progress-output
```

(Optional) Detect on RGB image:

```bash
deface \
  pic/image.png \
  --thresh 0.49 \
  --detections-output detections_image.json
```

### 2) Calibrate projection (interactive UI)

Use thermal video as the target view:

```bash
python deface/deface/calibrate_depth_on_line.py \
  --config camera_conf/config.yaml \
  --detections detections_video.json \
  --video videos/thermal_rect.mp4
```

Hotkeys:

* `E` — toggle epipolar lines.
* `M` / `,` — next/previous frame with detections.
* `F` — fullscreen toggle.
* `Q` / `ESC` — exit and print calibrated parameters.

Trackbars:

* Head Height (mm)
* Depth Corr (%)
* Size Mult (%)
* Offset X/Y (px)

On exit, the tool prints updated blocks for `model_parameters` and `anonymization`. Copy-paste them into `camera_conf/config.yaml`.

### 3) Anonymize thermal video

```bash
python deface/deface/blur_thermal_video.py \
  --config camera_conf/config.yaml \
  --detections detections_video.json \
  --input-video videos/thermal_rect.mp4 \
  --output-video anonymized_thermal_video.mp4
```

### (Optional) Anonymize thermal image

If you created detections for an image:

```bash
python deface/deface/blur_thermal_video.py \
  --config camera_conf/config.yaml \
  --detections detections_image.json \
  --input-image pic/image.png \
  --output-image anonymized_thermal_image.png
```

## Direct anonymization with detector (RGB-only, no projection)

If you want to quickly anonymize RGB media without the stereo projection:

Video:

```bash
python deface/deface/deface.py \
  videos/rgb_rect_synced.mp4 \
  --output rgb_anonymized.mp4 \
  --replacewith blur \
  --thresh 0.49 \
  --preview
```

Image:

```bash
python deface/deface/deface.py \
  pic/image.png \
  --output rgb_anonymized.png \
  --replacewith mosaic \
  --mosaicsize 20
```

## Troubleshooting

* **No window opens / OpenCV crash:**

  * Ensure a display is available (no headless session), try `export QT_QPA_PLATFORM=xcb` on Linux.
  * Check you passed a valid `--video` or `--input-image` path.

* **Epipolar lines look wrong:**

  * Verify intrinsics YAML paths.
  * Re-check `thermal_camera.extrinsics` translation/quaternion.
  * Use the trackbars to refine depth and offsets; lines should constrain point movement.

* **Git LFS warnings or missing media:**

  * Run `git lfs install && git lfs pull` from the repo root.

* **ONNX GPU issues:**

  * If no CUDA, switch to CPU runtime: `pip uninstall -y onnxruntime-gpu && pip install onnxruntime`.






