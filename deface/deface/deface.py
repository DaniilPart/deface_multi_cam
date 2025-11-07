#!/usr/bin/env python3

import argparse
import json
import mimetypes
import os
from typing import Dict, Tuple

import tqdm
import skimage.draw
import numpy as np
import imageio
import imageio.v2 as iio
import imageio.plugins.ffmpeg
import cv2

from deface import __version__
from deface.centerface import CenterFace


def scale_bb(x1, y1, x2, y2, mask_scale=1.0):
    s = mask_scale - 1.0
    h, w = y2 - y1, x2 - x1
    y1 -= h * s
    y2 += h * s
    x1 -= w * s
    x2 += w * s
    return np.round([x1, y1, x2, y2]).astype(int)


def draw_det(frame, score, det_idx, x1, y1, x2, y2, replacewith: str = 'blur', ellipse: bool = True,
             draw_scores: bool = False, ovcolor: Tuple[int] = (0, 0, 0), replaceimg=None, mosaicsize: int = 20):
    if replacewith == 'solid':
        cv2.rectangle(frame, (x1, y1), (x2, y2), ovcolor, -1)
    elif replacewith == 'blur':
        bf = 2
        blurred_box = cv2.blur(frame[y1:y2, x1:x2], (abs(x2 - x1) // bf, abs(y2 - y1) // bf))
        if ellipse:
            roibox = frame[y1:y2, x1:x2]
            ey, ex = skimage.draw.ellipse((y2 - y1) // 2, (x2 - x1) // 2, (y2 - y1) // 2, (x2 - x1) // 2)
            roibox[ey, ex] = blurred_box[ey, ex]
            frame[y1:y2, x1:x2] = roibox
        else:
            frame[y1:y2, x1:x2] = blurred_box
    elif replacewith == 'img':
        target_size = (x2 - x1, y2 - y1)
        resized_replaceimg = cv2.resize(replaceimg, target_size)
        if replaceimg.shape[2] == 3:
            frame[y1:y2, x1:x2] = resized_replaceimg
        elif replaceimg.shape[2] == 4:
            frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2] * (1 - resized_replaceimg[:, :, 3:] / 255) + resized_replaceimg[
                :, :, :3] * (resized_replaceimg[:, :, 3:] / 255)
    elif replacewith == 'mosaic':
        for y in range(y1, y2, mosaicsize):
            for x in range(x1, x2, mosaicsize):
                pt1 = (x, y)
                pt2 = (min(x2, x + mosaicsize - 1), min(y2, y + mosaicsize - 1))
                color = (int(frame[y, x][0]), int(frame[y, x][1]), int(frame[y, x][2]))
                cv2.rectangle(frame, pt1, pt2, color, -1)
    if draw_scores: cv2.putText(frame, f'{score:.2f}', (x1 + 0, y1 - 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0))


def anonymize_frame(dets, frame, mask_scale, replacewith, ellipse, draw_scores, replaceimg, mosaicsize):
    for i, det in enumerate(dets):
        boxes, score = det[:4], det[4]
        x1, y1, x2, y2 = boxes.astype(int)
        x1, y1, x2, y2 = scale_bb(x1, y1, x2, y2, mask_scale)
        y1, y2 = max(0, y1), min(frame.shape[0] - 1, y2)
        x1, x2 = max(0, x1), min(frame.shape[1] - 1, x2)
        draw_det(frame, score, i, x1, y1, x2, y2, replacewith=replacewith, ellipse=ellipse, draw_scores=draw_scores,
                 replaceimg=replaceimg, mosaicsize=mosaicsize)


def cam_read_iter(reader):
    while True: yield reader.get_next_data()


def video_detect(
        ipath: str, opath: str, centerface: CenterFace, threshold: float,
        enable_preview: bool, cam: bool, nested: bool, replacewith: str,
        mask_scale: float, ellipse: bool, draw_scores: bool, ffmpeg_config: Dict[str, str],
        detections_output_path: str = None,
        replaceimg=None, keep_audio: bool = False, mosaicsize: int = 20,
        disable_progress_output=False):
    try:
        reader: imageio.plugins.ffmpeg.FfmpegFormat.Reader = imageio.get_reader(ipath)
        meta = reader.get_meta_data()
        fps = meta.get('fps', 30)
    except Exception as e:
        print(f"Could not open file {ipath}: {e}")
        return

    read_iter = cam_read_iter(reader) if cam else reader.iter_data()
    nframes = None if cam else reader.count_frames()
    bar = tqdm.tqdm(dynamic_ncols=True, total=nframes, disable=disable_progress_output or cam,
                    position=1 if nested else 0, leave=True)

    writer = None
    if opath is not None:
        _ffmpeg_config = ffmpeg_config.copy()
        _ffmpeg_config.setdefault('fps', meta.get('fps', 30))
        if keep_audio and meta.get('audio_codec'):
            _ffmpeg_config.setdefault('audio_path', ipath)
            _ffmpeg_config.setdefault('audio_codec', 'copy')
        writer = imageio.get_writer(opath, format='FFMPEG', mode='I', **_ffmpeg_config)

    all_detections = []
    frame_number = 0

    try:
        for frame in read_iter:
            dets, _ = centerface(frame, threshold=threshold)

            if dets.shape[0] > 0:
                timestamp_sec = frame_number / fps

                for det in dets:
                    boxes, score = det[:4], det[4]
                    x1, y1, x2, y2 = map(int, boxes)

                    all_detections.append({
                        'timestamp_sec': timestamp_sec,
                        'bbox': [x1, y1, x2, y2],
                        'height': y2 - y1,
                        'width': x2 - x1,
                        'score': float(score)
                    })

            anonymize_frame(dets, frame, mask_scale, replacewith, ellipse, draw_scores, replaceimg, mosaicsize)

            if writer: writer.append_data(frame)

            if enable_preview:
                cv2.imshow('Preview', frame[:, :, ::-1])
                if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                    break

            bar.update()
            frame_number += 1

    finally:
        print("\nCleaning up and finalizing files...")

        reader.close()
        bar.close()
        if writer: writer.close()
        if enable_preview: cv2.destroyAllWindows()

        if detections_output_path and all_detections:
            print(f"Saving {len(all_detections)} detected faces to {detections_output_path}...")
            try:
                output_data = {
                    "source_video": ipath,
                    "fps": fps,
                    "detections": all_detections
                }
                with open(detections_output_path, 'w') as f:
                    json.dump(output_data, f, indent=4)
                print("...Data successfully saved.")
            except IOError as e:
                print(f"Error saving detection data: {e}")
        elif detections_output_path:
            print("No detections were made, so no data file was saved.")


def image_detect(
        ipath: str, opath: str, centerface: CenterFace, threshold: float, replacewith: str,
        mask_scale: float, ellipse: bool, draw_scores: bool, enable_preview: bool,
        keep_metadata: bool, detections_output_path: str = None, replaceimg=None, mosaicsize: int = 20):
    frame = iio.imread(ipath)
    exif_dict = None
    if keep_metadata:
        metadata = imageio.v3.immeta(ipath)
        exif_dict = metadata.get("exif", None)

    dets, _ = centerface(frame, threshold=threshold)

    all_detections = []
    if dets.shape[0] > 0:
        for det in dets:
            boxes, score = det[:4], det[4]
            x1, y1, x2, y2 = map(int, boxes)
            all_detections.append({
                'frame': 0,
                'bbox': [x1, y1, x2, y2],
                'height': y2 - y1, 'width': x2 - x1, 'score': float(score)
            })

    anonymize_frame(dets, frame, mask_scale, replacewith, ellipse, draw_scores, replaceimg, mosaicsize)
    imageio.imsave(opath, frame, exif=exif_dict if keep_metadata else None)

    if enable_preview:
        cv2.imshow('Preview', frame[:, :, ::-1])
        if cv2.waitKey(0) & 0xFF in [ord('q'), 27]: cv2.destroyAllWindows()

    if detections_output_path:
        print(f"Saving detection data to {detections_output_path}...")
        output_data = {
            "source_image": ipath,
            "detections": all_detections
        }
        with open(detections_output_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        print("...Done.")


def get_file_type(path):
    if path.startswith('<video'): return 'cam'
    if not os.path.isfile(path): return 'notfound'
    mime = mimetypes.guess_type(path)[0]
    if mime is None: return None
    if mime.startswith('video'): return 'video'
    if mime.startswith('image'): return 'image'
    return mime


def parse_cli_args():
    parser = argparse.ArgumentParser(description='Video anonymization by face detection', add_help=False)
    parser.add_argument('input', nargs='*', help='File path(s) or camera device name.')
    parser.add_argument('--output', '-o', default=None, metavar='O', help='Output file name.')
    parser.add_argument('--thresh', '-t', default=0.49, type=float, metavar='T', help='Detection threshold.')#
    parser.add_argument('--detections-output', default=None, help='Path to JSON file to save detection data.')
    parser.add_argument('--scale', '-s', default=None, metavar='WxH', help='Downscale images for inference.')
    parser.add_argument('--preview', '-p', default=False, action='store_true', help='Enable live preview.')
    parser.add_argument('--boxes', default=False, action='store_true', help='Use boxes instead of ellipse masks.')
    parser.add_argument('--draw-scores', default=False, action='store_true', help='Draw detection scores.')
    parser.add_argument('--disable-progress-output', default=False, action='store_true',
                        help='Disable video progress output to console.')
    parser.add_argument('--mask-scale', default=1.3, type=float, metavar='M', help='Scale factor for face masks.')
    parser.add_argument('--replacewith', default='blur', choices=['blur', 'solid', 'none', 'img', 'mosaic'],
                        help='Anonymization filter mode.')
    parser.add_argument('--replaceimg', default='replace_img.png', help='Image for --replacewith img option.')
    parser.add_argument('--mosaicsize', default=20, type=int, metavar='width', help='Mosaic size.')
    parser.add_argument('--keep-audio', '-k', default=False, action='store_true',
                        help='Keep audio from video source file.')
    parser.add_argument('--ffmpeg-config', default={"codec": "libx264"}, type=json.loads,
                        help='FFMPEG config arguments.')
    parser.add_argument('--backend', default='auto', choices=['auto', 'onnxrt', 'opencv'],
                        help='Backend for ONNX model execution.')
    parser.add_argument('--execution-provider', '--ep', default=None, metavar='EP',
                        help='Override onnxrt execution provider.')
    parser.add_argument('--version', action='version', version=__version__, help='Print version number and exit.')
    parser.add_argument('--keep-metadata', '-m', default=False, action='store_true',
                        help='Keep metadata of the original image.')
    parser.add_argument('--help', '-h', action='help', help='Show this help message and exit.')
    args = parser.parse_args()
    if not args.input: parser.print_help(); exit(1)
    if args.input == ['cam']: args.input, args.preview = ['<video0>'], True
    return args


def main():
    args = parse_cli_args()
    ipaths = [p for path in args.input for p in
              ([os.path.join(path, f) for f in os.listdir(path)] if os.path.isdir(path) else [path])]

    centerface = CenterFace(in_shape=(int(w), int(h)) if args.scale and (w, h := args.scale.split('x')) else None,
                            backend=args.backend, override_execution_provider=args.execution_provider)

    replaceimg = imageio.imread(args.replaceimg) if args.replacewith == "img" else None

    multi_file = len(ipaths) > 1
    if multi_file: ipaths = tqdm.tqdm(ipaths, position=0, dynamic_ncols=True, desc='Batch progress')

    for ipath in ipaths:
        opath = args.output if not multi_file else f'{os.path.splitext(ipath)[0]}_anonymized{os.path.splitext(ipath)[1]}'
        filetype = get_file_type(ipath)

        detections_path = args.detections_output
        if detections_path and multi_file:
            detections_path = f'{os.path.splitext(ipath)[0]}_detections.json'

        print(f'Input:  {ipath}\nOutput: {opath}')
        if opath is None and not args.preview: print(
            'No output file specified and preview is disabled. No output will be produced.')

        if filetype == 'video' or filetype == 'cam':
            video_detect(ipath=ipath, opath=opath, centerface=centerface, threshold=args.thresh,
                         cam=(filetype == 'cam'), replacewith=args.replacewith, mask_scale=args.mask_scale,
                         ellipse=not args.boxes, draw_scores=args.draw_scores, enable_preview=args.preview,
                         nested=multi_file, keep_audio=args.keep_audio, ffmpeg_config=args.ffmpeg_config,
                         replaceimg=replaceimg, mosaicsize=args.mosaicsize,
                         disable_progress_output=args.disable_progress_output,
                         detections_output_path=detections_path)
        elif filetype == 'image':
            image_detect(ipath=ipath, opath=opath, centerface=centerface, threshold=args.thresh,
                         replacewith=args.replacewith, mask_scale=args.mask_scale, ellipse=not args.boxes,
                         draw_scores=args.draw_scores, enable_preview=args.preview, keep_metadata=args.keep_metadata,
                         replaceimg=replaceimg, mosaicsize=args.mosaicsize,
                         detections_output_path=detections_path)
        else:
            print(f'Skipping file with unknown type: {ipath}')


if __name__ == '__main__':
    main()

