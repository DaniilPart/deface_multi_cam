import argparse
import json
import yaml
import cv2
import imageio
import tqdm
import numpy as np
import skimage.draw
from collections import defaultdict
from stereo_projector import StereoProjector


def scale_bb(x1, y1, x2, y2, mask_scale=1.0):
    s = mask_scale - 1.0
    h, w = y2 - y1, x2 - x1
    y1 -= h * s
    y2 += h * s
    x1 -= w * s
    x2 += w * s
    return np.round([x1, y1, x2, y2]).astype(int)


def draw_det(frame, x1, y1, x2, y2, replacewith='blur', ellipse=True, mosaicsize=20, replaceimg=None):
    if replacewith == 'solid':
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), -1)
    elif replacewith == 'blur':
        bf = 2
        kernel_w = abs(x2 - x1) // bf
        kernel_h = abs(y2 - y1) // bf
        if kernel_w % 2 == 0: kernel_w += 1
        if kernel_h % 2 == 0: kernel_h += 1
        if kernel_w > 0 and kernel_h > 0:
            blurred_box = cv2.GaussianBlur(frame[y1:y2, x1:x2], (kernel_w, kernel_h), 0)
            if ellipse:
                roibox = frame[y1:y2, x1:x2].copy()
                ey, ex = skimage.draw.ellipse((y2 - y1) // 2, (x2 - x1) // 2, (y2 - y1) // 2, (x2 - x1) // 2)
                roibox[ey, ex] = blurred_box[ey, ex]
                frame[y1:y2, x1:x2] = roibox
            else:
                frame[y1:y2, x1:x2] = blurred_box
    elif replacewith == 'mosaic':
        for y in range(y1, y2, mosaicsize):
            for x in range(x1, x2, mosaicsize):
                pt1 = (x, y)
                pt2 = (min(x2, x + mosaicsize), min(y2, y + mosaicsize))
                color = frame[y, x].tolist()
                cv2.rectangle(frame, pt1, pt2, color, -1)
    elif replacewith == 'img':
        if replaceimg is not None:
            target_size = (x2 - x1, y2 - y1)
            if target_size[0] > 0 and target_size[1] > 0:
                resized_img = cv2.resize(replaceimg, target_size)
                if resized_img.shape[2] == 4:
                    alpha = resized_img[:, :, 3:] / 255.0
                    frame[y1:y2, x1:x2] = (frame[y1:y2, x1:x2] * (1 - alpha) +
                                           resized_img[:, :, :3] * alpha).astype(np.uint8)
                else:
                    frame[y1:y2, x1:x2] = resized_img


def anonymize_frame(frame, detections_for_frame, anonymization_options, replaceimg=None):
    for det_bbox in detections_for_frame:
        x1, y1, x2, y2 = det_bbox
        x1, y1, x2, y2 = scale_bb(x1, y1, x2, y2, anonymization_options.get('mask_scale', 1.3))
        y1, y2 = max(0, y1), min(frame.shape[0], y2)
        x1, x2 = max(0, x1), min(frame.shape[1], x2)
        if x1 >= x2 or y1 >= y2:
            continue
        draw_det(frame, x1, y1, x2, y2,
                 replacewith=anonymization_options.get('replacewith', 'blur'),
                 ellipse=anonymization_options.get('ellipse', True),
                 mosaicsize=anonymization_options.get('mosaicsize', 20),
                 replaceimg=replaceimg)


def process_video(input_path, output_path, config_path, detections_json, anonymization_options, replaceimg):
    try:
        reader = imageio.get_reader(input_path)
        meta = reader.get_meta_data()
        fps = meta.get('fps', 30)
        writer = imageio.get_writer(output_path, fps=fps)
        video_width = meta['size'][0]
        video_height = meta['size'][1]
        projector = StereoProjector(config_path, video_width, video_height)
    except Exception as e:
        print(f"Error initializing video or projector: {e}")
        return

    try:
        with open(detections_json, 'r') as f:
            detection_data = json.load(f)
        detections = detection_data.get('detections', detection_data)
        detections_by_frame = defaultdict(list)
        for det in detections:
            frame_idx = int(det.get('timestamp_sec', 0) * fps)
            detections_by_frame[frame_idx].append(det['bbox'])
    except Exception as e:
        print(f"Error loading detections JSON: {e}")
        return

    frame_count = reader.count_frames()
    print(f"Processing video: {frame_count} frames...")

    for frame_number, frame in tqdm.tqdm(enumerate(reader), total=frame_count, desc="Processing video"):
        if frame_number in detections_by_frame:
            bboxes_rgb = detections_by_frame[frame_number]
            bboxes_thermal = []
            for bbox_rgb in bboxes_rgb:
                bbox_thermal = projector.project_bbox(bbox_rgb)
                if bbox_thermal:
                    bboxes_thermal.append(bbox_thermal)
            if bboxes_thermal:
                anonymize_frame(frame, bboxes_thermal, anonymization_options, replaceimg)
        writer.append_data(frame)

    writer.close()
    reader.close()
    print(f"\nVideo processing complete. Saved to: {output_path}")


def process_single_image(input_image, output_image, config_path, detections_json, anonymization_options, replaceimg):
    print(f"Processing single image: {input_image}")
    frame = cv2.imread(input_image)
    if frame is None:
        print(f"Error: Could not read image {input_image}")
        return
    H, W = frame.shape[:2]

    try:
        with open(detections_json, 'r') as f:
            detection_data = json.load(f)
        bboxes_rgb = [det['bbox'] for det in detection_data.get('detections', [])]
    except Exception as e:
        print(f"Error loading detections JSON: {e}")
        return

    try:
        projector = StereoProjector(config_path, W, H)
    except Exception as e:
        print(f"Error creating projector for {input_image}: {e}")
        return

    if bboxes_rgb:
        bboxes_thermal = []
        for bbox_rgb in bboxes_rgb:
            bbox_thermal = projector.project_bbox(bbox_rgb)
            if bbox_thermal:
                bboxes_thermal.append(bbox_thermal)
        if bboxes_thermal:
            anonymize_frame(frame, bboxes_thermal, anonymization_options, replaceimg)

    cv2.imwrite(output_image, frame)
    print(f"Image processing complete. Saved to: {output_image}")


def main():
    parser = argparse.ArgumentParser(
        description="Anonymize faces on thermal video or a single image using stereo projection."
    )
    parser.add_argument('--config', default='config.yaml', help="Path to the main configuration file.")
    parser.add_argument('--detections', required=True,
                        help="Path to the JSON file with detections from the RGB source.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input-video', help="Path to the thermal video to be processed.")
    group.add_argument('--input-image', help="Path to a single thermal image to be processed.")

    parser.add_argument('--output-video', help="Path to save the output video (required if --input-video is used).")
    parser.add_argument('--output-image',
                        help="Path to save the single processed image (required if --input-image is used).")
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        anonymization_options = config.get('anonymization', {})
    except Exception as e:
        print(f"Error loading config file: {e}")
        return

    replaceimg = None
    if anonymization_options.get('replacewith') == 'img':
        img_path = anonymization_options.get('replaceimg_path', 'replace_img.png')
        try:
            replaceimg = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if replaceimg is None:
                print(f"Warning: Could not load replacement image from {img_path}. Falling back to blur.")
                anonymization_options['replacewith'] = 'blur'
            else:
                print(f"Loaded replacement image: {img_path}")
        except Exception as e:
            print(f"Error loading replacement image: {e}. Falling back to blur.")
            anonymization_options['replacewith'] = 'blur'

    if args.input_video:
        if not args.output_video:
            parser.error("--output-video is required when --input-video is specified.")
        process_video(args.input_video, args.output_video, args.config, args.detections,
                      anonymization_options, replaceimg)
    elif args.input_image:
        if not args.output_image:
            parser.error("--output-image is required when --input-image is specified.")
        process_single_image(args.input_image, args.output_image, args.config, args.detections,
                             anonymization_options, replaceimg)


if __name__ == '__main__':
    main()
