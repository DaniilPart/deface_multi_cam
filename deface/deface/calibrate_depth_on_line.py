import cv2
import json
import yaml
import numpy as np
import argparse
from collections import defaultdict
from scipy.spatial.transform import Rotation as ScipyRotation


def load_yaml(filepath):
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)


def on_trackbar(val):
    pass


def draw_epiline(frame, line, color=(255, 0, 0)):
    _, c, _ = frame.shape
    r = line
    # Avoid division by zero if the line is vertical
    if abs(r[1]) > 1e-6:
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        cv2.line(frame, (x0, y0), (x1, y1), color, 1)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Full calibration UI for stereo projection from a config file.")
    parser.add_argument('--config', default='config.yaml', help="Path to the main configuration file.")
    parser.add_argument('--detections', required=True, help="Path to the JSON file with detections.")
    parser.add_argument('--video', required=True, help="Path to the thermal video or image.")
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        pylon_calib = load_yaml(config['pylon_camera']['intrinsics_path'])
        thermal_calib = load_yaml(config['thermal_camera']['intrinsics_path'])
        t = np.array(config['thermal_camera']['extrinsics']['translation'])
        q = np.array(config['thermal_camera']['extrinsics']['quaternion'])
        initial_params = config.get('model_parameters', {})
        initial_anonymization = config.get('anonymization', {})
        K1 = np.array(pylon_calib['camera_matrix']['data']).reshape(3, 3)
        K2 = np.array(thermal_calib['camera_matrix']['data']).reshape(3, 3)
        R = ScipyRotation.from_quat(q).as_matrix()
        with open(args.detections, 'r') as f:
            detection_data = json.load(f)
        detections = detection_data.get('detections', detection_data)
        fps = detection_data.get('fps', 30)

    except Exception as e:
        print(f"Error loading files/config: {e}")
        return

    is_image = args.video.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))

    if is_image:
        original_image = cv2.imread(args.video)
        if original_image is None:
            print(f"Error opening image: {args.video}")
            return
        H2, W2 = original_image.shape[:2]
    else:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"Error opening video: {args.video}")
            return
        W2 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H2 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    K1_inv = np.linalg.inv(K1)
    P2_standard = K2 @ np.hstack([R, t.reshape(3, 1)])
    H_mirror = np.array([[-1, 0, W2 - 1], [0, -1, H2 - 1], [0, 0, 1]], dtype=float)
    P2 = H_mirror @ P2_standard

    # --- START OF FIX ---
    # The Fundamental Matrix must also account for the mirror transform.
    # We derive it from the projection matrices P1 and P2.
    # P1 is the projection matrix for the first camera (identity rotation, zero translation)
    P1 = K1 @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=float)

    # We need to compute the epipole and pseudo-inverse for F calculation
    # Since P2 is already computed, we can use a generic method.
    # This calculation derives F directly from the final P1 and P2 matrices,
    # which is more robust than calculating E and then F.
    C = np.linalg.svd(P1)[-1][-1, :]
    e2 = P2 @ C
    P1_pinv = np.linalg.pinv(P1)
    F = np.cross(e2, P2 @ P1_pinv, axisa=0, axisb=0).T
    # --- END OF FIX ---

    detections_by_frame = defaultdict(list)
    for det in detections:
        frame_idx = det.get('frame', 0) if 'timestamp_sec' not in det else int(det['timestamp_sec'] * fps)
        detections_by_frame[frame_idx].append(det)

    sorted_det_frames = sorted(detections_by_frame.keys())
    if not sorted_det_frames:
        print("No detections found.")
        return
    current_frame_index = 0

    window_name = 'Full Calibration'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    is_fullscreen = False
    show_epilines = False

    cv2.createTrackbar('Head Height (mm)', window_name, int(initial_params.get('real_head_height_m', 0.25) * 1000), 350,
                       on_trackbar)
    cv2.createTrackbar('Depth Corr (%)', window_name,
                       int((initial_params.get('depth_correction_multiplier', 1.0) - 1.0) * 50 + 50), 100, on_trackbar)
    cv2.createTrackbar('Size Mult (%)', window_name, int(initial_anonymization.get('size_multiplier', 1.3) * 100), 200,
                       on_trackbar)
    cv2.createTrackbar('Offset X (px)', window_name, int(initial_anonymization.get('offset_x', 0)) + 50, 100,
                       on_trackbar)
    cv2.createTrackbar('Offset Y (px)', window_name, int(initial_anonymization.get('offset_y', 0)) + 50, 100,
                       on_trackbar)

    print("\n=== Hotkeys ===")
    print("E - Toggle Epipolar Lines")
    print("F - Toggle Fullscreen")
    print("M - Next frame with detections")
    print(", (comma) - Previous frame with detections")
    print("Q or ESC - Exit and print parameters\n")

    while True:
        head_height_mm = cv2.getTrackbarPos('Head Height (mm)', window_name)
        depth_corr_trackbar = cv2.getTrackbarPos('Depth Corr (%)', window_name)
        size_mult_trackbar = cv2.getTrackbarPos('Size Mult (%)', window_name)
        offset_x_trackbar = cv2.getTrackbarPos('Offset X (px)', window_name)
        offset_y_trackbar = cv2.getTrackbarPos('Offset Y (px)', window_name)

        frame_num = sorted_det_frames[current_frame_index]

        if is_image:
            frame = original_image.copy()
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                current_frame_index = (current_frame_index + 1) % len(sorted_det_frames)
                continue

        depth_multiplier = (depth_corr_trackbar - 50) / 50.0 + 1.0
        size_multiplier = size_mult_trackbar / 100.0
        offset_x = offset_x_trackbar - 50
        offset_y = offset_y_trackbar - 50

        real_head_height_m = head_height_mm / 1000.0
        fy_pylon = K1[1, 1]
        fy_thermal = K2[1, 1]

        center_points_rgb = []

        for det in detections_by_frame[frame_num]:
            u_rgb, v_rgb = (det['bbox'][0] + det['bbox'][2]) / 2, (det['bbox'][1] + det['bbox'][3]) / 2
            center_points_rgb.append([u_rgb, v_rgb])

            pixel_height_rgb = det['bbox'][3] - det['bbox'][1]
            if pixel_height_rgb <= 0 or real_head_height_m <= 0:
                continue

            Z_base = (real_head_height_m * fy_pylon) / pixel_height_rgb
            Z = Z_base * depth_multiplier

            if abs(Z) < 1e-6:
                continue

            p_norm = K1_inv @ np.array([u_rgb, v_rgb, 1])
            p3d = Z * p_norm
            p2d_h = P2 @ np.append(p3d, 1)

            if abs(p2d_h[2]) > 1e-6:
                u2 = (p2d_h[0] / p2d_h[2]) + offset_x
                v2 = (p2d_h[1] / p2d_h[2]) + offset_y

                pixel_height_thermal = (real_head_height_m * fy_thermal) / Z
                pixel_width_thermal = pixel_height_thermal * 0.75
                h = pixel_height_thermal * size_multiplier
                w = pixel_width_thermal * size_multiplier

                x1, y1 = int(u2 - w / 2), int(v2 - h / 2)
                x2, y2 = int(u2 + w / 2), int(v2 + h / 2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

        if show_epilines and len(center_points_rgb) > 0:
            pts1 = np.array(center_points_rgb, dtype=np.float32).reshape(-1, 1, 2)
            lines2 = cv2.computeCorrespondEpilines(pts1, 1, F)
            lines2 = lines2.reshape(-1, 3)
            for line in lines2:
                frame = draw_epiline(frame, line)

        info1 = f"Frame: {frame_num}/{len(sorted_det_frames) - 1}, Head: {head_height_mm}mm"
        info2 = f"Depth:{depth_multiplier:.2f}x, Size:{size_multiplier:.2f}x, Offset:({offset_x},{offset_y})"
        info3 = f"Hotkeys: (E)pilines | (F)ullscreen | (M)next | (,)prev | (Q)uit"
        cv2.putText(frame, info1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, info2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, info3, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 255), 1)
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord('e') or key == ord('E'):
            show_epilines = not show_epilines
            print(f"Epipolar lines: {'ON' if show_epilines else 'OFF'}")
        elif key == ord('f') or key == ord('F'):
            is_fullscreen = not is_fullscreen
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL)
            print(f"Fullscreen mode: {'ON' if is_fullscreen else 'OFF'}")
        elif key == ord('m') or key == ord('M'):
            current_frame_index = min(len(sorted_det_frames) - 1, current_frame_index + 1)
        elif key == ord(','):
            current_frame_index = max(0, current_frame_index - 1)

    if not is_image:
        cap.release()
    cv2.destroyAllWindows()

    final_head_height_mm = cv2.getTrackbarPos('Head Height (mm)', window_name)
    final_depth_corr = (cv2.getTrackbarPos('Depth Corr (%)', window_name) - 50) / 50.0 + 1.0
    final_size_mult = cv2.getTrackbarPos('Size Mult (%)', window_name) / 100.0
    final_offset_x = cv2.getTrackbarPos('Offset X (px)', window_name) - 50
    final_offset_y = cv2.getTrackbarPos('Offset Y (px)', window_name) - 50

    print("\n=== Final Calibrated Parameters ===")
    print("Copy this block and paste it into your config.yaml\n")
    print("model_parameters:")
    print(f"  real_head_height_m: {final_head_height_mm / 1000.0:.4f}")
    print(f"  depth_correction_multiplier: {final_depth_corr:.4f}")
    print("\nanonymization:")
    print(f"  size_multiplier: {final_size_mult:.4f}")
    print(f"  offset_x: {final_offset_x}")
    print(f"  offset_y: {final_offset_y}")
    print("\n")


if __name__ == '__main__':
    main()
