import numpy as np
import yaml
from scipy.spatial.transform import Rotation as ScipyRotation


def load_yaml(filepath):
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)


class StereoProjector:
    def __init__(self, config_path, video_width, video_height):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        pylon_calib = load_yaml(config['pylon_camera']['intrinsics_path'])
        thermal_calib = load_yaml(config['thermal_camera']['intrinsics_path'])

        self.K1 = np.array(pylon_calib['camera_matrix']['data']).reshape(3, 3)
        self.K2 = np.array(thermal_calib['camera_matrix']['data']).reshape(3, 3)

        t = np.array(config['thermal_camera']['extrinsics']['translation'])
        q = np.array(config['thermal_camera']['extrinsics']['quaternion'])
        R = ScipyRotation.from_quat(q).as_matrix()

        self.model_params = config['model_parameters']
        self.anonymization_options = config.get('anonymization', {})

        self.K1_inv = np.linalg.inv(self.K1)

        P2_standard = self.K2 @ np.hstack([R, t.reshape(3, 1)])
        H_mirror = np.array([
            [-1, 0, video_width - 1],
            [0, -1, video_height - 1],
            [0, 0, 1]
        ], dtype=float)
        self.P2 = H_mirror @ P2_standard

    def project_bbox(self, bbox_rgb):

        x1_rgb, y1_rgb, x2_rgb, y2_rgb = bbox_rgb
        pixel_height_rgb = y2_rgb - y1_rgb
        if pixel_height_rgb <= 0:
            return None

        fy_pylon = self.K1[1, 1]
        real_head_height = self.model_params['real_head_height_m']
        depth_multiplier = self.model_params['depth_correction_multiplier']

        Z = (real_head_height * fy_pylon) / pixel_height_rgb
        Z *= depth_multiplier

        u_rgb, v_rgb = (x1_rgb + x2_rgb) / 2, (y1_rgb + y2_rgb) / 2
        p_norm = self.K1_inv @ np.array([u_rgb, v_rgb, 1])
        p3d = Z * p_norm

        p2d_h = self.P2 @ np.append(p3d, 1)

        if abs(p2d_h[2]) < 1e-6:
            return None

        offset_x = self.anonymization_options.get('offset_x', 0)
        offset_y = self.anonymization_options.get('offset_y', 0)
        u2 = (p2d_h[0] / p2d_h[2]) + offset_x
        v2 = (p2d_h[1] / p2d_h[2]) + offset_y

        fy_thermal = self.K2[1, 1]
        pixel_height_thermal = (real_head_height * fy_thermal) / Z
        pixel_width_thermal = pixel_height_thermal * 0.75

        x1 = int(u2 - pixel_width_thermal / 2)
        y1 = int(v2 - pixel_height_thermal / 2)
        x2 = int(u2 + pixel_width_thermal / 2)
        y2 = int(v2 + pixel_height_thermal / 2)

        return [x1, y1, x2, y2]
