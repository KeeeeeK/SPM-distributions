import sys

import numpy as np


def XY_rect(x_mean: float | int, x_range: float | int, x_freq: float | int,
            y_mean: float | int, y_range: float | int, y_freq: float | int):
    return np.mgrid[x_mean - x_range:x_mean + x_range:complex(0, x_freq),
                    y_mean - y_range:y_mean + y_range:complex(0, y_freq)]


def XYZ_from_npy(file_name: str, absolute_path_to_file=False):
    absolute_path = sys.path[0] + '/' + file_name if absolute_path_to_file is False else file_name
    with open(absolute_path, 'rb') as f:
        Z = np.load(f)
        rect_width, rect_height = np.load(f)
    X, Y = XY_rect(0, rect_width / 2, Z.shape[0], 0, rect_height / 2, Z.shape[1])
    return X, Y, Z


def XY_sector(r_mean: float | int, r_range: float | int, r_freq: float | int,
              phi_mean: float | int, phi_range: float | int, phi_freq: float | int):
    r_vals, phi_vals = np.mgrid[r_mean - r_range:r_mean + r_range:complex(0, r_freq),
                       phi_mean - phi_range:phi_mean + phi_range:complex(0, phi_freq)]
    return r_vals * np.cos(phi_vals), r_vals * np.sin(phi_vals)
