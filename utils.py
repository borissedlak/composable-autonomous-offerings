import csv
import logging
import os
import time
from typing import Dict

logger = logging.getLogger('multiscale')
ROOT = os.path.dirname(__file__)


def get_env_param(var, default) -> str:
    env = os.environ.get(var)
    if env:
        logger.info(f'Found ENV value for {var}: {env}')
    else:
        env = default
        logger.warning(f"Didn't find ENV value for {var}, default to: {default}")
    return env


def print_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000.0
        logger.info(f"{func.__name__} took {execution_time_ms:.0f} ms to execute")
        # print(f"{func.__name__} took {execution_time_ms:.0f} ms to execute")
        return result

    return wrapper

def smoothstep(x, x0=0.0, x1=1.0) -> float:
    t = np.clip((x - x0) / (x1 - x0), 0.0, 1.0)
    return float(t * t * (3 - 2 * t))

class FPS_:
    def __init__(self, max_fps=300):
        self.prev_time = 0
        self.new_time = 0

        self.time_store = Cyclical_Array(max_fps)

    def tick(self) -> None:
        self.time_store.put(time.time())

    # @print_execution_time
    def get_current_fps(self) -> int:
        current_time = time.time()
        recent_timestamps = [t for t in self.time_store.data if current_time - t <= 1]
        return len(recent_timestamps)


class Cyclical_Array:
    def __init__(self, size):
        self.data = np.zeros(size, dtype=object)
        self.index = 0
        self.size = size

    def put(self, item):
        self.data[self.index % self.size] = item
        self.index = self.index + 1

    def get_average(self):
        return np.mean(self.data, dtype=np.float64)


def convert_prom_multi(raw_result, decimal=False, avg=False):
    return {
        item['metric']["metric_id"]: (float if decimal else int)(item['value'][1])
        for item in raw_result
    }


def filter_tuple(t, name, index):
    return next((item for item in t if item[index] == name), None)


# @print_execution_time
def write_metrics_to_csv(lines, pure_string=False):
    # Define the directory and file name
    directory = ROOT + "/share/metrics"
    file_name = "metrics.csv"
    file_path = os.path.join(directory, file_name)

    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Check if the file exists
    file_exists = os.path.isfile(file_path)

    # Open the file in append mode
    with open(file_path, mode='a', newline='') as file:
        csv_writer = csv.writer(file)

        if not file_exists or os.path.getsize(file_path) == 0:
            csv_writer.writerow(["timestamp", "service_type", "container_id", "avg_p_latency", "s_config", "cores",
                                 "rps", "throughput", "cooldown"])

        if pure_string:
            file.writelines(lines)
        else:
            csv_writer.writerows(lines)
        # print("Wrote lines")


def to_absolut_rps(client_arrivals: Dict[str, int]) -> int:
    return sum(i for i in client_arrivals.values())


def cores_to_threads(cores_reserved):
    return max(1, round(cores_reserved))


class SlidingWindow:
    def __init__(self, window_size):
        self.window_size = window_size
        self.values = []

    def add_value(self, value):
        self.values.append(value)
        # Keep only the last window_size elements
        if len(self.values) > self.window_size:
            self.values.pop(0)

    def get_average(self):
        if not self.values:
            return None
        return round(sum(self.values) / len(self.values), 3)


import matplotlib.pyplot as plt
import numpy as np


def visualize_ndarray(arr, title, cmap="YlGnBu"):
    """
    Visualizes a 2D or 3D numpy array.
    2D: Heatmap with text annotations.
    3D: 3D Scatter plot where point intensity represents value.
    """
    if arr.ndim == 2:
        # --- Existing 2D Logic ---
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(arr, cmap=cmap)
        # plt.colorbar(im, ax=ax)

        ax.set_xticks(np.arange(arr.shape[1]))
        ax.set_yticks(np.arange(arr.shape[0]))
        ax.set_xlabel("Data Quality")
        ax.set_ylabel("Resources")

        threshold = (np.nanmax(arr) + np.nanmin(arr[arr > -np.inf])) / 2. if np.any(arr > -np.inf) else 0
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                if val == -np.inf: continue
                color = "white" if val > threshold else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color)

    elif arr.ndim == 3:
        # --- New 3D Logic ---
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')

        # Get coordinates of all cells that are not -inf
        indices = np.argwhere(arr > -np.inf)
        if len(indices) == 0:
            print("Archive is empty. Nothing to plot.")
            return

        x = indices[:, 0]
        y = indices[:, 1]
        z = indices[:, 2]
        values = arr[arr > -np.inf]

        # Create scatter plot
        # s=size, c=color mapped to values
        img = ax.scatter(x, y, z, c=values, cmap=cmap, s=100, alpha=0.8, edgecolors='w')

        fig.colorbar(img, ax=ax, label='Fitness')

        ax.set_xlabel('Dim 1 (Res: High)')
        ax.set_ylabel('Dim 2 (Res: High)')
        ax.set_zlabel('Dim 3 (Res: Half)')

    else:
        raise ValueError(f"Unsupported dimensions: {arr.ndim}. Only 2D and 3D are supported.")

    # plt.title(title)
    plt.tight_layout()
    plt.savefig(ROOT + f"/figures/{title}.pdf")
    plt.show()

# --- Example Usage ---
# data = np.random.randint(0, 100, size=(8, 8))
# visualize_ndarray(data)