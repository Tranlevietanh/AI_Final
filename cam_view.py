"""
FFmpeg Camera Reader - No GStreamer
Handles multi-threaded RTSP reading with FFmpeg low-latency settings
"""

import cv2
import numpy as np
import threading
import time
from typing import List, Optional, Tuple, Callable, Dict, Any


# ========================
# FFmpeg CONFIG
# ========================
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_LATENCY = 50        # ms
SYNC_TOLERANCE = 0.2
SYNC_TIMEOUT = 5.0


# ========================
# FFmpeg Pipeline
# ========================
class FFmpegPipeline:
    @staticmethod
    def build(url: str, width: int, height: int) -> str:
        """
        Build FFmpeg pipeline string for OpenCV.
        Low latency + reconnect options.
        """

        pipeline = (
            f"{url}"
            f"?overrun_nonfatal=1&fifo_size=5000000"  # reduce queue
        )

        return pipeline


# ========================
# Camera Thread Worker
# ========================
def camera_thread_worker(
    url: str,
    idx: int,
    frames_buffer: List[Optional[np.ndarray]],
    frame_timestamps: List[Optional[float]],
    frame_counters: List[int],
    lock: threading.Lock,
    stop_event: threading.Event,
    width: int,
    height: int
):
    print(f"[Camera {idx}] Connecting to: {url}")

    ffmpeg_url = FFmpegPipeline.build(url, width, height)

    # Important: pass FFmpeg low-latency flags
    cap = cv2.VideoCapture(
        ffmpeg_url,
        cv2.CAP_FFMPEG
    )

    # Try to force width/height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"[Camera {idx}] ❌ Failed to open stream")
        return

    print(f"[Camera {idx}] ✅ Connected")

    frame_count = 0

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print(f"[Camera {idx}] ⚠️ Lost frame, retrying…")
            time.sleep(0.1)
            continue

        frame_count += 1
        timestamp = time.time()

        with lock:
            frames_buffer[idx] = frame
            frame_timestamps[idx] = timestamp
            frame_counters[idx] = frame_count

    cap.release()
    print(f"[Camera {idx}] Stopped")


# ========================
# Camera Reader Manager
# ========================
class CameraReader:
    def __init__(self, urls: List[str], width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT):
        self.urls = urls
        self.num_cameras = len(urls)
        self.width = width
        self.height = height

        self.frames_buffer = [None] * self.num_cameras
        self.frame_timestamps = [None] * self.num_cameras
        self.frame_counters = [0] * self.num_cameras

        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.threads: List[threading.Thread] = []

    def start(self):
        for idx, url in enumerate(self.urls):
            t = threading.Thread(
                target=camera_thread_worker,
                args=(url, idx, self.frames_buffer, self.frame_timestamps, self.frame_counters,
                      self.lock, self.stop_event, self.width, self.height),
                daemon=True
            )
            t.start()
            self.threads.append(t)

        time.sleep(1)

    def stop(self):
        self.stop_event.set()
        for t in self.threads:
            t.join()

    def get_frames(self):
        with self.lock:
            return [f.copy() if isinstance(f, np.ndarray) else None for f in self.frames_buffer]

    def wait_for_sync(self, timeout: float = SYNC_TIMEOUT):
        start = time.time()
        while time.time() - start < timeout:
            with self.lock:
                if None in self.frame_timestamps:
                    continue
                if max(self.frame_timestamps) - min(self.frame_timestamps) <= SYNC_TOLERANCE:
                    return self.get_frames()
        return self.get_frames()


# ========================
# MAIN
# ========================
if __name__ == "__main__":

    CAMERA_URLS = [
         "rtsp://admin:BWKUYM@192.168.1.144:554/ch1/main",
    ]

    reader = CameraReader(CAMERA_URLS)
    reader.start()

    try:
        while True:
            frames = reader.get_frames()

            for i, frame in enumerate(frames):
                if frame is not None:
                    frame = cv2.resize(frame,(1080,720))

                    cv2.imshow(f"Camera {i}", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        reader.stop()
        cv2.destroyAllWindows()