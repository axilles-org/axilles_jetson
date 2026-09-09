import os
import time
import queue
import threading
import pandas as pd
from datetime import datetime
from huggingface_hub import HfApi


class RealtimeExoBackup:
    """
    A thread-safe real-time sensor data logger and uploader.
    
    Designed for high-frequency acquisition loops (e.g., 200Hz).
    It buffers sensor frames in memory and handles file saving and 
    Hugging Face uploads in a non-blocking background thread.
    """

    def __init__(self, repo_id: str, hf_token: str, buffer_sec: float = 5.0, local_dir: str = "./backup_logs"):
        """
        Initialize the real-time backup worker.

        :param repo_id: Target Hugging Face repository ID (e.g., "username/dataset-name")
        :param hf_token: Hugging Face API token with Write permission
        :param buffer_sec: Time interval (in seconds) between automatic disk writes and uploads
        :param local_dir: Directory path for local backup CSV files
        """
        self.repo_id = repo_id
        self.hf_token = hf_token
        self.buffer_sec = buffer_sec
        self.local_dir = local_dir

        # Ensure the local logging directory exists
        os.makedirs(self.local_dir, exist_ok=True)

        # Thread-safe FIFO queue for incoming high-frequency sensor frames
        self.data_queue = queue.Queue()

        # Initialize Hugging Face API client
        self.api = HfApi(token=self.hf_token)

        # Worker thread control flag
        self.is_running = True

        # Unique session identifier based on start time
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.batch_counter = 0

        # Start background worker thread
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        print(f"[RealtimeBackup] Backup worker initialized. Target repo: {self.repo_id}")

    def log_sensor_frame(self, frame: dict) -> None:
        """
        Enqueue a single frame of sensor data from the main control loop.
        
        Execution time is minimal (~0.05 ms), preventing latency spikes
        in time-sensitive tasks (e.g., 200Hz execution).

        :param frame: Dictionary containing sensor readings and timestamps
        """
        if self.is_running:
            self.data_queue.put(frame)

    def _worker_loop(self) -> None:
        """
        Background thread worker function.
        
        Continuously pops sensor frames from the queue, flushes them to disk,
        and uploads batches to Hugging Face when the time buffer threshold is reached.
        """
        last_flush_time = time.time()
        buffer = []

        while self.is_running or not self.data_queue.empty():
            # Consume all currently available items in the queue
            try:
                while True:
                    frame = self.data_queue.get_nowait()
                    buffer.append(frame)
                    self.data_queue.task_done()
            except queue.Empty:
                pass

            # Check if buffer time interval elapsed or if system is shutting down
            current_time = time.time()
            if (current_time - last_flush_time >= self.buffer_sec or not self.is_running) and buffer:
                self._flush_buffer(buffer)
                buffer = []  # Clear internal buffer
                last_flush_time = current_time

            # Small sleep to prevent high CPU utilization during idle periods
            time.sleep(0.1)

    def _flush_buffer(self, buffer: list) -> None:
        """
        Save the accumulated buffer into a local CSV file and upload to Hugging Face.

        :param buffer: List of sensor frame dictionaries
        """
        self.batch_counter += 1
        filename = f"exo_session_{self.session_id}_batch_{self.batch_counter:04d}.csv"
        filepath = os.path.join(self.local_dir, filename)

        try:
            # Convert frame buffer to Pandas DataFrame and save locally
            df = pd.DataFrame(buffer)
            df.to_csv(filepath, index=False)
            print(f"[RealtimeBackup] Saved local batch: {filepath} ({len(buffer)} frames)")

            # Upload the batch file to Hugging Face Dataset repository
            self.api.upload_file(
                path_or_fileobj=filepath,
                path_in_repo=f"data/{filename}",
                repo_id=self.repo_id,
                repo_type="dataset"
            )
            print(f"[RealtimeBackup] Uploaded {filename} to HF repository successfully.")

        except Exception as e:
            print(f"[RealtimeBackup Error] Failed during batch flush or upload: {e}")

    def stop(self) -> None:
        """
        Safely shut down the background backup thread.
        
        Flushes any remaining data frames in the queue to disk and uploads them.
        """
        print("[RealtimeBackup] Gracefully stopping backup service and flushing queue...")
        self.is_running = False
        self.worker_thread.join()
        print("[RealtimeBackup] Service stopped successfully.")
