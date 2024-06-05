import pathlib
from typing import Generator, List, Tuple

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


class MoviNetPrediction:

    def __init__(
        self,
        logits: tf.Tensor,
        label_map: np.ndarray,
        buffer_frame_idx: List[int],
        buffer_timestamps: List[int],
        top_k: int,
    ):
        """
        Initialize MoviNetPrediction object.

        Args:
            logits (tf.Tensor): Logits from the MoviNet model.
            label_map (np.ndarray): Label map.
            buffer_frame_idx (List[int]): List of frame indices.
            buffer_timestamps (List[int]): List of frame timestamps.
            top_k: Number of top predictions to return.
        """
        self.buffer_frame_idx = buffer_frame_idx
        self.buffer_timestamps = buffer_timestamps

        probs = tf.nn.softmax(logits)

        top_predictions = tf.argsort(probs, axis=-1, direction="DESCENDING")[-1, :top_k]
        top_labels = tf.gather(label_map, top_predictions, axis=-1)
        top_labels = [label.decode("utf8") for label in top_labels.numpy()]

        top_probs = tf.gather(probs, top_predictions, axis=-1)
        top_probs = tf.reshape(top_probs, [-1])
        top_probs = top_probs.numpy().tolist()

        self.predictions = list(zip(top_labels, top_probs))

    @property
    def start_frame(self):
        """
        Return the start frame.
        """
        return self.buffer_frame_idx[0]

    @property
    def end_frame(self):
        """
        Return the end frame.
        """
        return self.buffer_frame_idx[-1]

    @property
    def start_time(self):
        """
        Return the start time.
        """
        return self.buffer_timestamps[0]

    @property
    def end_time(self):
        """
        Return the end time.
        """
        return self.buffer_timestamps[-1]

    def display_timestamp(self, ms):
        """
        Display timestamp in the format of MM:SS:MMM.
        """
        # Calculate total seconds from milliseconds
        total_seconds = ms / 1000

        # Extract minutes and seconds
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        milliseconds = ms % 1000

        return f"{int(minutes):02d}:{int(seconds):02d}:{int(milliseconds):03d}"

    def __repr__(self) -> str:
        """
        Return the string representation of the object.
        """
        return f"From {self.display_timestamp(self.start_time)} to {self.display_timestamp(self.end_time)}; Action: {self.predictions[0][0]}; Confidence: {self.predictions[0][1]:.2f}"


class MoviNet:
    def __init__(
        self,
        id: str = "a0",
        version: str = "3",
        input_width: int = 172,
        input_height: int = 172,
        number_buffer_frames: int = 50,
        model_fps: int = 5,
    ) -> None:
        """
        Initialize MoviNet model.

        Args:
            id (str, optional): MoviNet model name. Defaults to "a0".
            version (str, optional): MoviNet model version. Defaults to "3".
            input_width (int, optional): Input video width. Defaults to 172.
            input_height (int, optional): Input video height. Defaults to 172.
            number_buffer_frames (int, optional): Number of frames in the stream buffer. Defaults to 50.
            model_fps (int, optional): FPS that match with the pretrained MoviNet model. Defaults to 5.
        """
        mode = "stream"
        hub_url = f"https://tfhub.dev/tensorflow/movinet/{id}/{mode}/kinetics-600/classification/{version}"

        labels_path = tf.keras.utils.get_file(
            fname="labels.txt",
            origin="https://raw.githubusercontent.com/tensorflow/models/f8af2291cced43fc9f1d9b41ddbf772ae7b0d7d2/official/projects/movinet/files/kinetics_600_labels.txt",
        )
        labels_path = pathlib.Path(labels_path)
        lines = labels_path.read_text().splitlines()
        self.label_map = np.array([line.strip() for line in lines])

        self.model = hub.load(hub_url)

        self.input_width = input_width
        self.input_height = input_height
        self.input_channel = 3
        self.model_fps = model_fps
        self.number_buffer_frames = number_buffer_frames

    def __call__(
        self,
        file_path: str,
        start_frame: int = 0,
        end_frame: int = -1,
        top_k: int = -1,
    ) -> Generator[Tuple[List[int], List[Tuple[str, float]]], None, None]:
        """
        Process a video file and yield predictions for subclips.

        Args:
            file_path (str): Path to the video file.
            start_frame (int, optional): Starting frame for processing. Defaults to 0.
            end_frame (int, optional): Ending frame for processing. Defaults to -1.
            top_k (int, optional): Number of top predictions to return. Defaults to -1.

        Yields:
            Tuple[List[int], List[Tuple[str, float]]]: A tuple containing the buffer indices and the top_k predictions.
        """
        # Load the video
        cap = cv2.VideoCapture(file_path)

        # End frame needs to be multiple of number_buffer_frames
        if end_frame == -1:
            end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

        # Convert model_fps to frame skip
        frame_skip = int(cap.get(cv2.CAP_PROP_FPS) // self.model_fps)

        # Predefine frame index into subclip, where each subclip contains number_buffer_frames
        list_buffer_idx = []
        buffer_idx = []
        for i in range(start_frame, end_frame, frame_skip):
            # Allow adding the last buffer_idx
            if len(buffer_idx) == self.number_buffer_frames:
                list_buffer_idx.append(buffer_idx)
                buffer_idx = []
            buffer_idx.append(i)
        # Add the last buffer_idx
        if len(buffer_idx) > 0:
            list_buffer_idx.append(buffer_idx)

        # Init states
        tensor = tf.constant(
            [
                1,
                self.number_buffer_frames,
                self.input_height,
                self.input_width,
                self.input_channel,
            ],
            shape=(5,),
            dtype=tf.int32,
        )
        states = self.model.init_states(tensor)

        for buffer_frame_idx in list_buffer_idx:
            buffer_frames = []
            buffer_timestamps = []
            for idx in buffer_frame_idx:
                # Set the current frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

                # Capture current frame
                ret, frame = cap.read()
                if not ret:
                    break

                # Add frame to buffer
                frame = cv2.resize(frame, (self.input_width, self.input_height))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                buffer_frames.append(frame)
                buffer_timestamps.append(int(cap.get(cv2.CAP_PROP_POS_MSEC)))

            clip = tf.convert_to_tensor(buffer_frames, dtype=tf.float32) / 255.0
            clip = tf.expand_dims(clip, axis=0)

            logits, states = self.model({**states, "image": clip})

            yield MoviNetPrediction(
                logits, self.label_map, buffer_frame_idx, buffer_timestamps, top_k
            )

        # Release the video capture object
        cap.release()
