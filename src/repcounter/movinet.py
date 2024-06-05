import pathlib
from typing import List, Tuple, Generator

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


class MoviNet:
    def __init__(
        self,
        id: str = "a0",
        version: str = "3",
        input_width: int = 172,
        input_height: int = 172,
        input_number_frames: int = 50,
        input_fps: int = 5,
    ) -> None:
        """
        Initialize MoviNet model.

        Args:
            id (str, optional): MoviNet model name. Defaults to "a0".
            version (str, optional): MoviNet model version. Defaults to "3".
            input_width (int, optional): Input video width. Defaults to 172.
            input_height (int, optional): Input video height. Defaults to 172.
            input_number_frames (int, optional): Number of frames in the stream buffer. Defaults to 50.
            input_fps (int, optional): FPS that match with the pretrained MoviNet model. Defaults to 5.
        """
        mode = "stream"
        hub_url = f"https://tfhub.dev/tensorflow/movinet/{id}/{mode}/kinetics-600/classification/{version}"
        self.model = hub.load(hub_url)

        labels_path = tf.keras.utils.get_file(
            fname="labels.txt",
            origin="https://raw.githubusercontent.com/tensorflow/models/f8af2291cced43fc9f1d9b41ddbf772ae7b0d7d2/official/projects/movinet/files/kinetics_600_labels.txt",
        )
        labels_path = pathlib.Path(labels_path)
        lines = labels_path.read_text().splitlines()
        self.label_map = np.array([line.strip() for line in lines])

        self.input_width = input_width
        self.input_height = input_height
        self.input_channel = 3
        self.input_fps = input_fps
        self.input_number_frames = input_number_frames

    def __call__(
        self,
        file_path: str,
        start_frame: int = 0,
        end_frame: int = -1,
        top_k: int = 5,
    ) -> Generator[Tuple[List[int], List[Tuple[str, float]]], None, None]:
        """
        Process a video file and yield predictions for subclips.

        Args:
            file_path (str): Path to the video file.
            start_frame (int, optional): Starting frame for processing. Defaults to 0.
            end_frame (int, optional): Ending frame for processing. Defaults to -1.
            top_k (int, optional): Number of top predictions to return. Defaults to 5.

        Yields:
            Tuple[List[int], List[Tuple[str, float]]]: A tuple containing the buffer indices and the top_k predictions.
        """
        # Load the video
        cap = cv2.VideoCapture(file_path)

        # End frame needs to be multiple of input_number_frames
        if end_frame == -1:
            end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

        # Convert input_fps to frame skip
        frame_skip = int(cap.get(cv2.CAP_PROP_FPS) // self.input_fps)

        # Predefine frame index into subclip, where each subclip contains input_number_frames
        list_buffer_idx = []
        buffer_idx = []
        for i in range(start_frame, end_frame, frame_skip):
            # Allow adding the last buffer_idx
            if len(buffer_idx) == self.input_number_frames:
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
                self.input_number_frames,
                self.input_height,
                self.input_width,
                self.input_channel,
            ],
            shape=(5,),
            dtype=tf.int32,
        )
        states = self.model.init_states(tensor)

        for buffer_idx in list_buffer_idx:
            buffer_frames = []
            for idx in buffer_idx:
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

            clip = tf.convert_to_tensor(buffer_frames, dtype=tf.float32) / 255.0
            clip = tf.expand_dims(clip, axis=0)

            logits, states = self.model({**states, "image": clip})
            top_k_result = self._get_top_k(logits, top_k=top_k)

            yield buffer_idx, top_k_result

        # Release the video capture object
        cap.release()

    def _get_top_k(
        self, logits: tf.Tensor, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Get the top K predictions from the logits.

        Args:
            logits (tf.Tensor): Logits output from the model.
            top_k (int, optional): Number of top predictions to return. Defaults to 5.

        Returns:
            List[Tuple[str, float]]: List of tuples containing the label and probability.
        """
        probs = tf.nn.softmax(logits)

        top_predictions = tf.argsort(probs, axis=-1, direction="DESCENDING")[-1, :top_k]
        top_labels = tf.gather(self.label_map, top_predictions, axis=-1)
        top_labels = [label.decode("utf8") for label in top_labels.numpy()]

        top_probs = tf.gather(probs, top_predictions, axis=-1)
        top_probs = tf.reshape(top_probs, [-1])
        top_probs = top_probs.numpy().tolist()

        return list(zip(top_labels, top_probs))