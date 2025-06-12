import os
from typing import Callable

import av
import gymnasium as gym
import wandb
from gymnasium import logger
from gymnasium.core import ActType, ObsType
from gymnasium.wrappers.rendering import RecordVideo


class CustomRecordVideo(RecordVideo):
    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        video_folder: str,
        episode_trigger: Callable[[int], bool] | None = None,
        step_trigger: Callable[[int], bool] | None = None,
        video_length: int = 0,
        name_prefix: str = "rl-video",
        fps: int | None = None,
        disable_logger: bool = True,
        enable_wandb: bool = True,
        video_resolution: tuple[int, int] = (1280, 720),
        video_crf: int = 30,
    ):
        if enable_wandb and wandb.run.name is None:
            raise ValueError("wandb must be initialized before wrapping.")

        super().__init__(
            env,
            video_folder,
            episode_trigger,
            step_trigger,
            video_length,
            name_prefix,
            fps,
            disable_logger,
        )
        self.enable_wandb = enable_wandb
        self.video_resolution = video_resolution
        self.video_crf = video_crf

    def stop_recording(self):
        """Stop current recording and saves the video."""
        assert self.recording, "stop_recording was called, but no recording was started"

        if len(self.recorded_frames) == 0:
            logger.warn("Ignored saving a video as there were zero frames to save.")
        else:
            path = os.path.join(self.video_folder, f"{self._video_name}.mp4")
            output = av.open(path, "w")
            output_stream = output.add_stream(
                "libx264",
                rate=round(self.frames_per_sec),
            )
            output_stream.width, output_stream.height = self.video_resolution
            output_stream.pix_fmt = "yuv420p"
            output_stream.options = {"crf": str(self.video_crf), "preset": "veryslow"}
            for frame in self.recorded_frames:
                video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
                video_frame = video_frame.reformat(
                    width=self.video_resolution[0], height=self.video_resolution[1]
                )
                packet = output_stream.encode(video_frame)
                output.mux(packet)
            packet = output_stream.encode()
            output.mux(packet)
            output.close()
            if self.enable_wandb:
                wandb.log({"Video": wandb.Video(path)}, commit=False)

        self.recorded_frames = []
        self.recording = False
        self._video_name = None
