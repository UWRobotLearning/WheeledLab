
import os
import os.path
import av
from typing import Callable
import wandb

import gymnasium as gym

from gymnasium.wrappers.record_video import RecordVideo
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder

class CustomVideoRecorder(VideoRecorder):
    """Overrides the close method to write videos to wandb."""
    wandb = None

    def enable_wandb(self, wandb):
        self.wandb = wandb

    def close(self):
        """Flush all data to disk and close any open frame encoders."""
        if not self.enabled or self._closed:
            return

        # Close the encoder
        if len(self.recorded_frames) > 0:
            H, W = self.recorded_frames[0].shape[:2]
            output = av.open(self.path, 'w')
            output_stream = output.add_stream('libx264', rate=round(self.frames_per_sec))
            output_stream.width = W
            output_stream.height = H
            output_stream.pix_fmt = "yuv444p"
            output_stream.options = {"crf": "0", "preset": "veryslow"}
            for frame in self.recorded_frames:
                packet = output_stream.encode(av.VideoFrame.from_ndarray(frame, format='rgb24'))
                output.mux(packet)
            packet = output_stream.encode(None)
            output.mux(packet)
            output.close()

            # log video to wandb
            if self.wandb:
                wandb.log({"Video": wandb.Video(self.path)}, commit=False)
            self.recorded_frames, self.recorded_history = [], []
        else:
            # No frames captured. Set metadata.
            if self.metadata is None:
                self.metadata = {}
            self.metadata["empty"] = True

        self.write_metadata()

        # Stop tracking this for autoclose
        self._closed = True


class CustomRecordVideo(RecordVideo):

    def __init__(
        self,
        env: gym.Env,
        video_folder: str,
        episode_trigger: Callable[[int], bool] = None,
        step_trigger: Callable[[int], bool] = None,
        video_length: int = 0,
        name_prefix: str = "rl-video",
        disable_logger: bool = False,
        enable_wandb: bool = True,
    ):
        if enable_wandb and wandb.run.name is None:
            raise ValueError("wandb must be initialized before wrapping.")

        super().__init__(env, video_folder, episode_trigger,
                            step_trigger, video_length, name_prefix, disable_logger)
        self.enable_wandb = enable_wandb

    def start_video_recorder(self):
        """Starts video recorder using :class:`video_recorder.VideoRecorder`."""
        self.close_video_recorder()

        video_name = f"{self.name_prefix}-step-{self.step_id}"
        if self.episode_trigger:
            video_name = f"{self.name_prefix}-episode-{self.episode_id}"

        base_path = os.path.join(self.video_folder, video_name)
        self.video_recorder = CustomVideoRecorder(
            env=self.env,
            base_path=base_path,
            metadata={"step_id": self.step_id, "episode_id": self.episode_id},
            disable_logger=self.disable_logger,
        )
        self.video_recorder.enable_wandb(self.enable_wandb)

        self.video_recorder.capture_frame()
        self.recorded_frames = 1
        self.recording = True