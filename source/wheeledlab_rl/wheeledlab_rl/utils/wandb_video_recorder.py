
import os
import os.path
from typing import Callable
import wandb

import gymnasium as gym
from gymnasium import error

from gymnasium.wrappers.record_video import RecordVideo
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder

class WandbVideoRecorder(VideoRecorder):
    """Overrides the close method to write videos to wandb."""
    def close(self):
        """Flush all data to disk and close any open frame encoders."""
        if not self.enabled or self._closed:
            return

        # Close the encoder
        if len(self.recorded_frames) > 0:
            try:
                from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
            except ImportError as e:
                raise error.DependencyNotInstalled(
                    "moviepy is not installed, run `pip install moviepy`"
                ) from e

            clip = ImageSequenceClip(self.recorded_frames, fps=self.frames_per_sec)
            moviepy_logger = None if self.disable_logger else "bar"
            clip.write_videofile(self.path, logger=moviepy_logger)

            # log video to wandb
            wandb.log({"Video": wandb.Video(self.path)}, commit=False)
        else:
            # No frames captured. Set metadata.
            if self.metadata is None:
                self.metadata = {}
            self.metadata["empty"] = True

        self.write_metadata()

        # Stop tracking this for autoclose
        self._closed = True


class WandbRecordVideo(RecordVideo):

    def __init__(
        self,
        env: gym.Env,
        video_folder: str,
        episode_trigger: Callable[[int], bool] = None,
        step_trigger: Callable[[int], bool] = None,
        video_length: int = 0,
        name_prefix: str = "rl-video",
        disable_logger: bool = False,
    ):
        if wandb.run.name is None:
            raise ValueError("wandb must be initialized before wrapping.")

        super().__init__(env, video_folder, episode_trigger,
                            step_trigger, video_length, name_prefix, disable_logger)

    def start_video_recorder(self):
        """Starts video recorder using :class:`video_recorder.VideoRecorder`."""
        self.close_video_recorder()

        video_name = f"{self.name_prefix}-step-{self.step_id}"
        if self.episode_trigger:
            video_name = f"{self.name_prefix}-episode-{self.episode_id}"

        base_path = os.path.join(self.video_folder, video_name)
        self.video_recorder = WandbVideoRecorder(
            env=self.env,
            base_path=base_path,
            metadata={"step_id": self.step_id, "episode_id": self.episode_id},
            disable_logger=self.disable_logger,
        )

        self.video_recorder.capture_frame()
        self.recorded_frames = 1
        self.recording = True