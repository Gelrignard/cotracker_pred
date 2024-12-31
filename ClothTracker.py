import os
import torch
import argparse
import imageio.v3 as iio
import numpy as np

from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerOnlinePredictor

"""
This is a wrapper class for the CoTrackerOnlinePredictor class from the CoTracker library.
This class is used to track the movement of a piece of cloth from a camera feed.
The camera feed is a set of frames.
The CoTrackerOnlinePredictor class is used to track the movement of the cloth in the frames.
"""

class ClothTracker:
    def __init__(self, 
                #  video_path = "./assets/apple.mp4",
                 checkpoint = None,
                 grid_size = 10,
                 grid_query_frame = 0,
                 query_frame = None
                 ):
        # self.video_path = video_path
        self.checkpoint = checkpoint
        self.grid_size = grid_size
        self.grid_query_frame = grid_query_frame
        self.DEFAULT_DEVICE = (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        if self.checkpoint is not None:
            self.model = CoTrackerOnlinePredictor(checkpoint=self.checkpoint)
        else:
            self.model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
        self.model = self.model.to(self.DEFAULT_DEVICE)

        self.model.model.window_len = 10
        self.model.step = self.model.model.window_len // 2

        self.window_frame = []
        self.frame_num = 0
        self.is_first_step = True
        self.query_frame = query_frame
        self.pred_tracks = None
        self.pred_visibility = None

    def reset(self):
        # reset the frame input from camera feed
        self.frame_num = 0
        self.window_frame = []
        # self.query_frame = None
        self.is_first_step = True

    def _process_step(self, window_frames, is_first_step, grid_size = 10, grid_query_frame = 0, query_frame=None):
        video_chunk = (
            torch.tensor(
                np.stack(window_frames[-self.model.step * 2 :]), device=self.DEFAULT_DEVICE
            )
            .float()
            .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)
        return self.model(
            video_chunk,
            is_first_step=is_first_step,
            grid_size=grid_size,
            grid_query_frame=grid_query_frame,
            queries = query_frame
        )

    def track(self, frames, window_len = 10):
        # set window length and step
        self.model.model.window_len = window_len
        self.model.step = self.model.model.window_len // 2
        add_frame_num = len(frames)
        # print(f"add_frame_num: {add_frame_num}")
        if window_len > add_frame_num:
            if self.model.step == 0 or self.is_first_step:
                raise ValueError("Please provide a valid window length. Window length must be greater than the number of frames")
            else:
                # handle this case just like they are all resiudual frames
                cat_frame = np.concatenate([self.window_frame, frames], axis = 0)
                self.window_frame = cat_frame
                self.frame_num += add_frame_num
                pred_tracks, pred_visibility = self._process_step(
                    self.window_frame[-add_frame_num - self.model.step - 1 :],
                    is_first_step = self.is_first_step,
                    grid_size = self.grid_size,
                    grid_query_frame = self.grid_query_frame,
                    query_frame = self.query_frame
                )
                self.pred_tracks = pred_tracks
                self.pred_visibility = pred_visibility
                return pred_tracks, pred_visibility
            
        new_query_frame = self.query_frame

        for i, frame in enumerate(frames):
            # frame = torch.tensor(frame)
            # print(f"Frame {i} shape: {frame.size()}")
            # process step
            if i % self.model.step == 0 and i != 0:
                # !!! MUST ADD i!=0 condition to avoid the first step, OR THE SHAPE OF THE PREDICTED TRACKS WILL BE WRONG !!!
                if self.is_first_step:
                    pred_tracks, pred_visibility = self._process_step(
                        self.window_frame,
                        is_first_step = self.is_first_step,
                        grid_size = self.grid_size,
                        grid_query_frame = self.grid_query_frame,
                        query_frame = new_query_frame
                    )
                    self.is_first_step = False
                elif not self.is_first_step:
                    pred_tracks, pred_visibility = self._process_step(
                        self.window_frame,
                        is_first_step = self.is_first_step,
                        grid_size = self.grid_size,
                        grid_query_frame = self.grid_query_frame,
                        query_frame = new_query_frame
                    )
                    print(f"trk {i} shape: {pred_tracks.size()}")
                
            # add frames to the window
            self.window_frame.append(frame)
            self.frame_num += 1
        # process the last step
        pred_tracks, pred_visibility = self._process_step(
            self.window_frame[-(i % self.model.step) - self.model.step - 1 :],
            is_first_step = self.is_first_step,
            grid_size = self.grid_size,
            grid_query_frame = self.grid_query_frame,
            query_frame = new_query_frame
        )
        # print(f"Frame {self.frame_num}: {pred_tracks}")
        self.pred_tracks = pred_tracks
        self.pred_visibility = pred_visibility
        return pred_tracks, pred_visibility
    
    def test_with_video_input(self, video_path = "./assets/apple.mp4"):
        test_frame_flow_size = 15
        my_video = read_video_from_path(video_path)

        for i, frame in enumerate(my_video):
            if i % test_frame_flow_size == 0 and i != 0:
                query = torch.tensor([
                    [0, 400., 350.],
                    [1, 300., 300.]
                    ])
                self.modify_query_frame(query)
                frames = my_video[i - test_frame_flow_size : i]
                pred_tracks, pred_visibility = self.track(frames)
                print(f"Frame {i} shape: {pred_tracks.size()}")
                res = i
        # add final frames
        frames = my_video[res+1:]
        pred_tracks, pred_visibility = self.track(frames)
        print(f"Frame {i} shape: {pred_tracks.size()}")
        latest_points = self.get_latest_points_position()
        print(f"Latest points: {latest_points}")
        # visualize the result
        # only visualize since the last modification of the query frame
        video = torch.tensor(np.stack(self.window_frame), device=self.DEFAULT_DEVICE).permute(
            0, 3, 1, 2
        )[None]
        vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
        vis.visualize(
            video, pred_tracks, pred_visibility, query_frame=self.grid_query_frame
        )

    def modify_query_frame(self, query_frame):
        # Query is a n*3 tensor where n is the number of query points. The first column is the frame number of the input figure stream.
        # Therefore, we need to add the frame number to the query points.
        new_query_frame = query_frame
        if new_query_frame is not None:
            new_query_frame[:,0] += self.frame_num
            new_query_frame = new_query_frame[None]
        self.query_frame = new_query_frame
        # by reseting, the query frame will be updated
        self.reset()

    def get_latest_points_position(self):
        # the latest key point position is the last point as shape of [p1([x,y]), p2, p3, ...]
        return self.pred_tracks[0,-1]


if __name__ == "__main__":
    tracker = ClothTracker()
    queries = torch.tensor([
        [0., 400., 350.],  # point tracked from the first frame
        [5, 600., 500.], # frame number 10
    ])
    # tracker.modify_query_frame(queries)
    tracker.test_with_video_input()

