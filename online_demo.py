# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import argparse
import imageio.v3 as iio
import numpy as np

from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerOnlinePredictor

#use tic and toc to measure time
from time import time as tic
from time import time as toc
time0 = tic()

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

if __name__ == "__main__":

    #initialize the parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        default="./assets/apple.mp4",
        help="path to a video",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="CoTracker model parameters",
    )
    parser.add_argument("--grid_size", type=int, default=10, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.video_path):
        raise ValueError("Video file does not exist")

    if args.checkpoint is not None:
        model = CoTrackerOnlinePredictor(checkpoint=args.checkpoint)
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
    model = model.to(DEFAULT_DEVICE)

    # Test different parameters
    model.model.window_len = 20
    model.step = 10

    # try offline method
    # tmodel = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    # video = read_video_from_path(args.video_path)
    # print("Video shape: ", video.shape)
    # video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float().to(DEFAULT_DEVICE)
    # print("Video to device: ", video.size())
    # pred_tracks, pred_visibility = tmodel(video, grid_size=10)
    # print("Tracks size: ", pred_tracks.size())
    # time1 = toc()
    # atime = time1 - time0
    # print("Time elapsed before video: ", atime)
    # vis = Visualizer(save_dir='./videos', pad_value=100)
    # vis.visualize(video=video, tracks=pred_tracks, visibility=pred_visibility, filename='teaser')
    


    def _process_step(window_frames, is_first_step, grid_size = 10, grid_query_frame = 0, query_frame=None):
        video_chunk = (
            torch.tensor(
                np.stack(window_frames[-model.step * 2 :]), device=DEFAULT_DEVICE
            )
            .float()
            .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)
        return model(
            video_chunk,
            is_first_step=is_first_step,
            grid_size=grid_size,
            grid_query_frame=grid_query_frame,
            queries = query_frame
        )

    # time1 = toc()
    # atime = time1 - time0
    # print("Time elapsed before encoding: ", atime)
    # Iterating over video frames, processing one window at a time:
    window_frames = []
    is_first_step = True
    current_frame = 1
    print("Model step: ", model.step)
    for i, frame in enumerate(
        iio.imiter(
            args.video_path,
            plugin="FFMPEG",
        )
    ):
        # print(f"Processing frame {i}")
        # print(f"Frame shape: {frame.shape}")
        # print(f"Window frames: {len(window_frames)}")
        if i % model.step == 0 and i != 0:
            # The a value I set here is proved to be useless: The step is already done in _process_step
            # print("Window frame size: ", len(window_frames))
            # a = max( - model.step * 2, -len(window_frames))
            # print("a: ", a)
            # a = -(i % model.step) - model.step - 1
            pred_tracks, pred_visibility = _process_step(
                window_frames,
                # window_frames[a :],
                is_first_step,
                grid_size=args.grid_size,
                grid_query_frame=args.grid_query_frame,
            )
            # current_frame = i
            is_first_step = False
            if pred_tracks is not None:
                print("Tracks size: ", pred_tracks.size())
        window_frames.append(frame)
    # Processing the final video frames in case video length is not a multiple of model.step

    # time1 = toc()
    # atime = time1 - time0
    # print("Time elapsed after encoding whole: ", atime)
    # print("Window frames: ", len(window_frames))
    # print("Input window frames: ", len(window_frames[-(i % model.step) - model.step - 1 :]))
    
    print("Processing final frames", len(window_frames[-(i % model.step) - model.step - 1 :]))
    pred_tracks, pred_visibility = _process_step(
        # window_frames,
        window_frames[-(i % model.step) - model.step - 1 :],
        is_first_step,
        grid_size=args.grid_size,
        grid_query_frame=args.grid_query_frame,
    )
    if pred_tracks is not None:
        print("Tracks size: ", pred_tracks.size())

    print("Tracks are computed")

    # time1 = toc()
    # atime = time1 - time0
    # print("Time elapsed before video: ", atime)

    # save a video with predicted tracks
    seq_name = args.video_path.split("/")[-1]
    video = torch.tensor(np.stack(window_frames), device=DEFAULT_DEVICE).permute(
        0, 3, 1, 2
    )[None]
    vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
    vis.visualize(
        video, pred_tracks, pred_visibility, query_frame=args.grid_query_frame
    )

    # get time
    time1 = toc()
    atime = time1 - time0
    print("Time elapsed all: ", atime)
