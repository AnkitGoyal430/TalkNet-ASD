import sys, time, os, tqdm, torch, argparse, glob, subprocess, warnings, cv2, pickle, numpy
from model.faceDetector.s3fd import S3FD
from talkNet import talkNet
from scipy.io import wavfile
from scipy.interpolate import interp1d
import argparse, sys, os, subprocess
import time
from pathlib import Path

# Import necessary functions from demoTalkNet.py
from demoTalkNet import scene_detect, inference_video, track_shot, crop_video, evaluate_network

def get_faces_info(video_path, args):
    # Initialize face detector and TalkNet model
    DET = S3FD(device='cpu')
    s = talkNet()
    s.loadParameters(args.pretrainModel)
    s.eval()

    # Prepare directories
    args.videoFilePath = video_path
    args.pyaviPath = os.path.join(args.tempFolder, 'pyavi')
    args.pyframesPath = os.path.join(args.tempFolder, 'pyframes')
    args.pyworkPath = os.path.join(args.tempFolder, 'pywork')
    args.pycropPath = os.path.join(args.tempFolder, 'pycrop')
    
    for dir_path in [args.pyaviPath, args.pyframesPath, args.pyworkPath, args.pycropPath]:
        os.makedirs(dir_path, exist_ok=True)

    # Extract video frames
    command = f"ffmpeg -y -i {video_path} -qscale:v 2 -threads {args.nDataLoaderThread} -f image2 {args.pyframesPath}/%06d.jpg"
    subprocess.call(command, shell=True, stdout=None)

    # Extract audio
    args.audioFilePath = os.path.join(args.pyaviPath, 'audio.wav')
    command = f"ffmpeg -y -i {video_path} -qscale:a 0 -ac 1 -vn -threads {args.nDataLoaderThread} -ar 16000 {args.audioFilePath}"
    subprocess.call(command, shell=True, stdout=None)

    # Scene detection
    scene = scene_detect(args)

    # Face detection
    faces = inference_video(args)

    # Face tracking
    allTracks = []
    for shot in scene:
        if shot[1].frame_num - shot[0].frame_num >= args.minTrack:
            allTracks.extend(track_shot(args, faces[shot[0].frame_num:shot[1].frame_num]))

    # Face clips cropping
    vidTracks = []
    for ii, track in enumerate(allTracks):
        vidTracks.append(crop_video(args, track, os.path.join(args.pycropPath, f'{ii:05d}')))

    # Active Speaker Detection
    files = glob.glob(f"{args.pycropPath}/*.avi")
    files.sort()
    scores = evaluate_network(files, args)

    # Prepare results
    result = []
    for tidx, track in enumerate(vidTracks):
        score = scores[tidx]
        for fidx, frame in enumerate(track['track']['frame'].tolist()):
            s = numpy.mean(score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)])
            bbox = track['track']['bbox'][fidx]
            result.append({
                'frame': frame,
                'bbox': bbox.tolist(),
                'score': float(s)
            })

    return result

if __name__ == "__main__":
    # Example usage:
    # Create a dictionary with default arguments
    video_path = Path("../data/output/Shark.Tank.S06E05/scenes/Shark.Tank.S06E05-Scene-0527.mp4")
    args = {
        'videoName': str(video_path.stem),
        'tempFolder': str(video_path.parent.parent / "temp" / video_path.stem),
        'pretrainModel': "talknet/pretrain_TalkSet.model",
        'nDataLoaderThread': 10,
        'facedetScale': 0.25,
        'minTrack': 10,
        'numFailedDet': 10,
        'minFaceSize': 1,
        'cropScale': 0.40
    }

    # Convert the dictionary to a Namespace object
    args = argparse.Namespace(**args)

    # Assuming test_video_path is defined earlier in the notebook

    result = get_faces_info(str(video_path), args)

    # Display results
    for item in result:
        print(f"Frame: {item['frame']}, Bbox: {item['bbox']}, Score: {item['score']}")

