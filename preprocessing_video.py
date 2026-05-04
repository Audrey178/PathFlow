import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
import argparse





def extract_frames(video_path, output_dir):
    name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(output_dir, name)
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vsync", "0",
        f"{out_dir}/frame_%06d.png"
    ]
    subprocess.run(cmd)
    return f"[OK] {name}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from videos using ffmpeg")
    parser.add_argument('--video_dir', type=str, default="datasets/videos", help="Directory containing input videos")
    parser.add_argument('--frame_dir', type=str, default="datasets/frames", help="Directory to save extracted frames")
    parser.add_argument('--max_workers', type=int, default=4, help="Number of parallel workers for frame extraction")
    args = parser.parse_args()
    
    labels = [d for d in os.listdir(args.video_dir) if os.path.isdir(os.path.join(args.video_dir, d))]
    for label in labels:
        print(f"=============={label} processing===============")
        input_dir = os.path.join(args.video_dir, label)
        output_dir = os.path.join(args.frame_dir, label)
        os.makedirs(output_dir, exist_ok=True)
        videos = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
              if f.lower().endswith((".mp4", ".avi", ".mkv"))]

        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            for result in ex.map(extract_frames, videos, [output_dir]*len(videos)):
                print(result)
