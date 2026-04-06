"""
hccapet - Converts MP4 video to C64 PETSCII art video.
Usage: python hccapet.py input.mp4 output.mp4 [options]
"""
import argparse
import sys
import cv2
import numpy as np
from tqdm import tqdm
from converter import PetsciiConverter


def parse_args():
    p = argparse.ArgumentParser(description="hccapet - Convert MP4 to PETSCII art video")
    p.add_argument("input",  help="Input video file")
    p.add_argument("output", help="Output video file")
    p.add_argument("--cols",    type=int,   default=80,   help="Character columns (default: 80)")
    p.add_argument("--rows",    type=int,   default=50,   help="Character rows (default: 50)")
    p.add_argument("--fps",     type=float, default=0,    help="Output FPS (0 = auto from source, max 25)")
    p.add_argument("--palette", choices=["c64", "mono", "rgb", "grayscale"], default="c64", help="Color mode: c64 (default), mono, grayscale, or rgb")
    return p.parse_args()


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: cannot open '{args.input}'", file=sys.stderr)
        sys.exit(1)

    src_fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_fps = args.fps if args.fps > 0 else min(src_fps, 25.0)
    out_w   = args.cols * 8
    out_h   = args.rows * 8

    # Frame skip ratio to match output fps
    frame_skip = max(1, round(src_fps / out_fps))

    print(f"Input : {args.input}  ({src_w}x{src_h} @ {src_fps:.2f} fps, {total_frames} frames)")
    print(f"Output: {args.output}  ({out_w}x{out_h} @ {out_fps:.2f} fps)")
    print(f"Grid  : {args.cols} cols x {args.rows} rows  (frame skip: {frame_skip})")

    import tempfile, shutil, os, subprocess
    # VideoWriter fails on paths with spaces; write to temp then move
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(tmp_fd)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, out_fps, (out_w, out_h))
    if not writer.isOpened():
        # fallback: try avc1
        writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"avc1"), out_fps, (out_w, out_h))
    if not writer.isOpened():
        print(f"Error: cannot create video writer", file=sys.stderr)
        sys.exit(1)

    if args.palette == "mono":
        converter = PetsciiConverter()
        converter.palette = np.array([[0,0,0],[255,255,255]], dtype=np.uint8)
        converter._reinit_palette()
    elif args.palette == "grayscale":
        converter = PetsciiConverter()
        converter.palette = np.array([[0,0,0],[51,51,51],[119,119,119],[187,187,187],[255,255,255]], dtype=np.uint8)
        converter._reinit_palette()
    elif args.palette == "rgb":
        converter = PetsciiConverter(rgb_mode=True)
    else:
        converter = PetsciiConverter()

    frame_idx = 0
    written   = 0

    with tqdm(total=total_frames, unit="frame", desc="Converting") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                # Crop frame to fit exact grid
                crop_h = (src_h // args.rows) * args.rows
                crop_w = (src_w // args.cols) * args.cols
                frame_crop = frame[:crop_h, :crop_w]

                petscii_frame = converter.convert_frame(frame_crop, args.cols, args.rows)
                writer.write(petscii_frame)
                written += 1

            frame_idx += 1
            pbar.update(1)

    cap.release()
    writer.release()
    # Try to mux audio from source using ffmpeg
    tmp_fd2, tmp_mux = tempfile.mkstemp(suffix=".mp4")
    os.close(tmp_fd2)
    try:
        result = subprocess.run(
            ["ffmpeg", "-y",
             "-i", tmp_path,
             "-i", args.input,
             "-map", "0:v:0", "-map", "1:a:0",
             "-c:v", "copy", "-c:a", "aac",
             "-shortest", tmp_mux],
            capture_output=True
        )
        if result.returncode == 0:
            os.remove(tmp_path)
            shutil.move(tmp_mux, args.output)
            print(f"Done! Wrote {written} frames with audio -> {args.output}")
        else:
            raise RuntimeError(result.stderr.decode())
    except Exception as e:
        os.remove(tmp_mux) if os.path.exists(tmp_mux) else None
        shutil.move(tmp_path, args.output)
        print(f"Done! Wrote {written} frames (no audio: {e}) -> {args.output}")


if __name__ == "__main__":
    main()
