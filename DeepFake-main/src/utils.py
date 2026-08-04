import cv2
from facenet_pytorch import MTCNN  # pip install facenet-pytorch
import ffmpeg
from pathlib import Path
import torch

def detect_faces_mtcnn(image):
    """Face detection with MTCNN"""
    mtcnn = MTCNN(image_size=224, margin=20, min_face_size=20, device='cuda')
    boxes, probs = mtcnn.detect(image)
    return boxes, probs

def extract_video_frames(video_path, output_dir, fps=8):
    """Extract frames using OpenCV"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(video_fps / fps))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if frame_count % frame_interval == 0:
            cv2.imwrite(str(output_dir / f"frame_{frame_count:06d}.jpg"), frame)
        frame_count += 1
    cap.release()
    return frame_count

def load_model(model_class, model_path, device='cuda'):
    """Generic model loader"""
    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model
