#!/usr/bin/env python3
"""
Multi-Modal Deepfake Detection System - Main CLI
Usage: python main.py --text "fake text" --image fake.jpg --video fake.mp4
"""

import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Multi-Modal Deepfake Detector")
    parser.add_argument('--text', type=str, help="Text/caption to analyze")
    parser.add_argument('--image', type=str, help="Image path")
    parser.add_argument('--video', type=str, help="Video path")
    parser.add_argument('--explain', action='store_true', help="Generate explanations")
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/predictions/latest_prediction.json',
        help='JSON file path for prediction output',
    )
    
    args = parser.parse_args()
    
    if not (args.text or args.image or args.video):
        parser.error("Please provide at least one modality: --text, --image, or --video")

    from src.explainability import ExplainabilityEngine
    from src.fusion_engine import FusionEngine

    fusion = FusionEngine()
    explainer = ExplainabilityEngine()

    text_det = None
    img_det = None
    vid_det = None

    scores = {}

    if args.text:
        from src.text_detector import TextDetector
        text_det = TextDetector()
        scores['text'] = text_det.predict([args.text])[0]
        print(f"📄 Text Score: {scores['text']:.3f}")

    if args.image:
        from src.image_detector import ImageDetector
        if not Path(args.image).exists():
            raise FileNotFoundError(f"Image not found: {args.image}")
        img_det = ImageDetector()
        scores['image'] = img_det.predict([args.image])[0]
        print(f"🖼️ Image Score: {scores['image']:.3f}")

    if args.video:
        from src.video_detector import VideoDetector
        if not Path(args.video).exists():
            raise FileNotFoundError(f"Video not found: {args.video}")
        vid_det = VideoDetector()
        scores['video'], n_frames = vid_det.predict(args.video)
        print(f"🎥 Video Score: {scores['video']:.3f} ({n_frames} frames)")

    # Fusion
    final_score = fusion.fuse_scores(
        scores.get('text', 0.5),
        scores.get('image', 0.5),
        scores.get('video', 0.5)
    )
    
    label = "🟢 REAL" if final_score < 0.5 else "🔴 FAKE"
    print(f"\n🎯 FINAL PREDICTION: {label}")
    print(f"   Fake Probability: {final_score:.3f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_payload = {
        "scores": {key: float(value) for key, value in scores.items()},
        "final_score": float(final_score),
        "label": "REAL" if final_score < 0.5 else "FAKE",
        "threshold": 0.5,
    }
    output_path.write_text(json.dumps(output_payload, indent=2), encoding='utf-8')
    print(f"💾 Saved prediction JSON: {output_path}")

    if args.explain:
        image_heatmap = None
        if img_det and args.image:
            image_heatmap = img_det.explain_gradcam(args.image)

        explainer.explanations = {
            'text': float(scores.get('text', 0.0)),
            'image': float(scores.get('image', 0.0)),
            'video': float(scores.get('video', 0.0)),
        }
        explainer.generate_report(None, image_heatmap, None, final_score)

if __name__ == "__main__":
    main()
