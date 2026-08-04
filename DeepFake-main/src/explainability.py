import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pathlib import Path

class ExplainabilityEngine:
    """Unified explainability generator"""
    
    def __init__(self):
        self.explanations = {}
    
    def generate_report(self, text_shap, image_heatmap, video_temporal, fused_score):
        """Create comprehensive explanation report"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Text SHAP
        if text_shap is not None:
            axes[0,0].imshow(np.array(text_shap), aspect='auto')
            axes[0,0].set_title(f'Text SHAP (Score: {self.explanations.get("text", 0):.2f})')
        else:
            axes[0,0].text(0.5, 0.5, "No text explanation", ha='center', va='center')
            axes[0,0].set_title('Text SHAP')
            axes[0,0].axis('off')
        
        # Image GradCAM
        if image_heatmap is not None:
            axes[0,1].imshow(np.array(image_heatmap))
            axes[0,1].set_title(f'Image GradCAM (Score: {self.explanations.get("image", 0):.2f})')
        else:
            axes[0,1].text(0.5, 0.5, "No image explanation", ha='center', va='center')
            axes[0,1].set_title('Image GradCAM')
            axes[0,1].axis('off')
        
        # Video Temporal
        if video_temporal is not None and len(video_temporal) > 0:
            axes[1,0].plot(video_temporal)
            axes[1,0].set_title(f'Video Temporal (Score: {self.explanations.get("video", 0):.2f})')
        else:
            axes[1,0].text(0.5, 0.5, "No video explanation", ha='center', va='center')
            axes[1,0].set_title('Video Temporal')
            axes[1,0].axis('off')
        
        # Final Prediction
        axes[1,1].text(0.5, 0.5, f'FINAL PREDICTION\nFake Probability: {fused_score:.3f}', 
                      ha='center', va='center', fontsize=20, 
                      bbox=dict(boxstyle="round,pad=1", facecolor="lightblue"))
        axes[1,1].axis('off')
        
        plt.suptitle('🕵️ Multi-Modal Deepfake Explanation', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_dir / 'complete_explanation.png', dpi=300, bbox_inches='tight')
        plt.show()
        return fig
