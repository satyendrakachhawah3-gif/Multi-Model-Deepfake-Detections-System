import numpy as np

class FusionEngine:
    """Decision-level fusion with learned/configurable weights"""
    
    def __init__(self, weights=None):
        if weights is None:
            self.weights = np.array([0.3, 0.3, 0.4])  # text, image, video
        else:
            self.weights = np.array(weights)
    
    def fuse_scores(self, text_score, image_score, video_score):
        """Weighted average fusion"""
        if text_score is None: text_score = 0.5
        if image_score is None: image_score = 0.5
        if video_score is None: video_score = 0.5
        
        fused_score = np.dot([text_score, image_score, video_score], self.weights)
        return min(max(fused_score, 0.0), 1.0)
    
    def predict_label(self, fused_score, threshold=0.5):
        return 1 if fused_score > threshold else 0  # 1=fake
    
    def calibrate_weights(self, text_acc, img_acc, vid_acc):
        """Dynamic weight adjustment based on modality performance"""
        accuracies = np.array([text_acc, img_acc, vid_acc])
        self.weights = accuracies / accuracies.sum()
        self.weights /= self.weights.sum()  # Normalize
