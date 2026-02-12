
import numpy as np
import cv2
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SaliencyDetector:
    
    def __init__(self, method: str = 'spectral', device: str = 'cpu'):
        self.method = method
        self.device = device
        self.u2net_model = None
        
        if method == 'u2net':
            self._load_u2net()
        
        print(f"✓ SaliencyDetector initialized with method '{method}' on {device}")
    
    def _load_u2net(self):
        try:
            print("  Loading U2-Net model...")
            
            import torch.hub
            import os
            
            models_dir = 'models'
            os.makedirs(models_dir, exist_ok=True)
            
            model_path = os.path.join(models_dir, 'u2net.pth')
            
            if os.path.exists(model_path):
                print("  Loading U2-Net from local cache...")
                self.u2net_model = self._build_u2net()
                self.u2net_model.load_state_dict(torch.load(model_path, map_location=self.device))
            else:
                print("  Downloading U2-Net pretrained weights (176 MB)...")
                self.u2net_model = torch.hub.load('xuebinqin/U-2-Net', 'u2net', pretrained=True)
                torch.save(self.u2net_model.state_dict(), model_path)
                print("  Model cached for future use")
            
            self.u2net_model = self.u2net_model.to(self.device)
            self.u2net_model.eval()
            print("  ✓ U2-Net model loaded successfully")
            
        except Exception as e:
            print(f"  Warning: Could not load U2-Net: {e}")
            print("  Falling back to spectral method.")
            self.method = 'spectral'
            self.u2net_model = None
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        if self.method == 'spectral':
            return self._spectral_residual(image)
        elif self.method == 'fine_grained':
            return self._fine_grained(image)
        elif self.method == 'u2net' and self.u2net_model is not None:
            return self._u2net_saliency(image)
        else:
            return self._spectral_residual(image)
    
    def _spectral_residual(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        dft = np.fft.fft2(gray.astype(np.float32))
        magnitude = np.abs(dft)
        phase = np.angle(dft)
        
        log_magnitude = np.log(magnitude + 1e-8)
        smoothed = cv2.GaussianBlur(log_magnitude, (3, 3), 0)
        residual = log_magnitude - smoothed
        
        saliency_fft = np.exp(residual) * np.exp(1j * phase)
        saliency = np.abs(np.fft.ifft2(saliency_fft))
        
        saliency = cv2.GaussianBlur(saliency.astype(np.float32), (9, 9), 0)
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        return saliency
    
    def _fine_grained(self, image: np.ndarray) -> np.ndarray:
        try:
            saliency_detector = cv2.saliency.StaticSaliencyFineGrained_create()
            success, saliency_map = saliency_detector.computeSaliency(image)
            
            if success:
                saliency_map = (saliency_map - saliency_map.min()) / \
                              (saliency_map.max() - saliency_map.min() + 1e-8)
                return saliency_map
            else:
                print("  Warning: Fine-grained saliency failed, using spectral method.")
                return self._spectral_residual(image)
        except AttributeError:
            print("  Warning: Fine-grained saliency not available, using spectral method.")
            return self._spectral_residual(image)
    
    def _u2net_saliency(self, image: np.ndarray) -> np.ndarray:
        if self.u2net_model is None:
            return self._spectral_residual(image)
        
        h, w = image.shape[:2]
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image_resized = cv2.resize(image_rgb, (320, 320))
        
        image_tensor = torch.from_numpy(image_resized.transpose(2, 0, 1)).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.u2net_model(image_tensor)
            if isinstance(outputs, tuple):
                saliency_pred = outputs[0]
            else:
                saliency_pred = outputs
        
        saliency_map = saliency_pred.squeeze().cpu().numpy()
        saliency_map = cv2.resize(saliency_map, (w, h))
        
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
        
        return saliency_map
    
    def _build_u2net(self):
        try:
            import torch.hub
            model = torch.hub.load('xuebinqin/U-2-Net', 'u2net', pretrained=False)
            return model
        except:
            return None
    
    def detect_with_threshold(self, image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        saliency = self.detect(image)
        return (saliency > threshold).astype(np.uint8)
    
    def detect_multi_scale(self, image: np.ndarray, scales: list = [0.5, 1.0, 1.5]) -> np.ndarray:
        h, w = image.shape[:2]
        saliency_maps = []
        
        for scale in scales:
            new_h, new_w = int(h * scale), int(w * scale)
            scaled_img = cv2.resize(image, (new_w, new_h))
            
            sal = self.detect(scaled_img)
            
            sal = cv2.resize(sal, (w, h))
            saliency_maps.append(sal)
        
        fused_saliency = np.mean(saliency_maps, axis=0)
        
        fused_saliency = (fused_saliency - fused_saliency.min()) / \
                        (fused_saliency.max() - fused_saliency.min() + 1e-8)
        
        return fused_saliency

