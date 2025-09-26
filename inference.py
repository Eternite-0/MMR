import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import os
import argparse
from scipy.ndimage import gaussian_filter
import glob
from typing import List, Tuple

# --- Add project root to path to allow imports ---
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
# ---

# --- Imports from the project ---
from models.MMR import MMR_base
from models.MMR.utils import ForwardHook
from utils import load_backbones
from config import get_cfg

from torchvision import transforms


# --- Helper Function for Anomaly Map Calculation ---
def cal_anomaly_map(fs_list, ft_list, out_size=224):
    anomaly_map = np.zeros([fs_list[0].shape[0], out_size, out_size])
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map.squeeze(1).cpu().detach().numpy()
        anomaly_map += a_map
    return anomaly_map


# --- Main Inference Class ---
class MMR_Detector:
    def __init__(self, cfg, model_path, output_dir="detection_results"):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cfg = cfg
        self.output_dir = output_dir
        
        # Create output directory and subdirectories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        self.ok_dir = os.path.join(self.output_dir, "OK_Good")
        self.ng_dir = os.path.join(self.output_dir, "NG_Not_Good")
        os.makedirs(self.ok_dir, exist_ok=True)
        os.makedirs(self.ng_dir, exist_ok=True)

        print("Building model architecture...")
        self.teacher_model = load_backbones(cfg.TRAIN.backbone)
        self.student_model = MMR_base(cfg=cfg,
                                      scale_factors=cfg.TRAIN.MMR.scale_factors,
                                      FPN_output_dim=cfg.TRAIN.MMR.FPN_output_dim)

        print(f"Loading trained weights from: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        # Use weights_only=True to avoid security warning
        try:
            self.student_model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        except TypeError:
            # Fallback for older PyTorch versions that don't support weights_only
            self.student_model.load_state_dict(torch.load(model_path, map_location=self.device))

        self.teacher_model.to(self.device)
        self.student_model.to(self.device)
        self.teacher_model.eval()
        self.student_model.eval()

        self.teacher_outputs_dict = {}
        for extract_layer in cfg.TRAIN.MMR.layers_to_extract_from:
            forward_hook = ForwardHook(self.teacher_outputs_dict, extract_layer)
            network_layer = self.teacher_model.__dict__["_modules"][extract_layer]
            network_layer[-1].register_forward_hook(forward_hook)

        self.transform = transforms.Compose([
            transforms.Resize((cfg.DATASET.resize, cfg.DATASET.resize)),
            transforms.CenterCrop(cfg.DATASET.imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def detect(self, image_path, threshold=0.6):
        """Single image detection"""
        print(f"\n--- Detecting anomalies in {os.path.basename(image_path)} ---")

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            self.teacher_outputs_dict.clear()
            _ = self.teacher_model(image_tensor)
            teacher_features = [self.teacher_outputs_dict[key] for key in self.cfg.TRAIN.MMR.layers_to_extract_from]

            student_features_dict = self.student_model(image_tensor, mask_ratio=self.cfg.TRAIN.MMR.test_mask_ratio)
            student_features = [student_features_dict[key] for key in self.cfg.TRAIN.MMR.layers_to_extract_from]

        anomaly_map_batch = cal_anomaly_map(student_features, teacher_features, out_size=self.cfg.DATASET.imagesize)
        anomaly_map = anomaly_map_batch[0]
        smoothed_anomaly_map = gaussian_filter(anomaly_map, sigma=4)

        anomaly_score = np.max(smoothed_anomaly_map)
        decision = "NG (Not Good)" if anomaly_score > threshold else "OK (Good)"

        print(f"Max Anomaly Score: {anomaly_score:.4f}")
        print(f"Decision (Threshold={threshold}): {decision}")

        self.save_visualization(image, smoothed_anomaly_map, image_path, decision)

        return decision, anomaly_score, smoothed_anomaly_map

    def batch_detect(self, image_paths: List[str], threshold=0.6) -> List[Tuple[str, str, float, np.ndarray]]:
        """Batch detection for multiple images"""
        results = []
        print(f"\n--- Batch Detecting anomalies in {len(image_paths)} images ---")
        
        for i, image_path in enumerate(image_paths):
            print(f"Processing {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            try:
                result = self.detect(image_path, threshold)
                results.append((image_path, result[0], result[1], result[2]))
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                results.append((image_path, "ERROR", 0.0, None))
        
        return results

    def detect_folder(self, folder_path: str, threshold=0.6, extensions=None) -> List[Tuple[str, str, float, np.ndarray]]:
        """Detect anomalies in all images in a folder"""
        if extensions is None:
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        
        image_paths = []
        for extension in extensions:
            image_paths.extend(glob.glob(os.path.join(folder_path, extension)))
            image_paths.extend(glob.glob(os.path.join(folder_path, extension.upper())))
        
        print(f"Found {len(image_paths)} images in {folder_path}")
        return self.batch_detect(image_paths, threshold)

    def save_visualization(self, original_pil_image, anomaly_map, image_path, decision):
        # --- Start: Code Modification for High-Resolution Output ---

        # 1. Get the original image's dimensions
        original_width, original_height = original_pil_image.size

        # 2. Normalize the low-resolution anomaly map
        norm_anomaly_map = (anomaly_map - np.min(anomaly_map)) / (np.max(anomaly_map) - np.min(anomaly_map))
        norm_anomaly_map = (norm_anomaly_map * 255).astype(np.uint8)

        # 3. Apply a color map to the low-resolution map
        heatmap_low_res = cv2.applyColorMap(norm_anomaly_map, cv2.COLORMAP_JET)

        # 4. **CRUCIAL STEP: Upscale the heatmap to match the original image's size**
        # We use INTER_CUBIC for a smoother, higher-quality result.
        heatmap_high_res = cv2.resize(heatmap_low_res, (original_width, original_height), interpolation=cv2.INTER_CUBIC)

        # 5. Convert the original PIL image to an OpenCV image for blending
        original_cv_image = cv2.cvtColor(np.array(original_pil_image), cv2.COLOR_RGB2BGR)

        # 6. Overlay the high-resolution heatmap on the high-resolution original image
        overlay = cv2.addWeighted(original_cv_image, 0.6, heatmap_high_res, 0.4, 0)

        # --- End: Code Modification ---

        # Save the file to appropriate directory based on decision
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        
        # Determine output directory based on decision
        if "OK" in decision:
            output_path = os.path.join(self.ok_dir, f"{name}_detection_result_high_res.png")
        else:  # NG
            output_path = os.path.join(self.ng_dir, f"{name}_detection_result_high_res.png")

        cv2.imwrite(output_path, overlay)
        print(f"Saved high-resolution visualization to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MMR Anomaly Detection Inference Script")
    parser.add_argument('--image_path', type=str, help='Path to the input image for detection.')
    parser.add_argument('--folder_path', type=str, help='Path to the folder containing images for batch detection.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained .pth model file.')
    parser.add_argument('--config_path', type=str, default='method_config/AeBAD_S/MMR.yaml',
                        help='Path to the model config file.')
    parser.add_argument('--threshold', type=float, default=0.6, help='Anomaly score threshold for OK/NG decision.')
    parser.add_argument('--output_dir', type=str, default='detection_results', help='Directory to save detection results.')

    args = parser.parse_args()

    cfg = get_cfg()
    if os.path.exists(args.config_path):
        cfg.merge_from_file(args.config_path)
    else:
        raise FileNotFoundError(f"Config file not found at {args.config_path}")

    detector = MMR_Detector(cfg, args.model_path, args.output_dir)
    
    if args.folder_path:
        # Batch detection in folder
        results = detector.detect_folder(args.folder_path, args.threshold)
        print("\n--- Batch Detection Results ---")
        for image_path, decision, score, _ in results:
            print(f"{os.path.basename(image_path)}: {decision} (Score: {score:.4f})")
    elif args.image_path:
        # Single image detection
        detector.detect(args.image_path, args.threshold)
    else:
        print("Please provide either --image_path or --folder_path")