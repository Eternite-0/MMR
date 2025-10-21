# tools/evaluate.py
import sys
import os

# --- Start: Path correction code ---
# Add the project's root directory (the parent of 'tools') to the Python path
# This allows the script to find the 'utils' and 'models' modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
# --- End: Path correction code ---

import logging
import pprint
import torch

from utils import setup_logging, load_config, parse_args, get_dataloaders, load_backbones
from models.MMR import MMR_base, MMR_pipeline_

LOGGER = logging.getLogger(__name__)


def main():
    """
    Main function to spawn the evaluation process.
    """
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    for path_to_config in args.cfg_files:
        # 1. Load configuration
        cfg = load_config(args, path_to_config=path_to_config)
        setup_logging(cfg)
        LOGGER.info("Running evaluation with the following configuration:")
        LOGGER.info(pprint.pformat(cfg))

        # 2. Load the test dataset
        LOGGER.info("Loading test dataset...")
        test_dataloaders = get_dataloaders(cfg=cfg, mode='test')
        if not test_dataloaders:
            LOGGER.error("No test dataloader found. Check your dataset path and configuration.")
            return

        test_dataloader = test_dataloaders[0]
        LOGGER.info(f"Test dataloader '{test_dataloader.name}' loaded with {len(test_dataloader.dataset)} samples.")

        # 3. Build the model architecture
        LOGGER.info("Building model architecture...")
        # Improved device handling
        if torch.cuda.is_available() and args.device != "-1":
            cur_device = torch.device("cuda:0")
            LOGGER.info("Using GPU for evaluation")
        else:
            cur_device = torch.device("cpu")
            LOGGER.info("Using CPU for evaluation")

        cur_model = load_backbones(cfg.TRAIN.backbone)
        mmr_base = MMR_base(cfg=cfg,
                            scale_factors=cfg.TRAIN.MMR.scale_factors,
                            FPN_output_dim=cfg.TRAIN.MMR.FPN_output_dim)

        # Move models to device
        cur_model = cur_model.to(cur_device)
        mmr_base = mmr_base.to(cur_device)

        # 4. Load the trained model weights
        model_path = os.path.join(cfg.OUTPUT_DIR, "checkpoints", "mmr_model_final.pth")

        if not os.path.exists(model_path):
            LOGGER.error(f"Model file not found at: {model_path}")
            LOGGER.error("Please make sure you have successfully trained the model and the path is correct.")
            return

        LOGGER.info(f"Loading trained weights from: {model_path}")
        # Fixed: Add weights_only=True to address security warning
        try:
            checkpoint = torch.load(model_path, map_location=cur_device, weights_only=True)
            mmr_base.load_state_dict(checkpoint)
        except Exception as e:
            LOGGER.error(f"Failed to load model checkpoint: {e}")
            # Fallback for older PyTorch versions that don't support weights_only
            try:
                checkpoint = torch.load(model_path, map_location=cur_device)
                mmr_base.load_state_dict(checkpoint)
                LOGGER.warning("Loaded model without weights_only=True due to compatibility issues")
            except Exception as e2:
                LOGGER.error(f"Failed to load model even without weights_only: {e2}")
                return
        
        # Fixed: Set model to evaluation mode
        mmr_base.eval()
        cur_model.eval()

        # 5. Run evaluation
        LOGGER.info("Starting evaluation...")

        optimizer = None

        MMR_instance = MMR_pipeline_(cur_model=cur_model,
                                     mmr_model=mmr_base,
                                     optimizer=optimizer,
                                     device=cur_device,
                                     cfg=cfg)

        try:
            auc_sample, auroc_pixel, pro_auc = MMR_instance.evaluation(test_dataloader=test_dataloader)

            LOGGER.info("--- Evaluation Complete ---")
            LOGGER.info(f"{test_dataloader.name}'s Image-Level AUROC is {auc_sample * 100:.2f}%.")
            LOGGER.info(f"{test_dataloader.name}'s Pixel-Level AUROC is {auroc_pixel * 100:.2f}%.")
            LOGGER.info(f"{test_dataloader.name}'s PRO Score is {pro_auc * 100:.2f}%.")
        except Exception as e:
            LOGGER.error(f"Error during evaluation: {e}")
            raise


if __name__ == '__main__':
    main()