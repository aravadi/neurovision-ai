"""
pipeline.py - Standalone, production-ready inference engine for NeuroVision AI.
"""

import time
from pathlib import Path
from typing import Dict, Any, Union, Tuple
import numpy as np
import nibabel as nib
import torch
from scipy.ndimage import label as nd_label

from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    CropForegroundd,
    SpatialPadd,
    NormalizeIntensityd,
    EnsureTyped,
)


class NeuroVisionInferencePipeline:
    """
    End-to-end inference engine for 3D multi-modal brain tumor segmentation.
    """
    def __init__(
        self,
        model_weights_path: Union[str, Path],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        roi_size: Tuple[int, int, int] = (128, 128, 128),
        sw_overlap: float = 0.5,
        min_cluster_size: int = 16,
    ):
        self.device = torch.device(device)
        self.roi_size = roi_size
        self.sw_overlap = sw_overlap
        self.min_cluster_size = min_cluster_size

        # 1. Instantiate Network
        self.model = UNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=3,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
            dropout=0.1,
            bias=False,
        ).to(self.device)

        # 2. Load Weights
        weights_path = Path(model_weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found at: {weights_path}")

        checkpoint = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # 3. Sliding Window Inferer
        self.inferer = SlidingWindowInferer(
            roi_size=self.roi_size,
            sw_batch_size=2,
            overlap=self.sw_overlap,
            mode="gaussian"
        )

        # 4. Monai Preprocessing Pipeline (No labels required!)
        self.preprocessor = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            CropForegroundd(keys=["image"], source_key="image"),
            SpatialPadd(keys=["image"], spatial_size=self.roi_size, mode="constant"),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image"]),
        ])

    def _remove_small_clusters(self, binary_mask: np.ndarray) -> np.ndarray:
        """Removes isolated false positive specks smaller than min_cluster_size."""
        if np.sum(binary_mask) == 0:
            return binary_mask
        labeled_array, num_features = nd_label(binary_mask)
        if num_features == 0:
            return binary_mask
        counts = np.bincount(labeled_array.ravel())
        mask_sizes = counts > self.min_cluster_size
        mask_sizes[0] = False  # Keep background 0
        return mask_sizes[labeled_array].astype(np.uint8)

    def predict(
        self,
        flair_path: Union[str, Path],
        t1_path: Union[str, Path],
        t1ce_path: Union[str, Path],
        t2_path: Union[str, Path],
        output_nifti_path: Union[str, Path] = None,
    ) -> Dict[str, Any]:
        """
        Runs complete inference from raw NIfTI scans to multi-class segmentation.
        """
        timings = {}
        t0 = time.time()

        # 1. Check File Existence
        for p, name in [(flair_path, "FLAIR"), (t1_path, "T1"), (t1ce_path, "T1CE"), (t2_path, "T2")]:
            if not Path(p).exists():
                raise FileNotFoundError(f"{name} file not found: {p}")

        # Preserve original NIfTI geometry from T1CE (clinical reference)
        ref_img = nib.load(str(t1ce_path))
        ref_canonical = nib.as_closest_canonical(ref_img)
        ref_affine = ref_canonical.affine
        ref_header = ref_canonical.header
        ref_shape = ref_canonical.shape

        # 2. Preprocess Input Volumes
        data_dict = {"image": [str(flair_path), str(t1_path), str(t1ce_path), str(t2_path)]}
        preprocessed = self.preprocessor(data_dict)
        input_tensor = preprocessed["image"].unsqueeze(0).to(self.device)  # [1, 4, D, H, W]
        t1 = time.time()
        timings["preprocessing_s"] = round(t1 - t0, 3)

        # 3. Model Forward Pass
        with torch.no_grad():
            with torch.amp.autocast('cuda' if self.device.type == 'cuda' else 'cpu'):
                logits = self.inferer(input_tensor, self.model)
                probs = torch.sigmoid(logits)[0]  # [3, D, H, W]
                preds = (probs > 0.5).float().cpu().numpy()

        t2 = time.time()
        timings["inference_s"] = round(t2 - t1, 3)

        # 4. Post-processing & Morphological Cleanup
        # Channels: 0: WT, 1: TC, 2: ET
        wt_clean = self._remove_small_clusters(preds[0].astype(np.uint8))
        tc_clean = self._remove_small_clusters(preds[1].astype(np.uint8))
        et_clean = self._remove_small_clusters(preds[2].astype(np.uint8))

        # Reconstruct BraTS discrete multi-class mask
        # 0: Brain, 1: Necrotic Core, 2: Edema, 4: Enhancing Tumor
        seg_mask = np.zeros(wt_clean.shape, dtype=np.uint8)
        seg_mask[wt_clean == 1] = 2  # Edema
        seg_mask[tc_clean == 1] = 1  # Necrotic Core
        seg_mask[et_clean == 1] = 4  # Enhancing Tumor

        # 5. Calculate Physical Volumetrics (1mm^3 voxel = 0.001 mL)
        voxel_vol_ml = 0.001
        wt_vol_ml = float(np.sum(wt_clean) * voxel_vol_ml)
        tc_vol_ml = float(np.sum(tc_clean) * voxel_vol_ml)
        et_vol_ml = float(np.sum(et_clean) * voxel_vol_ml)
        ed_vol_ml = float(np.sum(seg_mask == 2) * voxel_vol_ml)
        nc_vol_ml = float(np.sum(seg_mask == 1) * voxel_vol_ml)

        t3 = time.time()
        timings["postprocessing_s"] = round(t3 - t2, 3)
        timings["total_pipeline_s"] = round(t3 - t0, 3)

        # 6. Save NIfTI File if requested
        if output_nifti_path is not None:
            output_path = Path(output_nifti_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            nii_out = nib.Nifti1Image(seg_mask, affine=ref_affine, header=ref_header)
            nib.save(nii_out, str(output_path))

        return {
            "seg_mask": seg_mask,                     # [D, H, W] discrete labels (0, 1, 2, 4)
            "preprocessed_image": input_tensor[0].cpu().numpy(), # [4, D, H, W]
            "affine": ref_affine,
            "volumes_ml": {
                "whole_tumor": round(wt_vol_ml, 2),
                "tumor_core": round(tc_vol_ml, 2),
                "enhancing_tumor": round(et_vol_ml, 2),
                "edema": round(ed_vol_ml, 2),
                "necrotic_core": round(nc_vol_ml, 2),
            },
            "timings": timings,
            "output_path": str(output_nifti_path) if output_nifti_path else None
        }
