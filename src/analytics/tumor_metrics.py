"""
tumor_metrics.py - Computes clinical morphometrics (volume, area, diameter, sphericity, centroid).
"""

from typing import Dict, Any, Tuple
import numpy as np
import pyvista as pv
from scipy.spatial.distance import pdist


class TumorAnalyticsEngine:
    """
    Computes volumetric, geometric, and spatial localization analytics
    from 3D segmentation arrays and PyVista PolyData surface meshes.
    """
    def __init__(self, voxel_spacing_mm: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        self.voxel_spacing = voxel_spacing_mm
        self.voxel_volume_ml = (voxel_spacing_mm[0] * voxel_spacing_mm[1] * voxel_spacing_mm[2]) / 1000.0

    def compute_region_metrics(self, binary_mask: np.ndarray, mesh: pv.PolyData = None, affine: np.ndarray = None) -> Dict[str, Any]:
        """Computes Volume, Surface Area, Diameter, Sphericity, and Centroid for a single region."""
        voxel_count = int(np.sum(binary_mask))
        if voxel_count == 0:
            return {
                "voxel_count": 0, "volume_ml": 0.0, "surface_area_cm2": 0.0,
                "max_diameter_mm": 0.0, "sphericity": 0.0, "centroid_ras_mm": [0.0, 0.0, 0.0]
            }

        # 1. Volume
        volume_ml = float(voxel_count * self.voxel_volume_ml)
        volume_mm3 = volume_ml * 1000.0

        # 2. Centroid in Physical World Space (RAS mm)
        voxel_coords = np.argwhere(binary_mask)
        mean_voxel = np.mean(voxel_coords, axis=0) # [i, j, k]
        if affine is not None:
            homog = np.append(mean_voxel, 1.0)
            phys_centroid = (affine @ homog)[:3]
            centroid_ras = [round(float(c), 1) for c in phys_centroid]
        else:
            centroid_ras = [round(float(c), 1) for c in mean_voxel]

        # 3. Surface Area (from PyVista PolyData)
        if mesh is not None and mesh.n_points > 0:
            surface_area_mm2 = float(mesh.area)
            surface_area_cm2 = round(surface_area_mm2 / 100.0, 2)

            # 4. Maximum 3D Diameter (Feret Diameter)
            # Sample boundary vertices to compute max pairwise distance quickly
            pts = mesh.points
            if len(pts) > 1000:
                indices = np.random.choice(len(pts), size=1000, replace=False)
                sample_pts = pts[indices]
            else:
                sample_pts = pts
            max_diameter_mm = round(float(np.max(pdist(sample_pts))), 1) if len(sample_pts) > 1 else 0.0

            # 5. Sphericity Index Psi = (pi^(1/3) * (6V)^(2/3)) / A
            if surface_area_mm2 > 0:
                sphericity = float((np.pi ** (1.0 / 3.0)) * ((6.0 * volume_mm3) ** (2.0 / 3.0)) / surface_area_mm2)
                sphericity = round(min(max(sphericity, 0.0), 1.0), 3)
            else:
                sphericity = 0.0
        else:
            surface_area_cm2 = 0.0
            max_diameter_mm = 0.0
            sphericity = 0.0

        return {
            "voxel_count": voxel_count,
            "volume_ml": round(volume_ml, 2),
            "surface_area_cm2": surface_area_cm2,
            "max_diameter_mm": max_diameter_mm,
            "sphericity": sphericity,
            "centroid_ras_mm": centroid_ras
        }

    def determine_anatomical_localization(self, centroid_ras: Tuple[float, float, float]) -> Dict[str, str]:
        """Translates RAS coordinates (mm) into standard anatomical nomenclature."""
        x, y, z = centroid_ras
        hemisphere = "Right Cerebral Hemisphere" if x > 0 else "Left Cerebral Hemisphere"
        ap_axis = "Anterior (Frontal)" if y > 0 else "Posterior (Occipito-Parietal)"
        si_axis = "Superior (Cranial)" if z > 0 else "Inferior (Basal/Temporal)"

        return {
            "hemisphere": hemisphere,
            "orientation": f"{ap_axis}, {si_axis}",
            "formatted_location": f"{hemisphere} - {ap_axis} / {si_axis}"
        }
