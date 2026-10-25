"""
mesh_generator.py - Extracts, smooths, transforms, and exports 3D medical meshes......
"""

from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import nibabel as nib
import pyvista as pv
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes


class BrainMeshGenerator:
    """
    Engine to convert 3D NIfTI brain and tumor volumes into smoothed,
    physically-calibrated PyVista PolyData surfaces.
    """
    def __init__(self, smoothing_sigma: float = 1.0, decimate_fraction: float = 0.5):
        self.smoothing_sigma = smoothing_sigma
        self.decimate_fraction = decimate_fraction

    def _extract_surface_mesh(
        self,
        binary_volume: np.ndarray,
        affine: np.ndarray,
        step_size: int = 1,
        smooth_iterations: int = 30
    ) -> Optional[pv.PolyData]:
        """
        Extracts an anti-aliased, smoothed, affine-transformed PyVista PolyData mesh.
        """
        if np.sum(binary_volume) == 0:
            return None

        # 1. 3D Gaussian Pre-filter to eliminate discrete stair-stepping
        continuous_field = gaussian_filter(binary_volume.astype(np.float32), sigma=self.smoothing_sigma)

        # 2. Marching Cubes Isosurface Extraction at 0.5 threshold
        verts, faces, normals, _ = marching_cubes(continuous_field, level=0.5, step_size=step_size)

        # 3. Transform Vertices from Voxel Coordinates -> Real-World Physical Space (mm)
        # Homogeneous coordinates [N, 4]
        homog_verts = np.hstack([verts, np.ones((verts.shape[0], 1))])
        physical_verts = (affine @ homog_verts.T).T[:, :3]

        # 4. Construct PyVista / VTK PolyData Object
        # VTK requires face array format: [3, v0, v1, v2, 3, v3, v4, v5, ...]
        vtk_faces = np.empty((faces.shape[0], 4), dtype=np.int64)
        vtk_faces[:, 0] = 3
        vtk_faces[:, 1:] = faces
        mesh = pv.PolyData(physical_verts, vtk_faces.ravel())

        # 5. Laplacian Mesh Smoothing
        if smooth_iterations > 0:
            mesh = mesh.smooth(n_iter=smooth_iterations, relaxation_factor=0.15)

        # 6. Quadric Decimation (Reduces triangle count while preserving curvature)
        if 0.0 < self.decimate_fraction < 1.0:
            mesh = mesh.decimate(target_reduction=self.decimate_fraction)

        # Recompute surface normal vectors for optimal specular lighting
        mesh.compute_normals(inplace=True)
        return mesh

    def generate_all_structures(
        self,
        brain_volume: np.ndarray,
        seg_mask: np.ndarray,
        affine: np.ndarray,
        brain_threshold: float = 0.1
    ) -> Dict[str, Optional[pv.PolyData]]:
        """
        Extracts Whole Brain, Edema, Tumor Core, and Enhancing Tumor meshes.
        """
        meshes = {}

        # 1. Whole Brain Anatomy Shell (Binary brain tissue from MRI)
        brain_binary = brain_volume > brain_threshold
        meshes["brain"] = self._extract_surface_mesh(
            brain_binary, affine, step_size=2, smooth_iterations=40
        )

        # 2. Peritumoral Edema (BraTS Label 2)
        edema_binary = (seg_mask == 2)
        meshes["edema"] = self._extract_surface_mesh(
            edema_binary, affine, step_size=1, smooth_iterations=25
        )

        # 3. Tumor Core (BraTS Label 1)
        core_binary = (seg_mask == 1)
        meshes["core"] = self._extract_surface_mesh(
            core_binary, affine, step_size=1, smooth_iterations=25
        )

        # 4. Enhancing Active Tumor (BraTS Label 4)
        enhancing_binary = (seg_mask == 4)
        meshes["enhancing"] = self._extract_surface_mesh(
            enhancing_binary, affine, step_size=1, smooth_iterations=25
        )

        return meshes

    def export_meshes(
        self,
        meshes: Dict[str, Optional[pv.PolyData]],
        output_dir: Path,
        file_prefix: str = "case",
        format: str = "obj"
    ) -> Dict[str, str]:
        """
        Exports extracted PolyData structures to standard 3D files (.obj or .stl).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = {}

        for name, mesh in meshes.items():
            if mesh is not None and mesh.n_points > 0:
                file_path = output_dir / f"{file_prefix}_{name}.{format}"
                mesh.save(str(file_path))
                saved_paths[name] = str(file_path)

        return saved_paths
