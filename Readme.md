NeuroVision AI: 3D Brain Tumor Visualization and Neuro-Navigation

When a neurosurgeon prepares to resect a brain tumor, they typically have to scroll through hundreds of flat, 2D black-and-white MRI slices across different sequences (T1, T1CE, T2, and FLAIR). Mentally reconstructing the true 3D shape, depth, and spatial orientation of the tumor while planning an operation is difficult, time-consuming, and leaves room for ambiguity.

NeuroVision AI was built to solve this problem. It is an end-to-end medical imaging platform that takes raw multi-modal MRI scans, segments the tumor using deep learning, and reconstructs an interactive 3D model of the patient's brain right in the browser. Instead of guessing where the tumor boundaries lie, a surgeon or radiologist can rotate the brain 360 degrees, zoom in, peel back swelling, inspect the necrotic cavity, and extract physical measurements before entering the operating room.

WHAT THE PLATFORM DOES

Multi-Modal MRI Ingestion: Accepts the four standard co-registered MRI sequences (T1, T1-contrast, T2, and FLAIR) in NIfTI format.

Automated 3D Segmentation: Uses a 3D UNet architecture built with MONAI and PyTorch, trained on the BraTS 2020 benchmark, to produce voxel-wise segmentations for the Whole Tumor, Tumor Core, and active Enhancing Tumor in seconds.

True 3D Anatomical Reconstruction: Uses Marching Cubes and PyVista to extract continuous, smoothed polygon surface meshes, nesting the color-coded tumor structures inside a semi-transparent gray brain shell.

Interactive Neuro-Navigation: Allows clinicians to rotate, pan, and zoom through the 3D volume, scrub through synchronized 2D orthogonal slices (axial, coronal, sagittal), and adjust layer transparencies to inspect internal necrotic cores.

Surgical Morphometrics: Calculates physical volumes in milliliters, triangulated surface area, maximal 3D diameter (aligned with RANO criteria), sphericity, and stereotactic 3D centroid coordinates (RAS space in millimeters) to support surgical targeting.

TECH STACK

Deep Learning and Medical AI: Python, PyTorch, MONAI, TorchIO, Nibabel
3D Geometry and Visualization: VTK, PyVista, Three.js, WebGL
Backend and Frontend: FastAPI, Streamlit, React
Deployment and Infrastructure: Docker, AWS

TARGET USERS AND CLINICAL IMPACT

Neurosurgeons: Plan safer, more targeted craniotomies and understand tumor depth and critical structure proximity prior to surgery.
Radiologists: Accelerate volumetric scan analysis and eliminate tedious manual slice-by-slice contouring.
Clinical Researchers and Hospitals: Standardize longitudinal tumor volume tracking and assess treatment response across patient cohorts.

PROJECT GOAL

To provide clinicians with an accurate, intuitive 3D visualization and localization tool that removes guesswork from complex brain tumor boundaries, helping surgical teams make faster, safer, and better-informed decisions.