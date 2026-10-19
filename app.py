"""
app.py - NeuroVision AI: 3D Brain Tumor Visualization and Neuro-Navigation Prototype
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import nibabel as nib
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, binary_fill_holes, binary_closing
from skimage.measure import marching_cubes
import pyvista as pv
import torch

# Optimize PyTorch CPU execution for Streamlit Cloud
if not torch.cuda.is_available():
    try:
        torch.set_num_threads(2)
    except Exception:
        pass

# Append local src to path
sys.path.append(str(Path(__file__).parent))

from src.inference.pipeline import NeuroVisionInferencePipeline
from src.analytics.tumor_metrics import TumorAnalyticsEngine

# ==============================================================================
# PAGE CONFIG & CYBER-CLINICAL THEME
# ==============================================================================
st.set_page_config(
    page_title="NeuroVision AI | 3D Neuro-Navigation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0a0a0e; color: #e0e6ed; }
    .stMetric { background-color: #141824; border: 1px solid #1f283e; border-radius: 8px; padding: 12px; }
    h1, h2, h3 { color: #00e5ff !important; font-family: monospace; }
    .stButton>button { background-color: #00e5ff; color: #0a0a0e; font-weight: bold; border-radius: 6px; border: none; width: 100%; }
    .stButton>button:hover { background-color: #00ffaa; color: #000; }
</style>
""", unsafe_allow_html=True)

# Initialize Global Session State
if "pipeline_results" not in st.session_state:
    st.session_state["pipeline_results"] = None
if "analytics" not in st.session_state:
    st.session_state["analytics"] = None
if "case_name" not in st.session_state:
    st.session_state["case_name"] = None
if "meshes" not in st.session_state:
    st.session_state["meshes"] = None
if "obj_strings" not in st.session_state:
    st.session_state["obj_strings"] = None

DATA_DIR = Path(__file__).resolve().parent / "demo_data"
CHECKPOINT_PATH = Path(__file__).resolve().parent / "models" / "best_metric_model.pth"

# Cache Pipeline Loading
@st.cache_resource(show_spinner=False)
def load_ai_pipeline():
    return NeuroVisionInferencePipeline(model_weights_path=CHECKPOINT_PATH)

# Fast 2D Slice Renderer (Instant NumPy + Alpha Overlay)
def fast_slice_render(underlay_slice, mask_slice=None, show_mask=True):
    s_min = float(np.min(underlay_slice))
    s_max = float(np.max(underlay_slice))
    if s_max > s_min:
        norm = ((underlay_slice - s_min) / (s_max - s_min) * 255.0).clip(0, 255).astype(np.uint8)
    else:
        norm = np.zeros_like(underlay_slice, dtype=np.uint8)

    # 3-channel grayscale RGB
    rgb = np.stack([norm, norm, norm], axis=-1)

    # Blend tumor segmentation labels
    if show_mask and mask_slice is not None:
        alpha = 0.70
        # Label 2: Yellow (Edema)
        m2 = (mask_slice == 2)
        if np.any(m2):
            rgb[m2] = (rgb[m2] * (1.0 - alpha) + np.array([250, 230, 25]) * alpha).astype(np.uint8)
        # Label 1: Red (Necrotic Core)
        m1 = (mask_slice == 1)
        if np.any(m1):
            rgb[m1] = (rgb[m1] * (1.0 - alpha) + np.array([242, 38, 38]) * alpha).astype(np.uint8)
        # Label 4: Blue (Enhancing Tumor)
        m4 = (mask_slice == 4)
        if np.any(m4):
            rgb[m4] = (rgb[m4] * (1.0 - alpha) + np.array([16, 128, 255]) * alpha).astype(np.uint8)

    return rgb

# Helper: Extract 3D Surface
def extract_surface_mesh(binary_vol, affine, sigma=1.0, step=1, smooth_iter=20, decimate=0.4):
    if np.sum(binary_vol) < 30:
        return None
    field = gaussian_filter(binary_vol.astype(np.float32), sigma=sigma)
    min_v, max_v = float(np.min(field)), float(np.max(field))

    # Safe level check
    if not (min_v < 0.5 < max_v):
        if max_v > 0.15:
            level = max_v * 0.5
        else:
            return None
    else:
        level = 0.5

    try:
        verts, faces, _, _ = marching_cubes(field, level=level, step_size=step)
    except Exception:
        return None

    homog = np.hstack([verts, np.ones((verts.shape[0], 1))])
    phys_verts = (affine @ homog.T).T[:, :3]
    vtk_faces = np.empty((faces.shape[0], 4), dtype=np.int64)
    vtk_faces[:, 0] = 3
    vtk_faces[:, 1:] = faces
    mesh = pv.PolyData(phys_verts, vtk_faces.ravel())
    if smooth_iter > 0:
        mesh = mesh.smooth(n_iter=smooth_iter, relaxation_factor=0.15)
    if 0.0 < decimate < 1.0:
        mesh = mesh.decimate(target_reduction=decimate)
    mesh.compute_normals(inplace=True)
    return mesh

# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================
st.sidebar.image("https://img.icons8.com/fluency/96/brain.png", width=70)
st.sidebar.title("NeuroVision AI")
st.sidebar.caption("3D Surgical Neuro-Navigation Prototype")

page = st.sidebar.radio(
    "Navigation",
    ["📁 1. Upload & Select Scans",
     "🧠 2. AI Segmentation & 2D Slices",
     "🎮 3. Interactive 3D Viewer",
     "📊 4. Quantitative Analytics",
     "📑 5. Surgical Report Dossier"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Modalities Active:**")
st.sidebar.markdown("`FLAIR` | `T1` | `T1CE` | `T2`")
st.sidebar.markdown("**Color Key:**")
st.sidebar.markdown("**Yellow**: Edema\n\n**Red**: Tumor Core\n\n**Blue**: Enhancing Tumor")

# ==============================================================================
# PAGE 1: UPLOAD & SELECT SCANS
# ==============================================================================
if page == "📁 1. Upload & Select Scans":
    st.title("Patient MRI Ingestion")
    st.write("Upload 4 co-registered NIfTI (.nii) MRI sequences or select a pre-loaded clinical case.")

    tab_demo, tab_upload = st.tabs(["Quick Demo Patients", "Upload Custom .nii Scans"])

    with tab_demo:
        available_cases = sorted([p.name for p in DATA_DIR.rglob("BraTS20_Training_*") if p.is_dir()])
        default_idx = available_cases.index("BraTS20_Training_047") if "BraTS20_Training_047" in available_cases else 0

        selected_case = st.selectbox("Select Benchmark Patient:", available_cases, index=default_idx)

        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("Load Patient Scans"):
                case_dir = list(DATA_DIR.rglob(selected_case))[0]
                st.session_state["selected_dir"] = case_dir
                st.session_state["case_name"] = selected_case
                st.success(f"Loaded {selected_case} successfully. Proceed to Page 2.")

        with col2:
            st.info("""
            **Recommended Cases:**
            - `BraTS20_Training_047`: Large Glioblastoma with prominent Red Core (30.2 mL), Yellow Edema (70.4 mL), and Blue Enhancing (32.0 mL).
            - `BraTS20_Training_001`: Classic frontal convexity glioma.
            """)

    with tab_upload:
        st.write("Upload all 4 co-registered NIfTI sequences for a new patient:")
        c_up1, c_up2 = st.columns(2)
        with c_up1:
            u_flair = st.file_uploader("1. Upload FLAIR (.nii)", type=["nii", "gz"], key="up_flair")
            u_t1    = st.file_uploader("2. Upload T1 (.nii)", type=["nii", "gz"], key="up_t1")
        with c_up2:
            u_t1ce  = st.file_uploader("3. Upload T1CE (.nii)", type=["nii", "gz"], key="up_t1ce")
            u_t2    = st.file_uploader("4. Upload T2 (.nii)", type=["nii", "gz"], key="up_t2")

        patient_name_input = st.text_input("Patient Identifier:", value="Custom_Patient_001")

        if u_flair and u_t1 and u_t1ce and u_t2:
            if st.button("Ingest & Process Uploaded Scans"):
                with st.spinner("Saving NIfTI scans to disk and verifying geometry..."):
                    upload_dir = Path(f"data/uploads/{patient_name_input}")
                    upload_dir.mkdir(parents=True, exist_ok=True)

                    with open(upload_dir / f"{patient_name_input}_flair.nii", "wb") as f: f.write(u_flair.getbuffer())
                    with open(upload_dir / f"{patient_name_input}_t1.nii", "wb") as f: f.write(u_t1.getbuffer())
                    with open(upload_dir / f"{patient_name_input}_t1ce.nii", "wb") as f: f.write(u_t1ce.getbuffer())
                    with open(upload_dir / f"{patient_name_input}_t2.nii", "wb") as f: f.write(u_t2.getbuffer())

                    st.session_state["selected_dir"] = upload_dir
                    st.session_state["case_name"] = patient_name_input
                    st.session_state["pipeline_results"] = None
                    st.session_state["meshes"] = None
                    st.session_state["obj_strings"] = None
                    st.session_state["analytics"] = None

                    st.success(f"All 4 sequences for '{patient_name_input}' saved. Now click '2. AI Segmentation' in the sidebar.")

# ==============================================================================
# PAGE 2: AI SEGMENTATION & 2D SLICER
# ==============================================================================
elif page == "🧠 2. AI Segmentation & 2D Slices":
    st.title("AI Segmentation & Orthogonal Slicer")

    if st.session_state.get("case_name") is None:
        st.warning("Please select or upload a patient on Page 1 first.")
        st.stop()

    case_name = st.session_state["case_name"]
    case_dir = st.session_state["selected_dir"]

    # Trigger Inference Button
    if st.session_state["pipeline_results"] is None or st.session_state["case_name"] != case_name:
        if st.button(f"Execute 3D UNet Segmentation for {case_name}"):
            with st.spinner("Running 3D sliding window inference..."):
                pipeline = load_ai_pipeline()
                flair = list(case_dir.glob("*flair.nii*"))[0]
                t1 = list(case_dir.glob("*t1.nii*"))[0]
                t1ce = list(case_dir.glob("*t1ce.nii*"))[0]
                t2 = list(case_dir.glob("*t2.nii*"))[0]

                t0 = time.time()
                results = pipeline.predict(flair, t1, t1ce, t2)
                elapsed = time.time() - t0

                st.session_state["pipeline_results"] = results
                st.session_state["inference_time"] = elapsed
                st.success(f"Inference completed in {elapsed:.2f} seconds.")
                st.rerun()

    # Fast Interactive 2D Slicer
    if st.session_state["pipeline_results"] is not None:
        results = st.session_state["pipeline_results"]
        seg_mask = results["seg_mask"]
        t1ce_prep = results["preprocessed_image"][2]
        flair_prep = results["preprocessed_image"][0]

        st.subheader("Orthogonal Multi-Planar Slicer")
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            underlay_type = st.radio("Underlay Modality:", ["T1CE (Contrast)", "FLAIR (Water/Edema)"], horizontal=True)
            underlay = t1ce_prep if "T1CE" in underlay_type else flair_prep
        with c2:
            show_tumor_mask = st.checkbox("Overlay AI Tumor Segmentation", value=True)
        with c3:
            st.metric("Pipeline Latency", f"{st.session_state.get('inference_time', 2.1):.2f}s")

        # Slider coordinates
        d_z, d_y, d_x = seg_mask.shape
        tumor_idx = np.where(seg_mask > 0)
        default_z = int(np.mean(tumor_idx[0])) if len(tumor_idx[0]) > 0 else d_z // 2
        default_y = int(np.mean(tumor_idx[1])) if len(tumor_idx[1]) > 0 else d_y // 2
        default_x = int(np.mean(tumor_idx[2])) if len(tumor_idx[2]) > 0 else d_x // 2

        col_z, col_y, col_x = st.columns(3)
        with col_z:
            slice_z = st.slider("Axial Cut (Z):", 0, d_z - 1, default_z)
        with col_y:
            slice_y = st.slider("Coronal Cut (Y):", 0, d_y - 1, default_y)
        with col_x:
            slice_x = st.slider("Sagittal Cut (X):", 0, d_x - 1, default_x)

        # Fast Instant Slice Rendering (Direct NumPy to st.image, no Matplotlib lag)
        col_img1, col_img2, col_img3 = st.columns(3)

        with col_img1:
            st.markdown(f"<p style='color:#00e5ff; font-weight:bold; margin-bottom:4px;'>Axial Plane (Z = {slice_z})</p>", unsafe_allow_html=True)
            img_axial = np.rot90(fast_slice_render(underlay[slice_z, :, :], seg_mask[slice_z, :, :], show_tumor_mask))
            st.image(img_axial, use_container_width=True)

        with col_img2:
            st.markdown(f"<p style='color:#00e5ff; font-weight:bold; margin-bottom:4px;'>Coronal Plane (Y = {slice_y})</p>", unsafe_allow_html=True)
            img_coronal = np.rot90(fast_slice_render(underlay[:, slice_y, :], seg_mask[:, slice_y, :], show_tumor_mask))
            st.image(img_coronal, use_container_width=True)

        with col_img3:
            st.markdown(f"<p style='color:#00e5ff; font-weight:bold; margin-bottom:4px;'>Sagittal Plane (X = {slice_x})</p>", unsafe_allow_html=True)
            img_sagittal = np.rot90(fast_slice_render(underlay[:, :, slice_x], seg_mask[:, :, slice_x], show_tumor_mask))
            st.image(img_sagittal, use_container_width=True)

# ==============================================================================
# PAGE 3: INTERACTIVE 3D VIEWER
# ==============================================================================
elif page == "🎮 3. Interactive 3D Viewer":
    st.title("Interactive 3D Neuro-Navigation")

    if st.session_state["pipeline_results"] is None:
        st.warning("Please execute AI Segmentation on Page 2 first.")
        st.stop()

    results = st.session_state["pipeline_results"]
    case_name = st.session_state["case_name"]
    flair_prep = results["preprocessed_image"][0]
    seg_mask = results["seg_mask"]
    affine = results["affine"]

    # Extract or Load Meshes
    if st.session_state["obj_strings"] is None:
        with st.spinner("Extracting smooth 3D polygon meshes with Marching Cubes..."):
            raw_brain = (flair_prep != 0) | (seg_mask > 0)
            solid_brain = binary_fill_holes(raw_brain)
            solid_brain = binary_closing(solid_brain, structure=np.ones((3, 3, 3)))

            meshes = {
                "brain": extract_surface_mesh(solid_brain, affine, sigma=2.0, step=2, smooth_iter=25, decimate=0.5),
                "edema": extract_surface_mesh(seg_mask == 2, affine, sigma=1.0, step=1, smooth_iter=20, decimate=0.4),
                "core": extract_surface_mesh(seg_mask == 1, affine, sigma=1.0, step=1, smooth_iter=20, decimate=0.4),
                "enhancing": extract_surface_mesh(seg_mask == 4, affine, sigma=1.0, step=1, smooth_iter=20, decimate=0.4),
            }
            st.session_state["meshes"] = meshes

            # Convert to OBJ strings
            mesh_out_dir = Path("outputs/meshes")
            mesh_out_dir.mkdir(parents=True, exist_ok=True)

            obj_strings = {}
            for k, m in meshes.items():
                if m is not None and m.n_points > 0:
                    tmp_p = mesh_out_dir / f"{case_name}_{k}.obj"
                    m.save(str(tmp_p))
                    with open(tmp_p, "r", encoding="utf-8") as f:
                        obj_strings[k] = f.read()
                else:
                    obj_strings[k] = ""
            st.session_state["obj_strings"] = obj_strings

    obj_strings = st.session_state["obj_strings"]

    # Three.js WebGL HTML with Collapsible HUD
    html_3d = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ margin:0; padding:0; overflow:hidden; background:#0a0a0e; font-family: monospace; }}
        #container {{ width:100vw; height:700px; }}
        #hud {{
            position:absolute; top:15px; left:15px;
            background:rgba(14,18,26,0.92); backdrop-filter:blur(10px);
            border:1px solid #00e5ff; border-radius:8px;
            padding:14px 18px; color:#fff; width:280px; z-index:100; user-select:none;
            box-shadow: 0 4px 20px rgba(0, 229, 255, 0.25);
        }}
        .hud-header {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; }}
        .hud-header h3 {{ margin:0; font-size:12px; color:#00e5ff; letter-spacing:1px; }}
        .btn-hide {{
            background:transparent; color:#8899a6; border:1px solid #445566;
            border-radius:4px; padding:2px 8px; font-size:10px; cursor:pointer;
        }}
        .btn-hide:hover {{ color:#00e5ff; border-color:#00e5ff; }}
        #btn-show-hud {{
            display:none; position:absolute; top:15px; left:15px; z-index:101;
            background:rgba(14,18,26,0.92); color:#00e5ff; border:1px solid #00e5ff;
            border-radius:6px; padding:8px 14px; font-size:11px; cursor:pointer; font-weight:bold;
        }}
        .ctrl {{ margin-bottom:8px; }}
        .ctrl label {{ display:flex; justify-content:space-between; font-size:11px; margin-bottom:2px; }}
        input[type="range"] {{ width:100%; height:4px; accent-color:#00e5ff; cursor:pointer; }}
        .row {{ display:flex; align-items:center; justify-content:space-between; font-size:11px; margin-top:4px; }}
        .badge {{ padding:2px 5px; border-radius:3px; font-weight:bold; font-size:9px; }}
        .btn-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:5px; margin-top:10px; }}
        button.view-btn {{
            background:#161d2b; color:#00e5ff; border:1px solid #00e5ff;
            border-radius:4px; padding:5px; font-size:10px; cursor:pointer;
        }}
        button.view-btn:hover {{ background:#00e5ff; color:#0a0a0e; font-weight:bold; }}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/OBJLoader.js"></script>
</head>
<body>
    <button id="btn-show-hud" onclick="toggleHud(true)">Controls</button>

    <div id="hud">
        <div class="hud-header">
            <h3>{case_name}</h3>
            <button class="btn-hide" onclick="toggleHud(false)">Hide</button>
        </div>
        <div class="row"><span><span class="badge" style="background:#555;">BRAIN</span> Anatomy</span><input type="checkbox" id="c-b" checked></div>
        <div class="ctrl"><label><span>Brain Opacity</span><span id="v-b">15%</span></label><input type="range" id="s-b" min="0" max="1" step="0.02" value="0.15"></div>

        <div class="row"><span><span class="badge" style="background:#f5d300; color:#000;">EDEMA</span> Yellow</span><input type="checkbox" id="c-e" checked></div>
        <div class="ctrl"><label><span>Edema Opacity</span><span id="v-e">25%</span></label><input type="range" id="s-e" min="0" max="1" step="0.02" value="0.25"></div>

        <div class="row"><span><span class="badge" style="background:#e62020;">CORE</span> Red</span><input type="checkbox" id="c-c" checked></div>
        <div class="ctrl"><label><span>Core Opacity</span><span id="v-c">100%</span></label><input type="range" id="s-c" min="0" max="1" step="0.02" value="1.00"></div>

        <div class="row"><span><span class="badge" style="background:#1080ff;">ACTIVE</span> Blue</span><input type="checkbox" id="c-a" checked></div>
        <div class="ctrl"><label><span>Active Opacity</span><span id="v-a">70%</span></label><input type="range" id="s-a" min="0" max="1" step="0.02" value="0.70"></div>

        <div class="btn-grid">
            <button class="view-btn" onclick="setCam('iso')">Isometric</button>
            <button class="view-btn" onclick="setCam('axial')">Axial</button>
            <button class="view-btn" onclick="setCam('coronal')">Coronal</button>
            <button class="view-btn" onclick="setCam('sagittal')">Sagittal</button>
        </div>
    </div>
    <div id="container"></div>

    <script>
        function toggleHud(show) {{
            document.getElementById('hud').style.display = show ? 'block' : 'none';
            document.getElementById('btn-show-hud').style.display = show ? 'none' : 'block';
        }}
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0a0a0e);
        const camera = new THREE.PerspectiveCamera(45, window.innerWidth / 700, 0.1, 1500);
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, 700);
        renderer.setPixelRatio(window.devicePixelRatio);
        document.getElementById('container').appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true; controls.dampingFactor = 0.05;

        scene.add(new THREE.AmbientLight(0xffffff, 0.7));
        const d1 = new THREE.DirectionalLight(0xffffff, 0.85); d1.position.set(200, 300, 200); scene.add(d1);
        const d2 = new THREE.DirectionalLight(0x00e5ff, 0.5); d2.position.set(-200, -200, -200); scene.add(d2);

        const matB = new THREE.MeshPhongMaterial({{ color:0xd0d0d8, transparent:true, opacity:0.15, side:THREE.DoubleSide, depthWrite:false }});
        const matE = new THREE.MeshPhongMaterial({{ color:0xf5d300, transparent:true, opacity:0.25, side:THREE.DoubleSide, depthWrite:false }});
        const matC = new THREE.MeshPhongMaterial({{ color:0xe62020, transparent:false, opacity:1.0, side:THREE.DoubleSide }});
        const matA = new THREE.MeshPhongMaterial({{ color:0x1080ff, transparent:true, opacity:0.70, side:THREE.DoubleSide }});

        const loader = new THREE.OBJLoader();
        let mB, mE, mC, mA, center = new THREE.Vector3(0, 0, 0);

        function loadObj(txt, mat, cb) {{
            if (!txt || txt.trim().length === 0) return;
            const obj = loader.parse(txt);
            obj.traverse(c => {{ if (c.isMesh) c.material = mat; }});
            scene.add(obj);
            if (cb) cb(obj);
        }}
        loadObj(`{obj_strings.get('brain', '')}`, matB, o => {{
            mB = o;
            const bbox = new THREE.Box3().setFromObject(mB);
            bbox.getCenter(center);
            controls.target.copy(center);
            setCam('iso');
        }});
        loadObj(`{obj_strings.get('edema', '')}`, matE, o => mE = o);
        loadObj(`{obj_strings.get('core', '')}`, matC, o => mC = o);
        loadObj(`{obj_strings.get('enhancing', '')}`, matA, o => mA = o);

        document.getElementById('s-b').oninput = e => {{ matB.opacity = parseFloat(e.target.value); document.getElementById('v-b').innerText = Math.round(e.target.value*100) + '%'; }};
        document.getElementById('c-b').onchange = e => {{ if(mB) mB.visible = e.target.checked; }};
        document.getElementById('s-e').oninput = e => {{ matE.opacity = parseFloat(e.target.value); document.getElementById('v-e').innerText = Math.round(e.target.value*100) + '%'; }};
        document.getElementById('c-e').onchange = e => {{ if(mE) mE.visible = e.target.checked; }};
        document.getElementById('s-c').oninput = e => {{ matC.opacity = parseFloat(e.target.value); matC.transparent = (e.target.value < 1.0); document.getElementById('v-c').innerText = Math.round(e.target.value*100) + '%'; }};
        document.getElementById('c-c').onchange = e => {{ if(mC) mC.visible = e.target.checked; }};
        document.getElementById('s-a').oninput = e => {{ matA.opacity = parseFloat(e.target.value); document.getElementById('v-a').innerText = Math.round(e.target.value*100) + '%'; }};
        document.getElementById('c-a').onchange = e => {{ if(mA) mA.visible = e.target.checked; }};

        function setCam(view) {{
            const dist = 320;
            if (view === 'axial') {{ camera.position.set(center.x, center.y, center.z + dist); camera.up.set(0, 1, 0); }}
            else if (view === 'coronal') {{ camera.position.set(center.x, center.y - dist, center.z); camera.up.set(0, 0, 1); }}
            else if (view === 'sagittal') {{ camera.position.set(center.x + dist, center.y, center.z); camera.up.set(0, 0, 1); }}
            else {{ camera.position.set(center.x + dist*0.7, center.y - dist*0.7, center.z + dist*0.7); camera.up.set(0, 0, 1); }}
            controls.target.copy(center); controls.update();
        }}
        function anim() {{ requestAnimationFrame(anim); controls.update(); renderer.render(scene, camera); }}
        anim();
    </script>
</body>
</html>
"""
    components.html(html_3d, height=720, scrolling=False)
    st.info("Navigation Tips: Left-Click + Drag: Rotate 360 | Scroll Wheel: Zoom | Click 'Hide' in the HUD to collapse controls.")

# ==============================================================================
# PAGE 4: QUANTITATIVE ANALYTICS
# ==============================================================================
elif page == "📊 4. Quantitative Analytics":
    st.title("Clinical Morphometrics & Analytics")

    if st.session_state["pipeline_results"] is None:
        st.warning("Please execute AI Segmentation on Page 2 first.")
        st.stop()

    case_name = st.session_state["case_name"]
    results = st.session_state["pipeline_results"]
    seg_mask = results["seg_mask"]
    affine = results["affine"]
    meshes = st.session_state.get("meshes", {})

    # Compute Analytics Engine
    if st.session_state["analytics"] is None:
        with st.spinner("Computing surface areas, diameters, and sphericity..."):
            engine = TumorAnalyticsEngine()
            mask_wt = (seg_mask > 0)
            mask_tc = np.isin(seg_mask, [1, 4])
            mask_et = (seg_mask == 4)
            mask_ed = (seg_mask == 2)
            mask_nc = (seg_mask == 1)

            st.session_state["analytics"] = {
                "wt": engine.compute_region_metrics(mask_wt, meshes.get("edema"), affine),
                "tc": engine.compute_region_metrics(mask_tc, meshes.get("core"), affine),
                "et": engine.compute_region_metrics(mask_et, meshes.get("enhancing"), affine),
                "ed": engine.compute_region_metrics(mask_ed, meshes.get("edema"), affine),
                "nc": engine.compute_region_metrics(mask_nc, meshes.get("core"), affine),
            }
            st.session_state["localization"] = engine.determine_anatomical_localization(
                st.session_state["analytics"]["wt"]["centroid_ras_mm"]
            )

    an = st.session_state["analytics"]
    loc = st.session_state["localization"]

    # KPI Metric Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Whole Tumor (WT)", f"{an['wt']['volume_ml']:.1f} mL")
    kpi2.metric("Surgical Core (TC)", f"{an['tc']['volume_ml']:.1f} mL")
    kpi3.metric("Max 3D Diameter", f"{an['wt']['max_diameter_mm']:.1f} mm")
    kpi4.metric("Sphericity Index", f"{an['wt']['sphericity']:.2f}")

    st.markdown("---")
    c_left, c_right = st.columns([1, 1])

    with c_left:
        st.subheader("Sub-Region Volumetrics (mL)")
        fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0e1118")
        ax.set_facecolor("#141824")
        labels = ["Edema (Yellow)", "Necrotic (Red)", "Active (Blue)"]
        vals = [an["ed"]["volume_ml"], an["nc"]["volume_ml"], an["et"]["volume_ml"]]
        colors = ["#f5d300", "#e62020", "#1080ff"]
        bars = ax.bar(labels, vals, color=colors, width=0.5, edgecolor="white")
        ax.set_ylabel("Volume (mL = cm³)", color="white")
        ax.tick_params(colors="white")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2., h + 1, f"{h:.1f}", ha='center', color="white", weight="bold")
        st.pyplot(fig)
        plt.close(fig)  # Prevent Matplotlib memory buildup

    with c_right:
        st.subheader("Neurosurgical Stereotactic Localization")
        st.markdown(f"""
        **Primary Hemisphere:** `{loc['hemisphere']}`
        **Spatial Orientation:** `{loc['orientation']}`

        **3D Physical Centroid (RAS mm):**
        - **X (Lateral/Medial):** `{an['wt']['centroid_ras_mm'][0]} mm`
        - **Y (Anterior/Posterior):** `{an['wt']['centroid_ras_mm'][1]} mm`
        - **Z (Superior/Inferior):** `{an['wt']['centroid_ras_mm'][2]} mm`

        **Morphology Classification:**
        - **Sphericity:** `{an['wt']['sphericity']}` ({'Compact Mass' if an['wt']['sphericity'] > 0.65 else 'Infiltrative Infiltration'})
        - **Total Surface Area:** `{an['wt']['surface_area_cm2']:.1f} cm²`
        """)

# ==============================================================================
# PAGE 5: SURGICAL REPORT DOSSIER
# ==============================================================================
elif page == "📑 5. Surgical Report Dossier":
    st.title("Surgical Neuro-Navigation Report")

    if st.session_state["pipeline_results"] is None or st.session_state["analytics"] is None:
        st.warning("Please complete AI Segmentation and Analytics first.")
        st.stop()

    case_name = st.session_state["case_name"]
    an = st.session_state["analytics"]
    loc = st.session_state["localization"]

    report_md = f"""# NEUROVISION AI - PRE-OPERATIVE SURGICAL DOSSIER
**Patient Case Identifier:** `{case_name}`
**Evaluation Date:** October 2026 | **Modality:** 4-Channel MRI (FLAIR, T1, T1CE, T2)
**Segmentation Architecture:** MONAI 3D Residual UNet (Isotropic 1.0mm)

---

### 1. CLINICAL MORPHOMETRIC MEASUREMENTS
| Tumor Sub-Region | Physical Volume (mL) | Surface Area (cm²) | Max 3D Feret Diameter (mm) | Sphericity (0-1) |
| :--- | :--- | :--- | :--- | :--- |
| **Whole Tumor (WT: 1+2+4)** | **{an['wt']['volume_ml']:.2f} mL** | {an['wt']['surface_area_cm2']:.1f} cm² | {an['wt']['max_diameter_mm']:.1f} mm | {an['wt']['sphericity']:.3f} |
| ├─ Peritumoral Edema (Yellow) | {an['ed']['volume_ml']:.2f} mL | {an['ed']['surface_area_cm2']:.1f} cm² | {an['ed']['max_diameter_mm']:.1f} mm | {an['ed']['sphericity']:.3f} |
| ├─ Tumor Core (TC: 1+4) | **{an['tc']['volume_ml']:.2f} mL** | {an['tc']['surface_area_cm2']:.1f} cm² | {an['tc']['max_diameter_mm']:.1f} mm | {an['tc']['sphericity']:.3f} |
| ├─ Necrotic Cavity (Red) | {an['nc']['volume_ml']:.2f} mL | {an['nc']['surface_area_cm2']:.1f} cm² | {an['nc']['max_diameter_mm']:.1f} mm | {an['nc']['sphericity']:.3f} |
| └─ Active Enhancing Rim (Blue) | {an['et']['volume_ml']:.2f} mL | {an['et']['surface_area_cm2']:.1f} cm² | {an['et']['max_diameter_mm']:.1f} mm | {an['et']['sphericity']:.3f} |

---

### 2. STEREOTACTIC CRANIOTOMY COORDINATES
* **Anatomical Target:** {loc['formatted_location']}
* **Center of Mass (RAS Physical Space):**
  * **$X$ (Right/Left):** `{an['wt']['centroid_ras_mm'][0]} mm`
  * **$Y$ (Anterior/Posterior):** `{an['wt']['centroid_ras_mm'][1]} mm`
  * **$Z$ (Superior/Inferior):** `{an['wt']['centroid_ras_mm'][2]} mm`

---

### 3. SURGICAL IMPLICATIONS
* **Resection Volume Target:** Planning target volume (TC) for maximal safe resection is **{an['tc']['volume_ml']:.1f} mL**.
* **Tissue Invasiveness:** Sphericity index of **{an['wt']['sphericity']}** indicates {'compact nodular boundaries' if an['wt']['sphericity'] > 0.65 else 'diffuse white matter infiltration requiring fluorescence-guided resection'}.
"""

    st.markdown(report_md)
    st.markdown("---")

    st.download_button(
        label="Download Clinical Report (Markdown)",
        data=report_md,
        file_name=f"{case_name}_Surgical_Report.md",
        mime="text/markdown"
    )