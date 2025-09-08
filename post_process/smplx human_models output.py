import os
import glob
import pickle
import numpy as np
import torch
import trimesh
import pyrender
import imageio
from tqdm import tqdm
import smplx
from scipy.spatial.transform import Rotation as R

# ==============================================================================
# ========================  YOU CAN EDIT THESE PATHS  ==========================
# ==============================================================================

# Path to the DIRECTORY containing the SMPL-X model files (e.g., SMPLX_NEUTRAL.npz)
SMPLX_MODEL_DIR = r"D:\KMUTT-Master's Degree\Research\Text2Sign\Code\SMPLest-X-ReverseE-GR\human_models\human_model_files\smplx\SMPLX_NEUTRAL_2020.npz"

# Path to the folder containing your extracted .pkl parameter files
PARAMS_DIR = r"D:\KMUTT-Master's Degree\Research\Text2Sign\Code\SMPLest-X-ReverseE-GR"

# Path where the output video will be saved
OUTPUT_BASE_DIR = r"D:\KMUTT-Master's Degree\Research\Text2Sign\Code\SMPLest-X-ReverseE-GR\demo output"

# You can also change the FPS of the output video here
VIDEO_FPS = 30

# ==============================================================================
# =====================  No need to edit below this line  ======================
# ==============================================================================

def setup_scene(camera_translation):
    """Creates the pyrender scene, camera, and light."""
    scene = pyrender.Scene(bg_color=[0.1, 0.1, 0.3, 1.0], ambient_light=[0.3, 0.3, 0.3])
    
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = camera_translation
    
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    scene.add(camera, pose=camera_pose)
    
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(light, pose=camera_pose)
    
    return scene

def main():
    """
    Main function to generate a 3D animation video from SMPL-X parameters.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 1. Load SMPL-X Model ---
    print("Loading SMPL-X model...")
    try:
        model = smplx.SMPLX(
            model_path=SMPLX_MODEL_DIR,
            gender='neutral',
            use_pca=False,
            flat_hand_mean=True,
            num_betas=10,
            num_expression_coeffs=10
        ).to(device)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"--- ERROR ---")
        print(f"Failed to load SMPL-X model from the directory: '{SMPLX_MODEL_DIR}'")
        print(f"Please ensure this directory contains 'SMPLX_NEUTRAL.npz' and other model files.")
        print(f"Error details: {e}")
        return

    # --- 2. Find Parameter Files ---
    param_files = sorted(glob.glob(os.path.join(PARAMS_DIR, '*.pkl')))
    if not param_files:
        print(f"ERROR: No .pkl files found in '{PARAMS_DIR}'.")
        return
    print(f"Found {len(param_files)} parameter files.")
    
    # --- 3. Set Output Path ---
    folder_name = os.path.basename(os.path.normpath(PARAMS_DIR))
    output_filename = f"{folder_name}_animation.mp4"
    output_path = os.path.join(OUTPUT_BASE_DIR, output_filename)


    # --- 4. Determine Initial Camera Position ---
    initial_camera_z = 2.5 # Default Z distance if not found
    with open(param_files[0], 'rb') as f:
        first_frame_params = pickle.load(f)
        if 'cam_trans' in first_frame_params:
            initial_transl = first_frame_params['cam_trans'][0]
            initial_camera_z = initial_transl[2] + 2.0 # Position camera in front
    
    camera_translation = np.array([0, 0.4, initial_camera_z]) # Slightly elevate camera
    print(f"Setting initial camera Z position to: {initial_camera_z:.2f}")

    # --- 5. Setup Renderer and Video Writer ---
    scene = setup_scene(camera_translation)
    renderer = pyrender.OffscreenRenderer(viewport_width=800, viewport_height=800)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    video_writer = imageio.get_writer(output_path, fps=VIDEO_FPS)
    print(f"Video will be saved to '{output_path}'.")

    correction_rot = R.from_euler('x', np.pi, degrees=False).as_matrix()

    # --- 6. Main Processing Loop ---
    for pkl_file in tqdm(param_files, desc=f"Rendering '{os.path.basename(output_path)}'"):
        with open(pkl_file, 'rb') as f:
            frame_params = pickle.load(f)

        current_node = None
        try:
            param_mapping = {
                'betas': 'smplx_shape',
                'body_pose': 'smplx_body_pose',
                'left_hand_pose': 'smplx_lhand_pose',
                'right_hand_pose': 'smplx_rhand_pose',
                'jaw_pose': 'smplx_jaw_pose',
                'expression': 'smplx_expr',
                'global_orient': 'smplx_root_pose',
                'transl': 'cam_trans'
            }

            body_params = {}
            for model_key, pkl_key in param_mapping.items():
                if pkl_key in frame_params:
                    tensor_data = torch.from_numpy(frame_params[pkl_key]).float().to(device)
                    body_params[model_key] = tensor_data

            original_orient_vec = body_params['global_orient'].squeeze().cpu().numpy()
            original_orient_mat = R.from_rotvec(original_orient_vec).as_matrix()
            corrected_orient_mat = correction_rot @ original_orient_mat
            body_params['global_orient'] = torch.from_numpy(R.from_matrix(corrected_orient_mat).as_rotvec()).unsqueeze(0).float().to(device)

            with torch.no_grad():
                model_output = model(return_verts=True, **body_params)
            
            vertices = model_output.vertices.detach().cpu().numpy().squeeze()
            mesh = trimesh.Trimesh(vertices, model.faces, process=False)
            
            render_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
            current_node = scene.add(render_mesh)

            color, _ = renderer.render(scene)
            video_writer.append_data(color)

        except Exception as e:
            print(f"\nError processing file {os.path.basename(pkl_file)}: {e}")
            blank_frame = np.full((800, 800, 3), (26, 26, 76), dtype=np.uint8)
            video_writer.append_data(blank_frame)
        
        finally:
            if current_node is not None:
                scene.remove_node(current_node)

    video_writer.close()
    renderer.delete()
    print("\n---------------------------------")
    print(f"Animation successfully created at: {output_path}")
    print("---------------------------------")

if __name__ == '__main__':
    # Sanity check the paths before running
    if not os.path.exists(SMPLX_MODEL_DIR):
        print(f"ERROR: SMPL-X model directory not found at '{SMPLX_MODEL_DIR}'")
        print("Please verify the path.")
    elif not os.path.exists(PARAMS_DIR):
        print(f"ERROR: Parameter directory not found at '{PARAMS_DIR}'")
        print("Please verify the path.")
    else:
        print("Paths seem to be configured correctly. Starting rendering process...")
        main()

