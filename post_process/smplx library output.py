import os
import glob
import pickle
import numpy as np
import torch
import cv2
import smplx
import trimesh
import pyrender
import imageio
from tqdm import tqdm

# --- YOU MUST EDIT THESE PATHS ---

# Path to the folder containing the SMPL-X model files (e.g., SMPLX_NEUTRAL.npz)
# This should point to the 'smplx' folder you created in the previous step.
SMPLX_MODEL_DIR = r"D:\KMUTT-Master's Degree\Research\Text2Sign\Code\SMPLest-X-ReverseE-GR\human_models\human_model_files\smplx\SMPLX_NEUTRAL_2020.npz"

# Path to the folder containing your extracted .pkl parameter files
PARAMS_DIR = r"D:\KMUTT-Master's Degree\Research\Text2Sign\Code\SMPLest-X-ReverseE-GR"

# Path where the output video will be saved
OUTPUT_BASE_DIR = r"D:\KMUTT-Master's Degree\Research\Text2Sign\Code\SMPLest-X-ReverseE-GR\demo output"

folder_name = os.path.basename(PARAMS_DIR.rstrip("\\/"))
output_file_name = f"vertices_output_{folder_name}.mp4"

OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_BASE_DIR, output_file_name)

# --- Sanity Check ---
if not os.path.exists(SMPLX_MODEL_DIR):
    print(f"ERROR: SMPL-X model directory not found at '{SMPLX_MODEL_DIR}'")
    print("Please complete Cell 3 and verify this path.")
elif not os.path.exists(PARAMS_DIR):
    print(f"ERROR: Parameter directory not found at '{PARAMS_DIR}'")
    print("Please upload your .pkl files and verify this path.")
else:
    print("Paths seem to be configured correctly. Ready to proceed.")

# --- Device Configuration ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def create_3d_animation(smplx_model_dir, params_dir, output_path, gender='neutral', video_fps=30):
    """
    Generates a 3D animation video from a sequence of SMPL-X parameter files.
    (Final Simplified Version with correct orientation)
    """
    print("Loading SMPL-X model...")
    model = smplx.SMPLX(
        model_path=smplx_model_dir,
        gender=gender,
        use_pca=False,
        flat_hand_mean=True,
        num_betas=10,
        num_expression_coeffs=10
    ).to(device)
    print("Model loaded successfully.")

    param_files = sorted(glob.glob(os.path.join(params_dir, '*.pkl')))
    if not param_files:
        print(f"ERROR: No .pkl files found in '{params_dir}'.")
        return
    print(f"Found {len(param_files)} parameter files.")

    # --- Automatic Camera Positioning Logic ---
    initial_z_translation = 2.5
    initial_x_translation = 0.0
    initial_y_translation = 0.0

    for pkl_file in param_files:
        with open(pkl_file, 'rb') as f:
            first_frame_params = pickle.load(f)
        if first_frame_params and 'transl' in first_frame_params[0]:
            initial_transl = first_frame_params[0]['transl']
            initial_x_translation = initial_transl[0]
            initial_y_translation = initial_transl[1]
            initial_z_translation = initial_transl[2]
            print(f"Found first person's initial translation: x={initial_x_translation:.2f}, y={initial_y_translation:.2f}, z={initial_z_translation:.2f}. Adjusting camera.")
            break

    # Position the camera BEHIND the model with no rotation. This will see the front.
    camera_z_pos = initial_z_translation + 2.5
    camera_x_pos = initial_x_translation
    camera_y_pos = initial_y_translation # Center the camera vertically

    # --- FINAL SIMPLIFIED Model Orientation Correction ---
    from scipy.spatial.transform import Rotation as R
    # We only need the one rotation to flip the model upright.
    model_correction_rot = R.from_euler('x', np.pi, degrees=False)
    # --- END Correction ---

    # --- Setup the 3D renderer ---
    scene = pyrender.Scene(bg_color=[0.1, 0.1, 0.3, 1.0], ambient_light=[0.3, 0.3, 0.3])
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    # The camera pose is a simple translation with NO rotation.
    camera_pose = np.array([
       [1.0, 0.0, 0.0, camera_x_pos],
       [0.0, 1.0, 0.0, camera_y_pos],
       [0.0, 0.0, 1.0, camera_z_pos],
       [0.0, 0.0, 0.0, 1.0],
    ])
    scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(light, pose=camera_pose)
    renderer = pyrender.OffscreenRenderer(viewport_width=800, viewport_height=800)

    # --- Setup Video Writer ---
    height, width = 800, 800
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    video_writer = imageio.get_writer(output_path, fps=video_fps)
    print(f"Video will be saved to '{output_path}'.")

    # --- Main processing loop ---
    for frame_idx, pkl_file in enumerate(tqdm(param_files, desc="Rendering Frames")):
        with open(pkl_file, 'rb') as f:
            frame_all_person_params = pickle.load(f)

        if not frame_all_person_params:
            blank_frame = np.full((height, width, 3), (26, 26, 76), dtype=np.uint8)
            video_writer.append_data(blank_frame)
            continue

        person_params = frame_all_person_params[0]

        try:
            body_params = {}
            for key in ['betas', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'expression', 'transl']:
                if key in person_params:
                    body_params[key] = torch.tensor(person_params[key], dtype=torch.float32).unsqueeze(0).to(device)

            original_global_orient = torch.tensor(person_params['global_orient'], dtype=torch.float32).unsqueeze(0).to(device)
            original_rot_matrix = R.from_rotvec(original_global_orient.squeeze().cpu().numpy()).as_matrix()

            # Apply the simple correction in WORLD space
            corrected_rot_matrix = model_correction_rot.as_matrix() @ original_rot_matrix

            body_params['global_orient'] = torch.tensor(R.from_matrix(corrected_rot_matrix).as_rotvec(), dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                model_output = model(return_verts=True, **body_params)
                vertices = model_output.vertices.detach().cpu().numpy().squeeze()
                faces = model.faces

            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

            pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)

            for node in list(scene.mesh_nodes):
                scene.remove_node(node)

            scene.add(pyrender_mesh)
            color, _ = renderer.render(scene)
            video_writer.append_data(color)

        except Exception as e:
            print(f"\nError processing frame {frame_idx}: {e}")
            blank_frame = np.full((height, width, 3), (26, 26, 76), dtype=np.uint8)
            video_writer.append_data(blank_frame)
            continue

    video_writer.close()
    renderer.delete()
    print("\n---------------------------------")
    print(f"Animation successfully created at: {output_path}")
    print("---------------------------------")

create_3d_animation(
    smplx_model_dir=SMPLX_MODEL_DIR,
    params_dir=PARAMS_DIR,
    output_path=OUTPUT_VIDEO_PATH,
    video_fps=30 # You can change the FPS of the output video here
)