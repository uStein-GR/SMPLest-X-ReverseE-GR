import sys
import os
os.environ['PYOPENGL_PLATFORM'] = 'win32'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import re
import argparse

def frames_to_video(image_folder, output_video_file, fps):
    """
    Combines sorted image frames from a folder into a video.

    Args:
        image_folder (str): Path to the folder containing the image frames.
        output_video_file (str): Name and path for the output video file (e.g., 'output/video.mp4').
        fps (int): Frames per second for the output video.
    """
    if not os.path.isdir(image_folder):
        print(f"Error: The image folder '{image_folder}' does not exist.")
        return

    images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
    
    if not images:
        print(f"Error: No images found in the folder '{image_folder}'.")
        return

    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

    images.sort(key=natural_sort_key)

    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"Error: Could not read the first image: {first_image_path}")
        return
        
    height, width, layers = frame.shape
    size = (width, height)

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_video_file)
    if output_dir: # Check if there is a directory part
        os.makedirs(output_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_video_file, fourcc, fps, size)
    
    if not out.isOpened():
        print(f"Error: Could not open video writer for '{output_video_file}'.")
        return

    print(f"Starting video creation with {len(images)} frames from '{image_folder}'...")

    for i, image_name in enumerate(images):
        image_path = os.path.join(image_folder, image_name)
        frame = cv2.imread(image_path)
        if frame is not None:
            out.write(frame)
        else:
            print(f"Warning: Could not read frame {image_name}. Skipping.")

    out.release()
    cv2.destroyAllWindows()
    
    print("\n-----------------------------------------")
    print(f"Success! Video has been saved as '{output_video_file}'")
    print("-----------------------------------------")


if __name__ == '__main__':
    # --- Command-Line Argument Parsing ---
    parser = argparse.ArgumentParser(description="Convert a folder of image frames into a video file.")
    parser.add_argument('--image_folder', type=str, required=True, help='Path to the folder containing image frames.')
    parser.add_argument('--output_video', type=str, required=True, help='Path and filename for the output video.')
    parser.add_argument('--fps', type=int, required=True, help='Frames per second for the output video.')
    
    args = parser.parse_args()
    
    # --- Call the main function with arguments from the command line ---
    frames_to_video(args.image_folder, args.output_video, args.fps)
