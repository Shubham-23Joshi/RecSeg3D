# RecSeg3D

RecSeg3D is a constrained optimization-based pipeline which performs 3D segmentation and refinement of rectangular planes, such as surfaces of boxes, from an RGB-D scene data. The pipeline takes segmentation masks (e.g. from YOLO or SAM) and RGB-D data of the target scene as input and iteratively refines the 2D and 3D segmentations and the 6D pose of the target rectangular planes from the scene using known geometric constraints of rectangular planar structures.
---

## Installation

#### 1. Create a Python virtual environment and activate it

On Linux:

python3 -m venv venv

source venv/bin/activate

#### 2. Clone the repository inside venv

git clone <REPO_URL>
cd <REPO_NAME>

#### 3. Upgrade pip and install dependancies 

pip install --upgrade pip
pip install -r requirements.txt

#### 5. Add segmentation model or masks

Option A: Use YOLO instance segmentation

- Download a YOLO instance segmentation model (.pt format)

- Place it in the appropriate model directory (as expected by your setup)

- Set runYOLO to True in the RecSeg3D_eval.py

Option B: Use any other segmentation method

- Generate binary mask images externally

- Place them in: RecSeg3D_data/permanent_data/YOLO_results/

The pipeline assumes masks already exist in this folder.


## Running the Pipeline

python3 RecSeg3D_eval.py --mask <MASK_INDEX>

Example:

python3 RecSeg3D_eval.py --mask 3

Important: Wayland + Open3D Warning
If you are running Ubuntu with Wayland, Open3D visualization will fail due to GLFW / GLEW limitations.

Either change to X11 or force X11 only for this command:

XDG_SESSION_TYPE=x11 python3 RecSeg3D_eval.py --mask <MASK_INDEX>

