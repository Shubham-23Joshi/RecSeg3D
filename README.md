Installation: 
1. create a python virtual environment (on Linux: python3 -m venv "--env_name--")
2. git clone "--repo--"
3. run: source venv/bin/activate
4. run: pip install --upgrade pip
5. run: pip install -r requirements.txt
6. install the yolo instance segmentation model in pt format and add to the folder, or use any other segmentation model and add the segmentation mask images in RecSeg3D_data/permanent_data/YOLO_results/ folder

Files and Folders:
1. RecSeg3D_eval.py - main pipeline code
run: python3 RecSeg3D_eval.py --mask {insert mask num here}