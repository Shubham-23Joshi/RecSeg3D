Installation: 
1. create a python virtual environment (on Linux: python3 -m venv "--env_name--")
2. git clone "--repo--"
3. run: source venv/bin/activate
4. run: pip install --upgrade pip
5. run: ip install -r requirements.txt
6. install the yolo instance segmentation model in pt forma and add to the folder (could not include in this git repo due to its size)

Files and Folders:
1. ReSeg3D_eval.py - main pipeline code
run: python3 ReSeg3D_eval.py --mask "--mask_number--"