# CV HW2: Stereo Disparity Estimation

## Install required packages:  
`pip install -r requirements.txt`  
`cuda` is necessary to run inference.

## Folder structure:
```
root/
├── images/
│   ├── art/
│   │   ├── disp1.png
│   │   ├── view1.png
│   │   ├── view5.png
│   │   └── pred.png # output of disparity estimation
│   └── ...
├── models/
│   └── finetune_250.tar
├── requirements.txt
└── main.py
```
- Estimated disparity maps are saved in the same directory as the images.  

## Run inference:  
- Modify some arguments in `main.py`:  
`args["imagedir"] = '/path/to/images'`  
`args["loadmodel"] = '/path/to/model'`  

- Run inference on all images:  
`python main.py`  

- Logs from `main.py` should look like:
```
Art: 21.09342041898062
Dolls: 22.057214279559666
Reindeer: 22.115694242556422
Average PSNR Score: 21.75544298036557
```