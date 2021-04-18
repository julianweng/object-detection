# object-detection
Captures video from camera or screen and detects objects frame by frame.

## Setup

Run in administrative mode
```
pip install -r requirements.txt
```

Copy https://pjreddie.com/media/files/yolov3.weights into /model

```
python yolo.py # use the front-camera
python yolo.py -r # use the on-screen video
```