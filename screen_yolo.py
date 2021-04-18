import numpy as np
import cv2
import mss
import yolo

# Load weights using OpenCV
# Create a list of colors for the labels
labels = open('model/coco.names').read().strip().split('\n')
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
net = cv2.dnn.readNetFromDarknet('model/yolov3.cfg', 'model/yolov3.weights')

# Get the ouput layer names
layer_names = net.getLayerNames()
layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

with mss.mss() as sct:
    monitor = {"top": 40, "left": 0, "width": 800, "height": 640} # Part of the screen to capture
    while "Screen capturing":
        image = np.array(sct.grab(monitor)) # red, green, blue, alpha
        image = np.delete(image, 3, 2) # delete the 'alpha' from 'rgba', (640,800,4) -> (640,800,3)
        boxes, confidences, classIDs, idxs = yolo.make_prediction(net, layer_names, labels, image, 0.5, 0.3)
        image = yolo.draw_bounding_boxes(image, boxes, confidences, classIDs, idxs, colors, labels)

        cv2.imshow('YOLO Object Detection', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()