import cv2
from datetime import datetime
import numpy as np

# Initialize the videocapture object
cap = cv2.VideoCapture('../../assets/images/stand-up.png')
frame_size = (int(cap.get(3)), int(cap.get(4)))

fps = 10

# ••••••••••••••••• Detection config •••••••••••••••••
# Detection confidence threshold
confThreshold = 0.24
nmsThreshold = 0.4

# Store Coco Names in a list
classes_file = "../../detectors/yolov3/coco.names"
class_names = open(classes_file).read().strip().split('\n')

# Class index for our required detection classes
required_class_index = [0, 2, 3, 5, 7]
detected_classNames = []

# Model Files
modelConfiguration = '../../detectors/yolov3/yolov3-320.cfg'
modelWeights = '../../detectors/yolov3/yolov3-320.weights'

# Configure the network model
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

# Configure the network backend
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# •••••••••••••• Styles config •••••••••••••••••••••••
font_color = (0, 255, 255)
font_size = 0.4
font_thickness = 1

# Define random colour for each class
np.random.seed(42)
colors = np.random.randint(100, 255, size=(len(class_names), 3), dtype='uint8')


# Function for finding the detected objects from the network output
def post_process(outputs, img):
    global detected_classNames
    height, width = img.shape[:2]
    boxes = []
    class_ids = []
    confidence_scores = []
    detection = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id in required_class_index:
                if confidence > confThreshold:
                    w, h = int(det[2] * width), int(det[3] * height)
                    x, y = int((det[0] * width) - w / 2), int((det[1] * height) - h / 2)
                    boxes.append([x, y, w, h])
                    class_ids.append(class_id)
                    confidence_scores.append(float(confidence))

    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, confThreshold, nmsThreshold)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
            # print(x,y,w,h)

            color = [int(c) for c in colors[class_ids[i]]]
            name = class_names[class_ids[i]]
            detected_classNames.append(name)
            # Draw classname and confidence score

            label = f'{name.upper()} {int(confidence_scores[i] * 100)}%'

            cv2.putText(img, label, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 0), 2)

            # Draw bounding rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
            # noinspection PyTypeChecker
            detection.append([x, y, w, h, required_class_index.index(class_ids[i])])

    label_total = "Total detections: " + str(len(detection))
    cv2.putText(img, label_total, (60, 30), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), 1)


def real_time():
    while True:
        success, img = cap.read()
        if not success:
            break

        blob = cv2.dnn.blobFromImage(img, 1 / 255, frame_size, [0, 0, 0], 1, crop=False)

        # Set the input of the network
        net.setInput(blob)
        layers_names = net.getLayerNames()
        # print(net.getUnconnectedOutLayers())
        if len(net.getUnconnectedOutLayers()):
            output_names = [layers_names[int(i - 1)] for i in net.getUnconnectedOutLayers()]
        else:
            output_names = []

        # Feed data to the network
        outputs = net.forward(output_names)

        # Find the objects from the network output
        post_process(outputs, img)

        cv2.imshow('Video', img)

        if cv2.waitKey() == ord('q'):
            break

    # Finally, release the capture object and destroy all active windows
    cap.release()


def now():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")


if __name__ == '__main__':
    print('Start process at ', now())
    real_time()
    print('Finish process at ', now())
