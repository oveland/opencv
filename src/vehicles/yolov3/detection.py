# Vehicle counting and Classification
import cv2
import csv
import numpy as np
from tracker import *
from datetime import datetime

# Initialize Tracker
tracker = EuclideanDistTracker()

# Initialize the videocapture object
cap = cv2.VideoCapture('../../assets/video/video.mp4')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

frame_size_in = (320, 160)
# frame_size_in = (int(frame_width/2), int(frame_width/2))
frame_size_out = (frame_width, frame_height)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('out.mp4', fourcc, 30.0, frame_size_out)

input_size = 0

# Detection confidence threshold
confThreshold = 0.2
nmsThreshold = 0.5

font_color = (0, 255, 255)
font_size = 0.4
font_thickness = 1

# Middle cross line position
middle_line_position = 120
up_line_position = middle_line_position - 20
down_line_position = middle_line_position + 20

# Store Coco Names in a list
classesFile = "../../detectors/yolov3/coco.names"
classNames = open(classesFile).read().strip().split('\n')

# class index for our required detection classes
required_class_index = [2, 3, 5, 7]

detected_classNames = []

# Model Files
modelConfiguration = '../../detectors/yolov3/yolov3-320.cfg'
modelWeights = '../../detectors/yolov3/yolov3-320.weights'

# configure the network model
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

# Configure the network backend

# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Define random colour for each class
np.random.seed(42)
colors = np.random.randint(100, 255, size=(len(classNames), 3), dtype='uint8')


# Function for finding the center of a rectangle
def find_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


# List for store vehicle count information
temp_up_list = []
temp_down_list = []
up_list = [0, 0, 0, 0]
down_list = [0, 0, 0, 0]


# Function for count vehicle
def count_vehicle(box_id, img):
    x, y, w, h, idd, index = box_id

    # Find the center of the rectangle for detection
    center = find_center(x, y, w, h)
    ix, iy = center

    # Find the current position of the vehicle
    if up_line_position < iy < middle_line_position:
        if idd not in temp_up_list:
            temp_up_list.append(idd)

    elif down_line_position > iy > middle_line_position:
        if idd not in temp_down_list:
            temp_down_list.append(idd)

    elif iy < up_line_position:
        if idd in temp_down_list:
            temp_down_list.remove(idd)
            up_list[index] = up_list[index] + 1

    elif iy > down_line_position:
        if idd in temp_up_list:
            temp_up_list.remove(idd)
            down_list[index] = down_list[index] + 1

    # Draw circle in the middle of the rectangle
    cv2.circle(img, center, 2, (0, 0, 255), -1)  # end here
    # print(up_list, down_list)
    cv2.putText(img, f'{idd}', center, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)


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
            name = classNames[class_ids[i]]
            detected_classNames.append(name)
            # Draw classname and confidence score
            cv2.putText(img, f'{name.upper()} {int(confidence_scores[i] * 100)}%', (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.3, color, 1)

            # Draw bounding rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
            # noinspection PyTypeChecker
            detection.append([x, y, w, h, required_class_index.index(class_ids[i])])

    # Update the tracker for each object
    boxes_ids = tracker.update(detection)
    for box_id in boxes_ids:
        count_vehicle(box_id, img)

    for new_detection in tracker.new_center_points:
        cv2.circle(img, new_detection, 3, (255, 0, 0), -1)  # end here


def real_time():
    skip = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        if skip % 10 == 0 & success:
            img = cv2.resize(img, frame_size_in, None)
            ih, iw, channels = img.shape

            blob = cv2.dnn.blobFromImage(img, 1 / 255, frame_size_in, [0, 0, 0], 1, crop=False)

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

            # Draw the crossing lines

            cv2.line(img, (0, middle_line_position), (iw, middle_line_position), (10, 200, 10), 1)
            cv2.line(img, (0, up_line_position), (iw, up_line_position), (200, 100, 255), 1)
            cv2.line(img, (0, down_line_position), (iw, down_line_position), (200, 100, 255), 1)

            # Draw counting texts in the frame
            cv2.putText(img, "->", (60, 30), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), 2)
            cv2.putText(img, "<-", (120, 30), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), 2)
            cv2.putText(img, "Carros: " + str(up_list[0]) + "       " + str(down_list[0]), (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
            cv2.putText(img, "Motos : " + str(up_list[1]) + "       " + str(down_list[1]), (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
            cv2.putText(img, "Buses : " + str(up_list[2]) + "       " + str(down_list[2]), (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
            cv2.putText(img, "Carga : " + str(up_list[3]) + "       " + str(down_list[3]), (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)

            # Show the frames

            img = cv2.resize(img, frame_size_out, None)

            cv2.imshow('Original', img)
            out.write(img)

        skip = skip + 1

        if cv2.waitKey(1) == ord('q'):
            break

    # Write the vehicle counting information in a file and save it
    with open("data.csv", 'w') as f1:
        csv_writer = csv.writer(f1)
        csv_writer.writerow(['Direction', 'car', 'motorbike', 'bus', 'truck'])
        # noinspection PyTypeChecker
        up_list.insert(0, "Up")
        # noinspection PyTypeChecker
        down_list.insert(0, "Down")
        csv_writer.writerow(up_list)
        csv_writer.writerow(down_list)
    f1.close()
    # print("Data saved at 'data.csv'")
    # Finally release the capture object and destroy all active windows
    cap.release()
    out.release()
    # cv2.destroyAllWindows()


def now():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")


if __name__ == '__main__':
    print('Start process at ', now())
    real_time()
    print('Finish process at ', now())
    # from_static_image(image_file)
