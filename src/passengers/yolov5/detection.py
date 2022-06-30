import cv2
import time
import sys
import numpy as np
from datetime import datetime

INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.4


def build_model(with_cuda):
    model_net = cv2.dnn.readNet("../../detectors/yolov5/yolov5s.onnx")
    if with_cuda:
        print("Attempt to use CUDA")
        model_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        model_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    else:
        print("Running on CPU")
        model_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        model_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return model_net


def detect(image, network):
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    network.setInput(blob)
    return network.forward()


def load_capture():
    return cv2.VideoCapture("../../assets/images/passengers.png")


def load_classes():
    with open("../../detectors/yolov5/classes.txt", "r") as f:
        c_list = [cname.strip() for cname in f.readlines()]
    return c_list

# ••••••••••••••••• Detection config •••••••••••••••••
# Detection confidence threshold
confThreshold = 0.24
nmsThreshold = 0.4

class_list = load_classes()
required_class_index = [0]

def wrap_detection(input_image, output_data):
    class_ids = []
    w_confidences = []
    w_boxes = []

    rows = output_data.shape[0]

    image_width, image_height, _ = input_image.shape

    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT

    for r in range(rows):
        row = output_data[r]
        w_confidence = row[4]
        if w_confidence >= 0.4:

            classes_scores = row[5:]
            _, _, _, max_index = cv2.minMaxLoc(classes_scores)
            w_class_id = max_index[1]

            if w_class_id in required_class_index:
                if classes_scores[w_class_id] > .25:
                    w_confidences.append(w_confidence)

                    class_ids.append(w_class_id)

                    x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                    left = int((x - 0.5 * w) * x_factor)
                    top = int((y - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    w_box = np.array([left, top, width, height])
                    w_boxes.append(w_box)

    indexes = cv2.dnn.NMSBoxes(w_boxes, w_confidences, confThreshold, nmsThreshold)

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(w_confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(w_boxes[i])

    return result_class_ids, result_confidences, result_boxes


def format_yolov5(f):
    row, col, _ = f.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = f
    return result


def process():
    colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]

    is_cuda = len(sys.argv) > 1 and sys.argv[1] == "cuda"

    net = build_model(is_cuda)
    capture = load_capture()

    start = time.time_ns()
    frame_count = 0
    total_frames = 0
    fps = -1

    while True:

        _, frame = capture.read()
        if frame is None:
            print("End of stream")
            break

        input_image = format_yolov5(frame)
        outs = detect(input_image, net)

        classes_ids, confidences, boxes = wrap_detection(input_image, outs[0])

        frame_count += 1
        total_frames += 1

        for (class_id, confidence, box) in zip(classes_ids, confidences, boxes):
            color = colors[int(class_id) % len(colors)]
            cv2.rectangle(frame, box, color, 2)
            cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)

            label = f'{class_list[class_id]} {int(confidence * 100)}%'

            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))

        if frame_count >= 30:
            end = time.time_ns()
            fps = 1000000000 * frame_count / (end - start)
            frame_count = 0
            start = time.time_ns()

        if fps > 0:
            fps_label = "FPS: %.2f" % fps
            cv2.putText(frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        label_total = "Total detections: " + str(len(boxes))
        cv2.putText(frame, label_total, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 255), 2)

        cv2.imshow("output", frame)

        print("Total frames: " + str(total_frames))

        if cv2.waitKey() == ord('q'):
            break


def now():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")


if __name__ == '__main__':
    print('Start process at ', now())
    process()
    print('Finish process at ', now())
