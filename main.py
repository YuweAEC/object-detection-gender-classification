import cv2
import numpy as np
import tensorflow as tf

# Load YOLO
net = cv2.dnn.readNet("yolov3/yolov3.weights", "yolov3/yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
classes = open("yolov3/coco.names").read().strip().split("\n")

# Load pre-trained gender classification model
gender_model = tf.keras.models.load_model("model/gender_model.h5")

def predict_gender(face):
    face_resized = cv2.resize(face, (64, 64))
    face_normalized = face_resized / 255.0
    face_reshaped = np.reshape(face_normalized, (1, 64, 64, 3))
    gender = gender_model.predict(face_reshaped)[0][0]
    return "Male" if gender < 0.5 else "Female"

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    height, width, channels = frame.shape

    # YOLO Object Detection
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label == "person":
                face = frame[y:y+h, x:x+w]
                gender = predict_gender(face)
                label = f"{label} ({gender})"

            color = (0, 255, 0) if label.startswith("person") else (255, 0, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Object Detection & Gender Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
