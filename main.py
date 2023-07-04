import cv2

net = cv2.dnn.readNet("model/yolov4-tiny.weights", "model/yolov4-tiny.cfg")
model = cv2.dnn.DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1 / 255)

classes = []
with open("model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    class_names, scores, bound_boxes = model.detect(frame=frame)

    for class_name, score, bound_box in zip(class_names, scores, bound_boxes):
        x, y, w, h = bound_box
        class_name = classes[class_name]
        cv2.putText(
            frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2
        )
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 3)

    # print(class_ids)
    # print(scores)
    # print(box)

    cv2.imshow("Object Detection", frame)
    cv2.waitKey(1)
    if cv2.getWindowProperty("Object Detection", cv2.WND_PROP_VISIBLE) < 1:
        break
cv2.destroyAllWindows()
