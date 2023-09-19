
import cv2
import time
import torch
import torchvision.transforms as T


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# 初始化模型並加載權重
# 
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

model.eval()

# def detect_rat(frame):
#     img = T.ToTensor()(frame).unsqueeze(0)
#     preds = model(img)

#     boxes = preds.pred[0][:, :4]  # x1, y1, x2, y2
#     confidences = preds.pred[0][:, 4]
#     class_ids = preds.pred[0][:, 5].int()

#     rat_boxes = [box for i, box in enumerate(boxes) if class_ids[i] == 2 and confidences[i] > 0.5]

#     return rat_boxes
# def detect_rat(frame):
#     img = T.ToTensor()(frame).unsqueeze(0)
#     results = model(img)
#     detections = results.xyxy[0].cpu().numpy()

#     rat_boxes = [det[:4] for det in detections if int(det[5]) == 2 and det[4] > 0.5]

#     return rat_boxes
def detect_rat(frame):
    results = model(frame)
    # Extract detections corresponding to the rat class (assuming class id 2)
    rat_detections = [x for x in results.pred[0] if int(x[5]) == 2 and x[4] > 0.5]
    rat_boxes = [x[:4] for x in rat_detections]
    return rat_boxes



cap = cv2.VideoCapture(0)  # 使用內建鏡頭
cap.set(6,cv2.VideoWriter.fourcc('M','J','P','G'))
#cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)#比較快的

prev_detection_time = None
stationary_time = 0
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 640))

    if not ret:
        break

    rat_boxes = detect_rat(frame)

    for box in rat_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Rat detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    current_time = time.time()
    if prev_detection_time and (current_time - prev_detection_time < 2):  # 2 seconds threshold
        stationary_time += current_time - prev_detection_time

    prev_detection_time = current_time

cap.release()
cv2.destroyAllWindows()

print(f"Rat stationary time: {stationary_time} seconds")
