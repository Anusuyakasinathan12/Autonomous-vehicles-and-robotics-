import cv2
import torch

# Load YOLOv5s model from torch hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)

def detect_objects(frame):
    results = model(frame)  # Perform detection
    results.print()  # Print detected objects to the console
    # Extract results (boxes, labels, etc.)
    boxes = results.xyxy[0].cpu().numpy()  # Detected boxes (x1, y1, x2, y2)
    labels = results.names  # Labels of detected objects
    return boxes, labels

def simulate_sensor(distance_cm):
    if distance_cm < 20:
        return "STOP"
    else:
        return "GO"

cap = cv2.VideoCapture(0)
print("Starting Autonomous Vehicle Simulation. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Simulate sensor (for now, we're always detecting an obstacle within 15 cm)
    decision = simulate_sensor(distance_cm=15)

    # Draw decision ("STOP" or "GO") on the frame
    if decision == "STOP":
        cv2.putText(frame, 'STOP: Obstacle Detected!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'GO: Path is Clear', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Detect objects only if the path is clear
    if decision == "GO":
        boxes, labels = detect_objects(frame)

        # Draw bounding boxes for detected objects
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw rectangle
            label = labels[int(box[5])]  # Get label of detected object
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame with detections and decision
    cv2.imshow('Autonomous Vehicle Simulation', frame)

    # Quit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
