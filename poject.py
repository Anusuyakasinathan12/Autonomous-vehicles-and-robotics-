# ====== LANE DETECTION USING OPENCV ======
import cv2
import numpy as np

def detect_lanes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)
    edges = cv2.Canny(blur, 50, 150)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, maxLineGap=50)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=3)
    return image

# ====== OBJECT DETECTION USING YOLOv5 ======
import torch

# Load YOLOv5 model
model = torch.hub.load(repo_or_dir='ultralytics/yolov5', model='yolov5s')

def detect_objects(image_path):
    results = model(image_path)
    results.print()
    results.show()
    # results.save()  # Uncomment to save results

# ====== BASIC PID CONTROLLER ======
class PIDController:
    def __init__(self, Kp, K1, Kd):
        self.Kp = Kp
        self.K1 = K1
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0

    def update(self, setpoint, current):
        error = setpoint - current
        self.integral += error
        derivative = error - self.prev_error
        output = (self.Kp * error) + (self.K1 * self.integral) + (self.Kd * derivative)
        self.prev_error = error
        return output

# ====== SENSOR FUSION WITH KALMAN FILTER ======
def kalman_filter(z, x_prev, P_prev, A, H, Q, R):
    # Prediction
    x_pred = A @ x_prev
    P_pred = A @ P_prev @ A.T + Q

    # Update
    K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
    x_new = x_pred + K @ (z - H @ x_pred)
    P_new = (np.eye(K.shape[0]) - K @ H) @ P_pred

    return x_new, P_new
