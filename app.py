from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2
import torch
import numpy as np

# Load YOLOv8 model
model = YOLO("models/best (3).pt")

app = Flask(__name__)

from threading import Thread

class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.cap.release()
                return
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

def gen_frames():
    vs = VideoStream().start()
    frame_rate = 5
    prev_time = 0

    while True:
        frame = vs.read()
        curr_time = cv2.getTickCount() / cv2.getTickFrequency()
        if curr_time - prev_time >= 1.0 / frame_rate:
            prev_time = curr_time

            frame = cv2.resize(frame, (640, 480))
            results = model(frame)
            results = results[0]

            for box in results.boxes:
                bbox = box.xyxy.numpy()[0]
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                label = f"{model.names[int(box.cls.item())]} {box.conf.item():.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run()
