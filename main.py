

import cv2
import numpy as np
import math
import os
from ultralytics import YOLO
from datetime import datetime
from tkinter import Tk, filedialog, messagebox, Button, Label, Canvas, Frame, BOTH
from PIL import Image, ImageTk

class ScoliosisAnalyzer:
    def __init__(self, master):
        self.master = master
        self.master.title("Scoliosis Analyzer")
        self.master.geometry("900x1100")

        self.top_frame = Frame(master)
        self.top_frame.pack(fill=BOTH, padx=10, pady=10)

        self.label = Label(self.top_frame, text="Select an X-ray image")
        self.label.pack(side="left", padx=5)

        self.button = Button(self.top_frame, text="Browse", command=self.load_image)
        self.button.pack(side="left", padx=5)

        self.canvas = Canvas(master, width=800, height=1000)
        self.canvas.pack(padx=10, pady=10)

        self.model = YOLO("v2.pt")
        self.tk_img = None

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")])
        if not file_path:
            return
        try:
            image = cv2.imread(file_path)
            results = self.model.predict(source=file_path, conf=0.4)[0]
            boxes = results.boxes.xyxy.cpu().numpy()

            vertebrae = []
            for box in boxes:
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                vertebrae.append({"box": box, "position": (cx, cy)})

            vertebrae = self.assign_anatomical_labels(vertebrae)

            for v in vertebrae:
                x1, y1, x2, y2 = map(int, v["box"])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cx, cy = map(int, v["position"])
                cv2.putText(image, v["id"], (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.circle(image, (cx, cy), 4, (0, 255, 255), -1)

            self.draw_vertical_reference_from_s1(image, vertebrae)
            angle = self.calculate_cobb_angle(vertebrae, image)

            if angle:
                cv2.putText(image, f"Cobb Angle: {angle:.2f} deg", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            bgr_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(bgr_image)
            img.thumbnail((800, 1000))
            self.tk_img = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor='nw', image=self.tk_img)
            self.canvas.config(scrollregion=self.canvas.bbox("all"))

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def assign_anatomical_labels(self, vertebrae):
        labels = ["S1"] + [f"L{i+5}" for i in reversed(range(5))] + [f"T{i+1}" for i in reversed(range(12))] + [f"C{i+1}" for i in reversed(range(7))]
        vertebrae = sorted(vertebrae, key=lambda v: v["position"][1], reverse=True)  # start from bottom
        for i, v in enumerate(vertebrae):
            v["id"] = labels[i] if i < len(labels) else f"V{i+1}"
        return vertebrae

    def draw_vertical_reference_from_s1(self, image, vertebrae):
        s1 = next((v for v in vertebrae if v.get("id") == "S1"), None)
        if s1:
            x, y = map(int, s1["position"])
            height = image.shape[0]
            cv2.line(image, (x, y), (x, 0), (0, 0, 255), 2)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(image, "S1", (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    def calculate_angle_between_lines(self, p1, p2, p3, p4):
        a1 = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        a2 = math.atan2(p4[1] - p3[1], p4[0] - p3[0])
        angle = abs(math.degrees(a2 - a1))
        if angle > 180:
            angle = 360 - angle
        return angle

    def calculate_cobb_angle(self, vertebrae, image=None):
        if len(vertebrae) < 4:
            return None

        max_angle = 0
        selected = (None, None)

        for i in range(len(vertebrae) - 2):
            pt1a = vertebrae[i]["position"]
            pt1b = vertebrae[i+1]["position"]
            pt2a = vertebrae[i+1]["position"]
            pt2b = vertebrae[i+2]["position"]

            angle = self.calculate_angle_between_lines(pt1a, pt1b, pt2a, pt2b)
            if angle > max_angle:
                max_angle = angle
                selected = (i, i + 2)

        if selected[0] is not None and image is not None:
            pt1 = tuple(map(int, vertebrae[selected[0]]['position']))
            pt2 = tuple(map(int, vertebrae[selected[0]+1]['position']))
            pt3 = tuple(map(int, vertebrae[selected[1]-1]['position']))
            pt4 = tuple(map(int, vertebrae[selected[1]]['position']))

            cv2.line(image, pt1, pt2, (0, 255, 255), 2)
            cv2.line(image, pt3, pt4, (0, 255, 255), 2)

        return max_angle if selected[0] is not None else None


if __name__ == '__main__':
    root = Tk()
    app = ScoliosisAnalyzer(root)
    root.mainloop()
