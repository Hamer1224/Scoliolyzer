# scoliosis_improved_gui.py

# Improved script with proper Cobb angle calculation for both C and S type scoliosis

import cv2
import numpy as np
import math
import os
from ultralytics import YOLO
from datetime import datetime
from tkinter import Tk, filedialog, messagebox, Button, Label, Canvas, Frame, BOTH, Text, Scrollbar
from PIL import Image, ImageTk
from scipy import signal
from scipy.interpolate import UnivariateSpline

class ScoliosisAnalyzer:
    def __init__(self, master):
        self.master = master
        self.master.title("Advanced Scoliosis Analyzer")
        self.master.geometry("1200x1100")

        self.top_frame = Frame(master)
        self.top_frame.pack(fill=BOTH, padx=10, pady=10)

        self.label = Label(self.top_frame, text="Select an X-ray image")
        self.label.pack(side="left", padx=5)

        self.button = Button(self.top_frame, text="Browse", command=self.load_image)
        self.button.pack(side="left", padx=5)

        # Create main frame for canvas and results
        self.main_frame = Frame(master)
        self.main_frame.pack(fill=BOTH, expand=True, padx=10, pady=5)

        # Canvas for image
        self.canvas = Canvas(self.main_frame, width=800, height=900)
        self.canvas.pack(side="left", padx=(0, 10))

        # Results text area
        self.results_frame = Frame(self.main_frame)
        self.results_frame.pack(side="right", fill="both", expand=True)
        
        Label(self.results_frame, text="Analysis Results:", font=("Arial", 12, "bold")).pack(anchor="w")
        
        self.results_text = Text(self.results_frame, width=40, height=30, wrap="word")
        scrollbar = Scrollbar(self.results_frame, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        self.results_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

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
                width, height = x2 - x1, y2 - y1
                vertebrae.append({
                    "box": box, 
                    "position": (cx, cy),
                    "width": width,
                    "height": height
                })

            vertebrae = self.assign_anatomical_labels(vertebrae)
            
            # Clear previous results
            self.results_text.delete(1.0, "end")
            
            # Draw vertebrae detection
            for v in vertebrae:
                x1, y1, x2, y2 = map(int, v["box"])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cx, cy = map(int, v["position"])
                cv2.putText(image, v["id"], (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.circle(image, (cx, cy), 3, (0, 255, 255), -1)

            # Calculate spine curvature and detect curves
            spine_analysis = self.analyze_spine_curvature(vertebrae, image)
            
            # Draw reference line
            self.draw_vertical_reference_from_s1(image, vertebrae)
            
            # Calculate Cobb angles for detected curves
            cobb_results = self.calculate_multiple_cobb_angles(vertebrae, spine_analysis, image)
            
            # Display results
            self.display_results(spine_analysis, cobb_results)

            # Convert and display image
            bgr_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(bgr_image)
            img.thumbnail((800, 900))
            self.tk_img = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor='nw', image=self.tk_img)
            self.canvas.config(scrollregion=self.canvas.bbox("all"))

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def assign_anatomical_labels(self, vertebrae):
        # More comprehensive labeling including sacral vertebrae
        labels = (["S1", "S2"] + 
                 [f"L{i}" for i in range(5, 0, -1)] + 
                 [f"T{i}" for i in range(12, 0, -1)] + 
                 [f"C{i}" for i in range(7, 0, -1)])
        
        # Sort from bottom to top (highest y to lowest y)
        vertebrae = sorted(vertebrae, key=lambda v: v["position"][1], reverse=True)
        
        for i, v in enumerate(vertebrae):
            v["id"] = labels[i] if i < len(labels) else f"V{i+1}"
            v["level"] = i  # Add level for easier processing
        
        return vertebrae

    def analyze_spine_curvature(self, vertebrae, image):
        """Analyze the spine curvature to detect curve patterns"""
        if len(vertebrae) < 6:
            return {"curve_type": "insufficient_data", "curves": []}
        
        # Extract x-coordinates (lateral deviation from vertical)
        positions = [v["position"] for v in sorted(vertebrae, key=lambda x: x["level"])]
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        # Create a reference vertical line (median x position)
        ref_x = np.median(x_coords)
        deviations = [x - ref_x for x in x_coords]
        
        # Smooth the curve to reduce noise
        if len(deviations) >= 4:
            # Use spline smoothing
            levels = list(range(len(deviations)))
            spline = UnivariateSpline(levels, deviations, s=len(deviations))
            smoothed_deviations = spline(levels)
        else:
            smoothed_deviations = deviations
        
        # Find peaks and valleys (curve apexes)
        peaks, _ = signal.find_peaks(np.abs(smoothed_deviations), height=5, distance=3)
        
        # Determine curve type and regions
        curve_analysis = self.classify_curve_pattern(smoothed_deviations, peaks, vertebrae)
        
        return curve_analysis

    def classify_curve_pattern(self, deviations, peaks, vertebrae):
        """Classify the scoliosis curve pattern"""
        significant_peaks = []
        
        for peak in peaks:
            if abs(deviations[peak]) > 10:  # Minimum deviation threshold
                significant_peaks.append({
                    "index": peak,
                    "deviation": deviations[peak],
                    "vertebra": vertebrae[peak]["id"] if peak < len(vertebrae) else "Unknown"
                })
        
        if len(significant_peaks) == 0:
            return {"curve_type": "normal", "curves": []}
        elif len(significant_peaks) == 1:
            return {
                "curve_type": "C-type (single curve)",
                "curves": [self.define_curve_region(significant_peaks[0], vertebrae, deviations)]
            }
        else:
            # Multiple curves - likely S-type
            curves = []
            for peak in significant_peaks:
                curves.append(self.define_curve_region(peak, vertebrae, deviations))
            
            return {
                "curve_type": f"S-type ({len(significant_peaks)} curves)",
                "curves": curves
            }

    def define_curve_region(self, peak_info, vertebrae, deviations):
        """Define the region of a curve for Cobb angle calculation"""
        peak_idx = peak_info["index"]
        
        # Find the neutral vertebrae (where curve changes direction)
        # Look backwards from peak
        upper_neutral = 0
        for i in range(peak_idx - 1, -1, -1):
            if abs(deviations[i]) < abs(deviations[i + 1]):
                upper_neutral = i
                break
        
        # Look forwards from peak
        lower_neutral = len(deviations) - 1
        for i in range(peak_idx + 1, len(deviations)):
            if abs(deviations[i]) < abs(deviations[i - 1]):
                lower_neutral = i
                break
        
        return {
            "apex_vertebra": vertebrae[peak_idx]["id"],
            "apex_index": peak_idx,
            "upper_end_vertebra": vertebrae[upper_neutral]["id"],
            "upper_end_index": upper_neutral,
            "lower_end_vertebra": vertebrae[lower_neutral]["id"],
            "lower_end_index": lower_neutral,
            "deviation": peak_info["deviation"],
            "direction": "right" if peak_info["deviation"] > 0 else "left"
        }

    def calculate_multiple_cobb_angles(self, vertebrae, spine_analysis, image):
        """Calculate Cobb angles for all detected curves"""
        cobb_results = []
        
        for i, curve in enumerate(spine_analysis["curves"]):
            angle = self.calculate_cobb_angle_for_curve(vertebrae, curve, image, i)
            if angle is not None:
                cobb_results.append({
                    "curve_number": i + 1,
                    "angle": angle,
                    "curve_info": curve
                })
        
        return cobb_results

    def calculate_cobb_angle_for_curve(self, vertebrae, curve, image, curve_number):
        """Calculate Cobb angle for a specific curve using proper endplate method"""
        try:
            upper_vertebra = next(v for v in vertebrae if v["id"] == curve["upper_end_vertebra"])
            lower_vertebra = next(v for v in vertebrae if v["id"] == curve["lower_end_vertebra"])
            
            # Get endplate lines for upper and lower vertebrae
            upper_endplate = self.get_vertebral_endplate_line(upper_vertebra, "upper")
            lower_endplate = self.get_vertebral_endplate_line(lower_vertebra, "lower")
            
            # Calculate angle between endplates
            angle = self.calculate_angle_between_lines(
                upper_endplate[0], upper_endplate[1],
                lower_endplate[0], lower_endplate[1]
            )
            
            # Draw the measurement lines on image
            color = [(255, 0, 0), (0, 255, 0), (255, 0, 255)][curve_number % 3]
            
            cv2.line(image, tuple(map(int, upper_endplate[0])), 
                    tuple(map(int, upper_endplate[1])), color, 2)
            cv2.line(image, tuple(map(int, lower_endplate[0])), 
                    tuple(map(int, lower_endplate[1])), color, 2)
            
            # Draw perpendicular lines for visualization
            self.draw_perpendicular_lines(image, upper_endplate, lower_endplate, color)
            
            # Label the measurement
            mid_y = (upper_vertebra["position"][1] + lower_vertebra["position"][1]) / 2
            cv2.putText(image, f"Curve {curve_number + 1}: {angle:.1f}°", 
                       (50, int(mid_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            return angle
            
        except Exception as e:
            print(f"Error calculating Cobb angle for curve {curve_number + 1}: {e}")
            return None

    def get_vertebral_endplate_line(self, vertebra, endplate_type):
        """Get the endplate line points for a vertebra"""
        x1, y1, x2, y2 = vertebra["box"]
        
        if endplate_type == "upper":
            # Upper endplate
            p1 = (x1, y1)
            p2 = (x2, y1)
        else:
            # Lower endplate
            p1 = (x1, y2)
            p2 = (x2, y2)
        
        return [p1, p2]

    def draw_perpendicular_lines(self, image, line1, line2, color):
        """Draw perpendicular lines to visualize the angle measurement"""
        # Calculate perpendiculars for better visualization
        center_x = image.shape[1] // 2
        
        # Get line directions
        dir1 = np.array(line1[1]) - np.array(line1[0])
        dir2 = np.array(line2[1]) - np.array(line2[0])
        
        # Normalize
        dir1 = dir1 / np.linalg.norm(dir1)
        dir2 = dir2 / np.linalg.norm(dir2)
        
        # Create perpendicular lines from a common point
        common_x = center_x
        y1 = line1[0][1] + (common_x - line1[0][0]) * dir1[1] / dir1[0] if dir1[0] != 0 else line1[0][1]
        y2 = line2[0][1] + (common_x - line2[0][0]) * dir2[1] / dir2[0] if dir2[0] != 0 else line2[0][1]
        
        # Draw perpendicular lines
        perp1 = np.array([-dir1[1], dir1[0]]) * 50
        perp2 = np.array([-dir2[1], dir2[0]]) * 50
        
        start1 = (int(common_x), int(y1))
        end1 = (int(common_x + perp1[0]), int(y1 + perp1[1]))
        
        start2 = (int(common_x), int(y2))
        end2 = (int(common_x + perp2[0]), int(y2 + perp2[1]))
        
        cv2.line(image, start1, end1, color, 1)
        cv2.line(image, start2, end2, color, 1)

    def calculate_angle_between_lines(self, p1, p2, p3, p4):
        """Calculate angle between two lines defined by points"""
        # Calculate vectors
        v1 = np.array(p2) - np.array(p1)
        v2 = np.array(p4) - np.array(p3)
        
        # Calculate angles with horizontal
        angle1 = math.atan2(v1[1], v1[0])
        angle2 = math.atan2(v2[1], v2[0])
        
        # Calculate difference
        angle_diff = abs(angle2 - angle1)
        angle_deg = math.degrees(angle_diff)
        
        # Ensure we get the acute angle
        if angle_deg > 90:
            angle_deg = 180 - angle_deg
            
        return angle_deg

    def draw_vertical_reference_from_s1(self, image, vertebrae):
        """Draw vertical reference line from S1"""
        s1 = next((v for v in vertebrae if v.get("id") == "S1"), None)
        if s1:
            x, y = map(int, s1["position"])
            height = image.shape[0]
            cv2.line(image, (x, y), (x, 0), (128, 128, 128), 1)
            cv2.circle(image, (x, y), 4, (0, 0, 255), -1)
            cv2.putText(image, "S1 Reference", (x + 10, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    def display_results(self, spine_analysis, cobb_results):
        """Display analysis results in the text widget"""
        results = f"SCOLIOSIS ANALYSIS RESULTS\n"
        results += "=" * 40 + "\n\n"
        
        results += f"Curve Type: {spine_analysis['curve_type']}\n\n"
        
        if not cobb_results:
            results += "No significant scoliotic curves detected.\n"
        else:
            results += f"Number of curves detected: {len(cobb_results)}\n\n"
            
            for result in cobb_results:
                curve_info = result["curve_info"]
                results += f"CURVE {result['curve_number']}:\n"
                results += f"  Cobb Angle: {result['angle']:.1f}°\n"
                results += f"  Direction: {curve_info['direction'].capitalize()}\n"
                results += f"  Upper End Vertebra: {curve_info['upper_end_vertebra']}\n"
                results += f"  Apex Vertebra: {curve_info['apex_vertebra']}\n"
                results += f"  Lower End Vertebra: {curve_info['lower_end_vertebra']}\n"
                results += f"  Severity: {self.classify_severity(result['angle'])}\n\n"
            
            # Overall classification
            max_angle = max(result['angle'] for result in cobb_results)
            results += f"OVERALL CLASSIFICATION:\n"
            results += f"  Maximum Cobb Angle: {max_angle:.1f}°\n"
            results += f"  Severity: {self.classify_severity(max_angle)}\n"
            
            if len(cobb_results) > 1:
                results += f"  Pattern: Double curve (S-type)\n"
            else:
                results += f"  Pattern: Single curve (C-type)\n"
        
        self.results_text.insert("1.0", results)

    def classify_severity(self, angle):
        """Classify scoliosis severity based on Cobb angle"""
        if angle < 10:
            return "Normal"
        elif angle < 25:
            return "Mild"
        elif angle < 40:
            return "Moderate"
        elif angle < 50:
            return "Severe"
        else:
            return "Very Severe"


if __name__ == '__main__':
    root = Tk()
    app = ScoliosisAnalyzer(root)
    root.mainloop()
