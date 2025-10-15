
import cv2
import numpy as np
import math
import customtkinter
from ultralytics import YOLO
from tkinter import Canvas, messagebox
from PIL import Image, ImageTk
import traceback

class ScoliosisAnalyzer:
    MIN_CURVATURE_THRESHOLD = 5.0  # Degrees. Below this, the spine is considered straight.

    def __init__(self, master):
        self.master = master
        self.master.title("Scoliosis Analyzer v10.1 (Fixed)")
        self.master.geometry("1000x800")

        # --- Class Variables ---
        self.tk_img = None
        self.processed_image = None
        self.original_filepath = None

        # --- Theme and Layout Configuration ---
        customtkinter.set_appearance_mode("System")
        customtkinter.set_default_color_theme("blue")
        self.master.grid_columnconfigure(0, weight=1)
        self.master.grid_rowconfigure(2, weight=1)

        # --- Title ---
        self.title_label = customtkinter.CTkLabel(master, text="Automated Cobb Angle Measurement", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.title_label.grid(row=0, column=0, padx=20, pady=(10, 0), sticky="ew")

        # --- Top Frame for Controls ---
        self.top_frame = customtkinter.CTkFrame(master, corner_radius=0)
        self.top_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
        self.top_frame.grid_columnconfigure(3, weight=1)

        self.browse_button = customtkinter.CTkButton(self.top_frame, text="Browse X-ray Image", command=self.load_image)
        self.browse_button.grid(row=0, column=0, padx=10, pady=10)

        self.save_button = customtkinter.CTkButton(self.top_frame, text="Save Results", command=self.save_image, state="disabled")
        self.save_button.grid(row=0, column=1, padx=10, pady=10)

        self.info_label = customtkinter.CTkLabel(self.top_frame, text="Load an image to begin analysis.")
        self.info_label.grid(row=0, column=2, padx=20, pady=10)

        self.theme_switch = customtkinter.CTkSwitch(self.top_frame, text="Dark Mode", command=self.toggle_theme)
        self.theme_switch.grid(row=0, column=4, padx=10, pady=10, sticky="e")

        # --- Canvas Frame ---
        self.canvas_frame = customtkinter.CTkFrame(master)
        self.canvas_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)

        self.canvas = Canvas(self.canvas_frame, bg="#2B2B2B" if customtkinter.get_appearance_mode().lower() == "dark" else "#DBDBDB", highlightthickness=0)
        self.v_scroll = customtkinter.CTkScrollbar(self.canvas_frame, command=self.canvas.yview)
        self.h_scroll = customtkinter.CTkScrollbar(self.canvas_frame, orientation="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.v_scroll.set, xscrollcommand=self.h_scroll.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.h_scroll.grid(row=1, column=0, sticky="ew")

        try:
            
            self.model = YOLO("v2.pt")
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load YOLO model 'v2.pt'.\nError: {e}")
            self.master.destroy()

    def toggle_theme(self):
        mode = "dark" if self.theme_switch.get() == 1 else "light"
        customtkinter.set_appearance_mode(mode)
        self.canvas.configure(bg="#2B2B2B" if mode == "dark" else "#DBDBDB")

    def load_image(self):
        self.original_filepath = customtkinter.filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")])
        if not self.original_filepath: return

        try:
            self.info_label.configure(text="Processing, please wait...")
            self.master.update_idletasks()
            image = cv2.imread(self.original_filepath)
            if image is None:
                messagebox.showerror("Error", f"Could not read image: {self.original_filepath}")
                self.info_label.configure(text="Failed to load image.")
                return
            self.process_image(image, self.original_filepath)
        except Exception as e:
            messagebox.showerror("Processing Error", f"An error occurred: {e}\n\n{traceback.format_exc()}")
            self.info_label.configure(text="An error occurred during processing.")

    def process_image(self, image, file_path):
        results = self.model.predict(source=file_path, conf=0.5)[0]
        boxes = results.boxes.xyxy.cpu().numpy()

        if len(boxes) < 4:
            messagebox.showwarning("Warning", "Not enough vertebrae detected for a reliable analysis.")
            self.info_label.configure(text="Analysis complete (Not enough vertebrae).")
            self.display_image(image)
            return

        vertebrae = [{"box": box, "center": ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)} for box in boxes]
        vertebrae = self.assign_anatomical_labels(vertebrae)

        for v in vertebrae:
            x1, y1, x2, y2 = map(int, v["box"])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, v["id"], (int(v["box"][0]) - 10, int(v["box"][1]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        self.draw_vertical_reference_from_s1(image, vertebrae)
        self.draw_plumb_line(image, vertebrae)

        cobb_data = self.find_cobb_angle(vertebrae, image)

        if cobb_data and 'angle' in cobb_data:
            angle = cobb_data['angle']
            cv2.putText(image, f"Cobb Angle: {angle:.2f} degrees", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            self.info_label.configure(text=f"Analysis Complete. Cobb Angle: {angle:.2f}°")
        elif cobb_data and 'message' in cobb_data:
            message = cobb_data['message']
            cv2.putText(image, message, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            self.info_label.configure(text=f"Analysis Complete. {message}")
        else:
            self.info_label.configure(text="Analysis complete. Could not calculate Cobb angle.")

        self.processed_image = image
        self.save_button.configure(state="normal")
        self.display_image(image)

    def assign_anatomical_labels(self, vertebrae):
       
        labels = ["S1"] + [f"L{i}" for i in range(5, 0, -1)] + [f"T{i}" for i in range(12, 0, -1)] + [f"C{i}" for i in range(7, 0, -1)]
        vertebrae.sort(key=lambda v: v["center"][1], reverse=True) 
        for i, v in enumerate(vertebrae):
            v["id"] = labels[i] if i < len(labels) else f"Unknown{i+1}"
        return vertebrae

    def get_local_tangent_angles(self, vertebrae, neighbor_span=1):
        """
        Her vertebra için komşu merkezler arasından lokal eğimi hesaplar.
        neighbor_span: kaç komşuyu kullanarak (i-1, i+1) gibi hesap yapılacağını belirler.
        Döndürülen açılar derece cinsindendir ve -180..180 aralığında olabilir.
        """
        centers = [v['center'] for v in vertebrae]
        n = len(centers)
        angles = np.zeros(n, dtype=float)

        for i in range(n):
           
            i_prev = max(0, i - neighbor_span)
            i_next = min(n - 1, i + neighbor_span)
            if i_prev == i_next:
                angles[i] = 0.0
                continue
            x_prev, y_prev = centers[i_prev]
            x_next, y_next = centers[i_next]
            dx = x_next - x_prev
            dy = y_next - y_prev
            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                angles[i] = 0.0
            else:
                ang = math.degrees(math.atan2(dy, dx))
                angles[i] = ang
        return angles

    def find_cobb_angle(self, vertebrae, image):
        try:
            if len(vertebrae) < 2: return None

            tangent_angles = self.get_local_tangent_angles(vertebrae, neighbor_span=1)

            tilt_range = np.max(tangent_angles) - np.min(tangent_angles)
            if tilt_range < self.MIN_CURVATURE_THRESHOLD:
                return {'message': f"No significant curve detected (< {self.MIN_CURVATURE_THRESHOLD}°)"}

           
            v_upper_idx = int(np.argmin(tangent_angles))  
            v_lower_idx = int(np.argmax(tangent_angles))

            
            if vertebrae[v_upper_idx]['center'][1] > vertebrae[v_lower_idx]['center'][1]:
                
                v_upper_idx, v_lower_idx = v_lower_idx, v_upper_idx

            if v_lower_idx == v_upper_idx:
                messagebox.showwarning("Analysis Info", "Could not identify distinct end vertebrae.")
                return None

            upper_v = vertebrae[v_upper_idx]
            lower_v = vertebrae[v_lower_idx]

            angle_upper = tangent_angles[v_upper_idx]
            angle_lower = tangent_angles[v_lower_idx]
            cobb_angle = abs(angle_upper - angle_lower)
            if cobb_angle > 90:
                cobb_angle = 180 - cobb_angle

            self.draw_line_through_center_with_angle(image, upper_v["center"], angle_upper, (255, 0, 255), 3, label="Upper End")
            self.draw_line_through_center_with_angle(image, lower_v["center"], angle_lower, (255, 0, 255), 3, label="Lower End")

            return {"angle": cobb_angle}
        except Exception as e:
            messagebox.showerror("Cobb Angle Error", f"Failed during Cobb angle calculation: {e}\n\n{traceback.format_exc()}")
            return None

    def draw_line_through_center_with_angle(self, img, center, angle_deg, color, thickness, label=None):
        """
        Verilen merkez ve açıya göre görüntü boyunca line çizer.
        angle_deg: derece
        """
        cx, cy = center
       
        angle_rad = math.radians(angle_deg)
        if abs(math.cos(angle_rad)) < 1e-3:
            x = int(round(cx))
            cv2.line(img, (x, 0), (x, img.shape[0]), color, thickness)
        else:
            m = math.tan(angle_rad)
            b = cy - m * cx
            x0 = 0
            y0 = int(round(m * x0 + b))
            x1 = img.shape[1]
            y1 = int(round(m * x1 + b))
            cv2.line(img, (x0, y0), (x1, y1), color, thickness)

        if label:
            try:
                cv2.putText(img, label, (int(cx) + 10, int(cy) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            except:
                pass

    def draw_vertical_reference_from_s1(self, image, vertebrae):
        s1 = next((v for v in vertebrae if v.get("id") == "S1"), None)
        if s1:
            cx, cy = map(int, s1["center"])
            for y in range(cy, 0, -20): cv2.line(image, (cx, y), (cx, y - 10), (255, 0, 0), 2)
            cv2.circle(image, (cx, cy), 8, (255, 0, 0), -1)

    def draw_plumb_line(self, image, vertebrae):
        highest_v = vertebrae[-1] if vertebrae else None
        if highest_v:
            cx, cy = map(int, highest_v["center"])
            for y in range(cy, image.shape[0], 20): cv2.line(image, (cx, y), (cx, y + 10), (0, 255, 255), 2)
            cv2.circle(image, (cx, cy), 8, (0, 255, 255), -1)
            cv2.putText(image, "Plumb Line", (cx + 15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def display_image(self, cv2_image):
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if w < 50 or h < 50: w, h = 800, 600

        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_image)
        img.thumbnail((w, h))

        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor='nw', image=self.tk_img)
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def save_image(self):
        if self.processed_image is None or self.original_filepath is None:
            messagebox.showwarning("Warning", "No processed image to save.")
            return

        original_path, original_ext = self.original_filepath.rsplit('.', 1)
        save_path = customtkinter.filedialog.asksaveasfilename(
            initialfile=f"{original_path.rsplit('/', 1)[-1]}_analyzed.{original_ext}",
            defaultextension=f".{original_ext}",
            filetypes=[("PNG file", "*.png"), ("JPG file", "*.jpg"), ("All files", "*.*")])

        if save_path:
            try:
                cv2.imwrite(save_path, self.processed_image)
                messagebox.showinfo("Success", f"Image successfully saved to:\n{save_path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save image.\nError: {e}")

if __name__ == '__main__':
    app = customtkinter.CTk()
    ScoliosisAnalyzer(app)
    app.mainloop()
