import customtkinter as ctk
from tkinter import filedialog
from PIL import Image

class ScoliolyzerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Scoliolyzer - AI Scoliosis Analyzer")
        self.geometry("800x600")

        # Create a main frame
        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.pack(padx=20, pady=20, fill="both", expand=True)

        # Add a title label
        self.title_label = ctk.CTkLabel(self.main_frame, text="Scoliolyzer", font=ctk.CTkFont(size=24, weight="bold"))
        self.title_label.pack(pady=20)

        # Add a button to upload an image
        self.upload_button = ctk.CTkButton(self.main_frame, text="Upload X-ray Image", command=self.upload_image)
        self.upload_button.pack(pady=10)

        # Add a placeholder for the image
        self.image_label = ctk.CTkLabel(self.main_frame, text="X-ray image will be displayed here.", width=400, height=400, fg_color="gray")
        self.image_label.pack(pady=10)

        # Add a label to display the analysis result
        self.result_label = ctk.CTkLabel(self.main_frame, text="Analysis result: N/A", font=ctk.CTkFont(size=16))
        self.result_label.pack(pady=20)

    def upload_image(self):
        """
        Opens a file dialog to select an image and displays it in the UI.
        """
        filepath = filedialog.askopenfilename(
            title="Select an X-ray Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if not filepath:
            return

        # Display the selected image
        pil_image = Image.open(filepath)
        ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(400, 400))
        self.image_label.configure(image=ctk_image, text="")
        self.image_label.image = ctk_image  # Keep a reference

        # Placeholder for the analysis function
        self.analyze_image(filepath)

    def analyze_image(self, filepath):
        """
        Placeholder for the scoliosis analysis logic.
        This function will be updated to use the YOLOv4-tiny model.
        """
        # For now, just update the result label
        self.result_label.configure(text="Analysis result: Ready for YOLOv4-tiny model.")


if __name__ == "__main__":
    app = ScoliolyzerApp()
    app.mainloop()