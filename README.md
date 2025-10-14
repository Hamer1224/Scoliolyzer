# Automated Scoliosis Cobb Angle Tool

An advanced desktop application that uses the **YOLOv4-tiny** object detection model to **automatically detect vertebrae** and calculate the Cobb angle from spinal X-ray images.



---

## Overview

This tool streamlines scoliosis assessment by replacing manual point selection with a powerful deep learning model. The application processes a given X-ray, identifies the key vertebrae, and instantly computes the Cobb angle, providing a fast and objective measurement.

## Features

* **🧠 Automated Vertebrae Detection:** Employs a pre-trained YOLOv4-tiny model to instantly locate vertebrae in the spinal column.
* **📐 Automatic Cobb Angle Calculation:** An algorithm analyzes the detected vertebrae to identify the curve's endplates and calculate the angle—no manual clicks needed.
* **🖼️ Rich Visual Feedback:** Overlays bounding boxes on detected vertebrae and draws the Cobb angle lines and value directly onto the image.
* **✅ Simple Workflow:** Just load an image and let the AI do the work.

---

## How It Works

The application follows a simple yet powerful pipeline:
1.  **Image Input:** The user loads a spinal X-ray through the GUI.
2.  **AI Detection:** The image is passed to the YOLOv4-tiny model, which returns the coordinates for all detected vertebrae.
3.  **Angle Calculation:** A post-processing algorithm determines the most tilted upper and lower vertebrae from the detected set and calculates the Cobb angle based on their orientation.
4.  **Display Results:** The original image is presented with the vertebrae boxes, measurement lines, and the final Cobb angle clearly displayed.

---

## Installation

To get this application running on your local machine, follow these steps.

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Download YOLOv4-tiny Model Files:**
    You need the pre-trained model weights and configuration files. Download `yolov4-tiny.weights`, `yolov4-tiny.cfg`, and `obj.names` and place them in the `model/` directory of the project.
    *(You can add a link to your model files here)*

3.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

4.  **Install Dependencies:**
    This project requires libraries for deep learning and image processing. Install them from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```


---

## How to Use

1.  **Run the Application:**
    Execute the main Python script from your terminal.
    ```bash
    python main.py
    ```
2.  **Load an Image:**
    Click the **"Load Image"** button and select a spinal X-ray file.
3.  **View the Results:**
    The application will automatically process the image. The detected vertebrae and the calculated Cobb angle will appear on the screen in moments.

---

## ⚠️ Medical Disclaimer

**This tool is for educational and research purposes only.** It is **NOT** a certified medical device and should **NOT** be used for primary diagnosis, treatment decisions, or any other clinical purposes. AI models can make mistakes. Always consult a qualified healthcare professional for any medical advice or diagnosis.

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for more details.
