import torch
from PIL import Image
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', 'best.pt', True)

def detect_objects():
    # Load and preprocess input image
    image_path = filedialog.askopenfilename()
    img = Image.open(image_path)

    # Perform object detection inference
    results = model(img)

    # Display detected objects on the original image
    labeled_image = results.render()[0]

    # Convert PIL image to Tkinter PhotoImage
    labeled_image_tk = ImageTk.PhotoImage(labeled_image)

    # Update the label with the new image
    result_label.config(image=labeled_image_tk)
    result_label.image = labeled_image_tk

# Create a Tkinter window
root = tk.Tk()
root.title("Blood Detection App")

# Create a button to trigger object detection
detect_button = tk.Button(root, text="Detect Objects", command=detect_objects)
detect_button.pack(pady=10)

# Create a label to display the resulting image
result_label = tk.Label(root)
result_label.pack()

# Run the Tkinter event loop
root.mainloop()
