from tensorflow import keras
import tkinter as tk
import numpy as np
from tkinter import filedialog
from PIL import Image, ImageTk


# 1. Load model
fpath = './models/mnist_number_predictor_model28_50epochs.keras'
model = keras.models.load_model(fpath)

# 2. Load image in GUI
image_label = None
output_label = None
keras_input_image = None
prediction = None


def upload_image():
    global image_label, output_label, keras_input_image, prediction

    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
    )

    if file_path:
        # Load image and give PNG backgrounds
        image = Image.open(file_path).convert("RGBA")
        background = Image.new("RGB", image.size, (255, 255, 255))
        image = Image.alpha_composite(background.convert("RGBA"), image)
        image.thumbnail((28, 28))

        # --- Preprocess for Keras ---
        # Resize to model input size (e.g., 28x28)
        image = image.convert("L")
        keras_image = image.resize((28, 28))
        keras_array = np.array(keras_image) / 255.0  # normalize to 0-1
        keras_input_image = np.expand_dims(1.0 - keras_array, axis=(0, -1))  # shape (1, 28, 28, 1)

        # Setup display image
        display_array = (255 * (1.0 - keras_array)).astype(np.uint8)
        display_image = Image.fromarray(display_array)
        photo = ImageTk.PhotoImage(display_image)
        image_label.config(image=photo)
        image_label.image = photo  # Keep reference

        # Predict image
        prediction = model.predict(keras_input_image)
        print(prediction)
        print(np.argmax(prediction, axis=1))

        output_label.config(text=f"Prediction: {np.argmax(prediction, axis=1)}")


# ---- Main GUI Setup ----
root = tk.Tk()
root.title("Image Uploader for Keras")
root.geometry("500x500")

tk.Label(root, text="Upload an image for Keras model", font=("Arial", 14)).pack(pady=20)

tk.Button(root, text="Upload Image", command=upload_image).pack(pady=10)

image_label = tk.Label(root)
image_label.pack()

output_label = tk.Label(root, text="", font=("Arial", 14))
output_label.pack(pady=20)

root.mainloop()

print(prediction)
