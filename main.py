#AVSM visiting card management system



#Improvment 10
import os
import logging
import cv2
import pytesseract
from PIL import Image
import pandas as pd
import easyocr
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

class VisitingCardScannerApp:
    def __init__(self, master):
        self.master = master
        master.title("Visiting Card Scanner")
        master.geometry("900x500")  
        master.configure(bg="#f0f0f0")  

        self.capture_button = tk.Button(master, text="Capture Visiting Card", command=self.capture_visiting_card, bg="#4CAF50", fg="white")
        self.capture_button.pack(pady=10)

        self.save_text_button = tk.Button(master, text="Save Extracted Text", command=self.save_text, bg="#008CBA", fg="white")
        self.save_text_button.pack(pady=5)

        self.exit_button = tk.Button(master, text="Exit", command=master.quit, bg="#f44336", fg="white")
        self.exit_button.pack(pady=5)

        self.extracted_text_tesseract = tk.StringVar()
        self.extracted_text_easyocr = tk.StringVar()

        self.tesseract_label = tk.Label(master, textvariable=self.extracted_text_tesseract, wraplength=580, justify="left", bg="#f0f0f0")
        self.tesseract_label.pack(pady=10)

        self.easyocr_label = tk.Label(master, textvariable=self.extracted_text_easyocr, wraplength=580, justify="left", bg="#f0f0f0")
        self.easyocr_label.pack(pady=10)

    def capture_visiting_card(self):
        try:
            cap = cv2.VideoCapture(0)
            cv2.waitKey(1000)
            while True:
                ret, frame = cap.read()
                cv2.imshow("Visiting Card Preview", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            ret, frame = cap.read()
            cap.release()
            cv2.destroyAllWindows()

            captured_image = Image.fromarray(frame)

            extracted_text_tesseract = self.extract_text_with_tesseract(captured_image)
            self.extracted_text_tesseract.set("Tesseract Extracted Text:\n" + extracted_text_tesseract)

            extracted_text_easyocr = self.extract_text_with_easyocr(captured_image)
            self.extracted_text_easyocr.set("EasyOCR Extracted Text:\n" + extracted_text_easyocr)

        except Exception as e:
            logging.error("An error occurred: %s", str(e))

    def extract_text_with_tesseract(self, image):
        try:
            preprocessed_image = self.preprocess_image(image)
            if preprocessed_image:
                extracted_text = pytesseract.image_to_string(preprocessed_image, lang='eng', config='--psm 6')
                return extracted_text.strip()
            else:
                return ""
        except Exception as e:
            logging.error("Error extracting text with Tesseract: %s", str(e))
            return ""

    def extract_text_with_easyocr(self, image):
        try:
            reader = easyocr.Reader(['en'])
            result = reader.readtext(np.array(image))
            extracted_text = ' '.join([text[1] for text in result])
            return extracted_text.strip()
        except Exception as e:
            logging.error("Error extracting text with EasyOCR: %s", str(e))
            return ""

    def preprocess_image(self, image):
        try:
            grayscale_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
            blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)
            thresholded_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            inverted_image = cv2.bitwise_not(thresholded_image)
            processed_image = Image.fromarray(inverted_image)
            return processed_image
        except Exception as e:
            logging.error("Error preprocessing image: %s", str(e))
            return None

    def save_text(self):
        try:
            tesseract_text = self.extracted_text_tesseract.get().split('\n')[1].strip()
            easyocr_text = self.extracted_text_easyocr.get().split('\n')[1].strip()

            # Display a custom dialog box with TesseractOCR and EasyOCR options
            choice = messagebox.askyesno("Choose OCR Text", "Save TesseractOCR text?")

            if choice:  # Save Tesseract text
                extracted_text = tesseract_text
            else:  # Save EasyOCR text
                extracted_text = easyocr_text

            
            excel_path = "extracted_data.xlsx"
            if os.path.exists(excel_path):
                existing_data = pd.read_excel(excel_path)
            else:
                existing_data = pd.DataFrame(columns=["Extracted Text"])

            new_data = pd.DataFrame({"Extracted Text": [extracted_text]})
            updated_data = pd.concat([existing_data, new_data], ignore_index=True)
            updated_data.to_excel(excel_path, index=False)

            messagebox.showinfo("Save Complete", "Chosen OCR text saved successfully.")
        except Exception as e:
            logging.error("An error occurred while saving text: %s", str(e))

root = tk.Tk()
app = VisitingCardScannerApp(root)
root.mainloop()





# including deep learning


# import os
# import logging
# import cv2
# import pytesseract
# from PIL import Image
# import pandas as pd
# import easyocr
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from sklearn.model_selection import train_test_split

# # Setup logging configuration
# logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# # Set the path to the Tesseract OCR executable
# TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# def preprocess_image(image: Image.Image) -> Image.Image:
#     """
#     Preprocesses the input image for better OCR accuracy.

#     :param image: The input image in PIL format.
#     :return: The preprocessed image in PIL format.
#     """
#     try:
#         # Convert the image to grayscale
#         grayscale_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)

#         # Apply Gaussian blurring to reduce noise
#         blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)

#         # Apply adaptive thresholding to binarize the image
#         thresholded_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

#         # Invert the binarized image
#         inverted_image = cv2.bitwise_not(thresholded_image)

#         # Convert the processed image back to PIL format
#         processed_image = Image.fromarray(inverted_image)

#         return processed_image
#     except Exception as e:
#         logging.error("Error preprocessing image: %s", str(e))
#         return None

# def generate_alphabet_dataset(image_size: tuple, num_samples_per_class: int) -> tuple:
#     """
#     Generates a synthetic dataset of alphabet images.

#     :param image_size: The size of the generated images.
#     :param num_samples_per_class: The number of samples per alphabet class.
#     :return: A tuple containing the generated images and corresponding labels.
#     """
#     alphabet = "abcdefghijklmnopqrstuvwxyz"
#     num_classes = len(alphabet)

#     X = []
#     y = []

#     for i, char in enumerate(alphabet):
#         for _ in range(num_samples_per_class):
#             image = np.random.rand(*image_size)  # Generate random image (replace with actual image loading)
#             X.append(image)
#             y.append(i)  # Assign label to each image

#     X = np.array(X)
#     y = np.array(y)

#     return X, y

# def train_text_recognition_model(X_train, y_train, image_size, num_classes):
#     """
#     Trains a basic deep learning model for text recognition.

#     :param X_train: The training images.
#     :param y_train: The training labels.
#     :param image_size: The size of the images.
#     :param num_classes: The number of classes.
#     :return: The trained model.
#     """
#     model = keras.Sequential([
#         keras.layers.Flatten(input_shape=image_size),  # Flatten input images
#         keras.layers.Dense(128, activation='relu'),    # Add a dense layer
#         keras.layers.Dense(num_classes, activation='softmax')  # Output layer with softmax activation
#     ])

#     model.compile(optimizer='adam',
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])

#     model.fit(X_train, y_train, epochs=10)

#     return model

# def extract_text_with_dl(image: Image.Image, model) -> str:
#     """
#     Extracts text from the input image using a trained deep learning model.

#     :param image: The input image in PIL format.
#     :param model: The trained deep learning model.
#     :return: The extracted text as a string.
#     """
#     try:
#         preprocessed_image = preprocess_image(image)
#         if preprocessed_image:
#             resized_image = preprocessed_image.resize((32, 32))  # Resize image to match model input size
#             normalized_image = np.array(resized_image) / 255.0  # Normalize image
#             normalized_image = normalized_image.reshape(1, *normalized_image.shape)  # Reshape for model input
#             prediction = model.predict(normalized_image)
#             predicted_class = np.argmax(prediction)
#             alphabet = "abcdefghijklmnopqrstuvwxyz"
#             extracted_text = alphabet[predicted_class]
#             return extracted_text
#         else:
#             return ""
#     except Exception as e:
#         logging.error("Error extracting text with deep learning model: %s", str(e))
#         return ""

# def save_image_and_text(image: Image.Image, extracted_text_tesseract: str, extracted_text_dl: str):
#     """
#     Saves the image and extracted text to disk and logs the process.

#     :param image: The input image in PIL format.
#     :param extracted_text_tesseract: The extracted text from Tesseract OCR.
#     :param extracted_text_dl: The extracted text from the deep learning model.
#     """
#     sorted_folder = "SortedImages"
#     if not os.path.exists(sorted_folder):
#         os.makedirs(sorted_folder)

#     image_path = os.path.join(sorted_folder, "visiting_card.jpg")
#     image.save(image_path)

#     excel_path = os.path.join(sorted_folder, "extracted_data.xlsx")
#     if os.path.exists(excel_path):
#         existing_data = pd.read_excel(excel_path)
#     else:
#         existing_data = pd.DataFrame(columns=["File Name", "Tesseract Extracted Text", "DL Extracted Text"])

#     new_data = pd.DataFrame({"File Name": [os.path.basename(image_path)], "Tesseract Extracted Text": [extracted_text_tesseract], "DL Extracted Text": [extracted_text_dl]})
#     updated_data = pd.concat([existing_data, new_data], ignore_index=True)
#     updated_data.to_excel(excel_path, index=False)

#     logging.info("Image and extracted text saved to SortedImages folder.")
#     logging.info("Extracted text saved to Excel: %s", excel_path)

# if __name__ == "__main__":
#     try:
#         # Generate synthetic dataset
#         image_size = (32, 32)  # Adjust image size as needed
#         num_samples_per_class = 1000
#         X, y = generate_alphabet_dataset(image_size, num_samples_per_class)
#         num_classes = len(np.unique(y))

#         # Split dataset into training and testing sets
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         # Train text recognition model
#         model = train_text_recognition_model(X_train, y_train, image_size, num_classes)

#         # Capture visiting card
#         cap = cv2.VideoCapture(0)
#         cv2.waitKey(1000)
#         while True:
#             ret, frame = cap.read()
#             cv2.imshow("Visiting Card Preview", frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#         ret, frame = cap.read()
#         cap.release()
#         cv2.destroyAllWindows()
#         image = Image.fromarray(frame)

#         # Extract text using Tesseract OCR
#         extracted_text_tesseract = pytesseract.image_to_string(image, lang='eng', config='--psm 6')

#         # Extract text using deep learning model
#         extracted_text_dl = extract_text_with_dl(image, model)

#         # Save image and extracted text
#         save_image_and_text(image, extracted_text_tesseract, extracted_text_dl)

#     except Exception as e:
#         logging.error("An error occurred: %s", str(e))