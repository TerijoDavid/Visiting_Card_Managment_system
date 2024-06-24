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
