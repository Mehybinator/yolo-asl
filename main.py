from mediapipe import solutions as mp
from ultralytics import YOLO
import customtkinter as ctk
import pyautogui
import pyttsx3
import torch
import time
import cv2
import os

# Load the trained YOLOv8 model
wd = os.path.dirname(__file__)
model = YOLO(os.path.join(wd, 'runs/classify/train7/weights/best.pt'))  # Replace with your model path

# Initialize MediaPipe Hand detector
mp_hands = mp.hands.Hands()

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 175)
engine.setProperty('voice', engine.getProperty('voices')[1].id)

# Function to preprocess the frame
def preprocess_frame(frame, img_size=384):
    # Resize frame to the input size of the model
    frame = cv2.resize(frame, (img_size, img_size))
    # Convert the image from BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert the image to a PyTorch tensor and reorder to (channels, height, width)
    frame = torch.from_numpy(frame).float().permute(2, 0, 1)
    # Normalize the frame
    frame /= 255.0
    # Add a batch dimension
    frame = frame.unsqueeze(0)

    return frame

# Function to detect hand and return bounding box
def detect_hand(frame):
    results = mp_hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        return True
    return False

# Function to control multimedia using playerctl
def control_multimedia(action):
    if action == "C":
        pyautogui.press('playpause')
    elif action == "D":
        pyautogui.press('nexttrack')
    elif action == "E":
        pyautogui.press('prevtrack')

# Initialize variables for confidence system
start_time = None
predictions = []
interval = 1  # seconds
confidence_threshold = 0.7

perform = False

# Capture video from the webcam
cap = cv2.VideoCapture(0)

# Warm-up the model with a dummy input
dummy_input = torch.zeros((1, 3, 384, 384))
with torch.no_grad():
    model(dummy_input)

# Function to close the program
def close_program():
    cap.release()
    window.quit()
    window.destroy()

# Create a customtkinter window to show actions
ctk.set_appearance_mode("System")  # Modes: "System" (default), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (default), "green", "dark-blue"

window = ctk.CTk()
window.geometry("300x400")
window.title("ASL Classification")

label = ctk.CTkLabel(window, text="Possible Actions:")
label.pack(pady=10)

actions = {
    "C": "Play/Pause",
    "D": "Next Track",
    "E": "Previous Track"
}

for key, action in actions.items():
    action_label = ctk.CTkLabel(window, text=f"{key}: {action}")
    action_label.pack()

action_label = ctk.CTkLabel(window, text="Y: Toggle Action")
action_label.pack()

close_button = ctk.CTkButton(window, text="Close", command=close_program)
close_button.pack(pady=20)

# Main loop for video capture and processing
def process_frame():
    global start_time, predictions, perform, actions

    ret, frame = cap.read()
    if not ret:
        return

    # Detect hand
    hand_detected = detect_hand(frame)
    if hand_detected:
        if not start_time:
            start_time = time.time()

        input_frame = preprocess_frame(frame)

        # Perform inference
        with torch.no_grad():
            results = model(input_frame)
        
        pred_class = results[0].probs.top1

        # Assuming you have a list of class names
        class_names = ["C", "D", "E", "F", "I", "L", "O", "S", "V", "W", "Y"]  # Fill with your class names

        # Get the class name
        class_name = class_names[pred_class]

        # Add prediction to the list
        predictions.append(class_name)
        
        # Check for confidence interval
        if time.time() - start_time > interval:
            most_common = max(set(predictions), key=predictions.count)
            confidence = predictions.count(most_common) / len(predictions)
            
            if confidence > confidence_threshold:
                if perform:
                    if most_common in actions:
                        control_multimedia(most_common)
                        engine.say(f"Action: {actions[most_common]}")
                        engine.runAndWait()

                    elif most_common == 'Y':
                        engine.say("Action canceled")
                        engine.runAndWait()
                        perform = False

                elif most_common == 'Y':
                    engine.say("Next letter will be the action")
                    engine.runAndWait()
                    perform = True
            
            # Reset predictions and timer
            predictions = []
            start_time = None

def update():
    process_frame()
    window.after(100, update)

def welcome():
    engine.say("Welcome")
    engine.runAndWait()

window.after(100, welcome)
window.after(100, update)
window.mainloop()