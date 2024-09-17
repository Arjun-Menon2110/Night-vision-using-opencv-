import cv2

import numpy as np

# Open video capture (0 means the default webcam)
cap = cv2.VideoCapture(0)

while True:
    # Capture each frame from the webcam
    ret, frame = cap.read()
    
    # If frame not captured properly, break the loop
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate the average light intensity (brightness) of the current frame
    light_intensity = np.mean(gray)

    # Apply histogram equalization to enhance contrast
    equalized = cv2.equalizeHist(gray)

    # Add green tint to the frame to simulate night vision
    night_vision_frame = cv2.merge((np.zeros_like(equalized), equalized, np.zeros_like(equalized)))

    # Optionally increase brightness and contrast (alpha: contrast, beta: brightness)
    night_vision_frame = cv2.convertScaleAbs(night_vision_frame, alpha=1.5, beta=30)

    # Add a light intensity meter on the frame
    intensity_text = f"Light Intensity: {int(light_intensity)}"
    cv2.putText(night_vision_frame, intensity_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Optionally change the appearance if the light intensity is too high
    if light_intensity > 150:  # Arbitrary threshold for high brightness
        warning_text = "Too Much Light!"
        cv2.putText(night_vision_frame, warning_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Display the night vision frame with light intensity meter
    cv2.imshow('Night Vision with Light Intensity Meter', night_vision_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

