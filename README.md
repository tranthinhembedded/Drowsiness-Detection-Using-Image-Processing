<div style="font-family: Montserrat, sans-serif; text-align: justify;">

# REAL-TIME MONITORING OF DRIVER’S DROWSINESS USING EYE AND MOUTH FEATURES
Abstract. Detecting drowsiness in drivers, especially those involved in long-haul journeys at night, is crucial for ensuring safety. In this paper, the research team proposes a drowsiness detection system using a combination of hardware and image processing algorithms such as face detection, eyes aspects ratio, and enhancing low light images. Detecting driver drowsiness quickly and accurately, the system triggers an alarm immediately upon detecting signs of drowsiness. The methodology integrates face shape detection and analysis to determine drowsiness indicators such as eye socket ratio and distance between eyelids. The system operates effectively under different lighting conditions. This study aims to contribute to improving driver safety on the road by detecting drowsiness, thereby advancing traffic safety technology.

- Keywords. Drowsiness detection, Image Processing, Facial Landmarks, OpenCV, Dlib, Eyes aspect ratio, Technology integration.
  
## Usage:
1. **Step 1.** Install the required libraries
   ```bash
   pip install -r requirements.txt
2. **Step 2.** Run the main program file
   ```bash
   python Drowsiness_Detection_Image_Process.py
3. **Key file**
  - **Drowsiness_Detection_Image_Process.py:** Main program file.
  -  **haarcascade_frontalface_default.xml**: Model for face detection.
   - **shape_predictor_68_face_landmarks.dat:** Model for detecting 68 facial landmarks.
   - **music.wav:** Audio file used for the alert sound.

## Hardware
- Jetson Nano 
- Camera

## Notes
- Ensure your camera is functional before running the program.
- Detection accuracy might be affected by poor lighting conditions or improper camera angles.
</div>


