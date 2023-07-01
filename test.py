import cv2
import dlib

# Load the face detector and shape predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor_path = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)

# Open the video capture
video_capture = cv2.VideoCapture(0)  # Replace with the appropriate video source index if not using the default camera

while True:
    # Read a frame from the video capture
    ret, frame = video_capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    # Iterate over the detected faces
    for face in faces:
        # Detect landmarks for the face
        landmarks = predictor(gray, face)

        # Get the coordinates of the left eye
        left_eye_x = landmarks.part(36).x
        left_eye_y = landmarks.part(36).y

        # Get the coordinates of the right eye
        right_eye_x = landmarks.part(45).x
        right_eye_y = landmarks.part(45).y

        # Display the frame with text showing the coordinates
        cv2.putText(frame, "Left Eye: ({}, {})".format(left_eye_x, left_eye_y), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Right Eye: ({}, {})".format(right_eye_x, right_eye_y), (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Video', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
video_capture.release()
cv2.destroyAllWindows()



