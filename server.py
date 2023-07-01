import cv2
import socket
import pickle
import struct
import numpy as np
import dlib
import random

predictor_path = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)
ques = ["Turn your face left", "Turn your face right"]
detector = dlib.get_frontal_face_detector()

# Server socket setup
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_ip = ''  # Replace with the server's IP address (leave empty to bind to all available interfaces)
port = 9999  # Choose a suitable port number
socket_address = (server_ip, port)

# Bind and listen for connections
server_socket.bind(socket_address)
server_socket.listen(5)
print("Server Listening....")
# Accept a client connection
client_socket, addr = server_socket.accept()
print('Connected with:', addr)

# Logic variables
#question = random.choice(ques)
left_turn_count = 0
same_question_left_frames = 0
right_turn_count=0
same_question_right_frames = 0
totol_try=5
tried=0

while True:
    # Receive and process video frames from the client
    data = b""
    payload_size = struct.calcsize("Q")
    while len(data) < payload_size:
        packet = client_socket.recv(4 * 1024)  # Adjust buffer size as needed
        if not packet:
            print("packet not found")
            break
        data += packet
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    print(f"packed msg size: {len(packed_msg_size)}")

    if len(packed_msg_size) == 0:
        packed_msg_size = struct.pack("Q", 8)

    msg_size = struct.unpack("Q", packed_msg_size)[0]

    while len(data) < msg_size:
        data += client_socket.recv(4 * 1024)  # Adjust buffer size as needed
    frame_data = data[:msg_size]
    data = data[msg_size:]

    frame, flag = pickle.loads(frame_data)
    print(f"Received flag {flag}")
    frame1=frame.copy()
    frame2=frame.copy()
    frame3=frame.copy()
    # Process the frame based on the flag
    if flag:
        mask = np.zeros_like(frame)
        cv2.ellipse(mask, (320, 250), (170, 220), 0, 0, 360, (255, 255, 255), -1)
        processed_frame = cv2.bitwise_and(frame,mask)
        processed_frame[np.where((mask == [0, 0, 0]).all(axis=2))] = [255, 255, 255]
        # Send processed frame back to the client
        processed_frame_data = pickle.dumps(processed_frame)
        client_socket.sendall(struct.pack("Q", len(processed_frame_data)) + processed_frame_data)
    else:
         
        cv2.putText(frame1, f"{ques[0]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        frame_height, frame_width, _ = frame1.shape
        processed_frame = cv2.rectangle(frame1, (0, 0), (frame_width, frame_height), (0, 0, 255), 2)
        
        # Detect face and check if it is turned left
        gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        for face in faces:
            landmarks = predictor(gray, face)
            left_eye_x = landmarks.part(36).x
            right_eye_x = landmarks.part(45).x
            if left_eye_x > 320 and left_turn_count<=20:
                left_turn_count += 1
                cv2.putText(frame1, f"{ques[0]} : okk", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0, 255), 2)
                frame_height, frame_width, _ = frame1.shape
                processed_frame = cv2.rectangle(frame1, (0, 0), (frame_width, frame_height), (0, 0, 255), 2)

        same_question_left_frames += 1
    
        # If 30 consecutive frames with left turn detected or 50 frames reached, break the loop
        if left_turn_count >= 20:
            cv2.putText(frame2, ques[1], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            frame_height, frame_width, _ = frame2.shape
            processed_frame = cv2.rectangle(frame2, (0, 0), (frame_width, frame_height), (0, 0, 255), 2)
            if right_eye_x<=310 and right_turn_count<=20:
                right_turn_count+=1
                cv2.putText(frame2,f"{ques[1]} : okk",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                frame_height, frame_width, _ = frame2.shape
                processed_frame = cv2.rectangle(frame2, (0, 0), (frame_width, frame_height), (0, 0, 255), 2)
        same_question_right_frames += 1
        if right_turn_count>20:
            cv2.putText(frame3,"Liveliness Successfull",(10,60),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2) 
            frame_height, frame_width, _ = frame3.shape
            processed_frame = cv2.rectangle(frame3, (0, 0), (frame_width, frame_height), (0, 0, 255), 2) 
            processed_frame_data = pickle.dumps(processed_frame)
            client_socket.sendall(struct.pack("Q", len(processed_frame_data)) + processed_frame_data)
            break
        if same_question_left_frames>=500:
            tried+=1
            cv2.putText(frame3, "Liveliness Failed !", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            frame_height, frame_width, _ = frame3.shape
            processed_frame = cv2.rectangle(frame3, (0, 0), (frame_width, frame_height), (0, 0, 255), 2) 
            processed_frame_data = pickle.dumps(processed_frame)
            client_socket.sendall(struct.pack("Q", len(processed_frame_data)) + processed_frame_data)
            break
           
             
        if same_question_right_frames>=500:
            tried+=1
            cv2.putText(frame3, "Liveliness Failed !", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            frame_height, frame_width, _ = frame3.shape
            processed_frame = cv2.rectangle(frame3, (0, 0), (frame_width, frame_height), (0, 0, 255), 2) 
            processed_frame_data = pickle.dumps(processed_frame)
            client_socket.sendall(struct.pack("Q", len(processed_frame_data)) + processed_frame_data)
            break

        if totol_try==tried:
            cv2.putText(frame3, "Fake Face detected !", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            frame_height, frame_width, _ = frame3.shape
            processed_frame = cv2.rectangle(frame3, (0, 0), (frame_width, frame_height), (0, 0, 255), 2) 
            break
        # Send processed frame back to the client
        processed_frame_data = pickle.dumps(processed_frame)
        client_socket.sendall(struct.pack("Q", len(processed_frame_data)) + processed_frame_data)
