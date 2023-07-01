from flask import Flask, render_template, Response
import requests
import cv2       
import socket   
import json
import pickle   
import numpy as np   
import base64
import time  
import struct
latest_frame = None
flag = True 
conn=False
# Connection to the server  ....    
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_ip = '127.0.0.1'  # IP address of the server
port = 9999  # Use the same port number as the server
socket_address = (server_ip, port)    
 

app = Flask(__name__)
@app.route('/')
def index():   
     
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Define a generator function to process frames and yield them
    def generate_frames():
        global latest_frame,flag,conn
        # Client socket setup
       
        if  conn==False:
            client_socket.connect(socket_address)
            print('Connected to server')
            conn=True
         
        # Try different camera indices
        for camera_index in range(4):
            video = cv2.VideoCapture(camera_index)
            if video.isOpened():
                break
        if not video.isOpened():
            print("Failed to open the camera.")
            client_socket.close()
            exit(1)

        while True:
            # Read video frames
            _, frame = video.read()

            # Serialize frame
            data = pickle.dumps((frame, True))
            message_size = struct.pack("Q", len(data))
            print(len(message_size))
            # Send frame to the server
            client_socket.sendall(message_size)
            client_socket.sendall(data)

            # Receive processed frame from the server
            processed_data = b""
            while len(processed_data) < 8:  # Adjust buffer size to 8 bytes
                packet = client_socket.recv(8 - len(processed_data))  # Adjust buffer size to 8 bytes
                if not packet:
                    break
                processed_data += packet

            msg_size = struct.unpack("Q", processed_data)[0]
            received_data = b""
            while len(received_data) < msg_size:
                packet = client_socket.recv(4 * 1024)  # Adjust buffer size as needed
                if not packet:
                    break
                received_data += packet

            processed_frame = pickle.loads(received_data)
            
            
            latest_frame = processed_frame
            # Encode the processed frame as JPEG
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()

            # Yield the frame as response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_image', methods=['POST'])
def capture_image():
    global latest_frame
    if latest_frame is None:
        return 'Error: No frame available'

    # create a filename for the captured image using the current timestamp
    timestamp = int(time.time())
    filename =  "static/images/faces/real.jpg"

    # save the latest frame to a JPG file
    cv2.imwrite(filename, latest_frame)
    print("saved")
    # return a JSON response indicating success
    return render_template("live_detection.html",msg="image captured successfully !") 





@app.route("/live_detection")
def live_dete():
    return render_template("live_detection.html")




@app.route('/video')
def video():
    # Define a generator function to process frames and yield them
    def generate_frame():
        global flag
        # Try different camera indices
        for camera_index in range(4):
            video = cv2.VideoCapture(camera_index)
            if video.isOpened():
                break

        if not video.isOpened():
            print("Failed to open the camera.")
            client_socket.close()
            exit(1)

        while True:
    # Read video frames
            _, frame = video.read()

            # Serialize frame
            data = pickle.dumps((frame,False))
            message_size = struct.pack("Q", len(data))
            print(len(message_size))
            # Send frame to the server
            client_socket.sendall(message_size + data)
            

            # Receive processed frame from the server
            processed_data = b""
            while len(processed_data) < 8:  # Adjust buffer size to 8 bytes
                packet = client_socket.recv(8 - len(processed_data))  # Adjust buffer size to 8 bytes
                if not packet:
                    break
                processed_data += packet

            msg_size = struct.unpack("Q", processed_data)[0]
            received_data = b""
            while len(received_data) < msg_size:
                packet = client_socket.recv(4 * 1024)  # Adjust buffer size as needed
                if not packet:
                    break    
                received_data += packet
            
            processed_frame = pickle.loads(received_data)
             
            latest_frame = processed_frame
            # Encode the processed frame as JPEG
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()

            # Yield the frame as response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    return Response(generate_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route("/verify")
def verify():
    url = "http://localhost:9999/verifyFace"
    with open(r"static/images/faces/real.jpg", "rb") as f:
        encoded_string = base64.b64encode(f.read())
        encoded_string = encoded_string.decode('utf-8')
    response = requests.post(url, data=json.dumps({"a":'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', "b": encoded_string}))
    if response.status_code == 200:
        response_data = json.loads(response.text)
        result = response_data['Result']   
        per = response_data['percentage']
        print(f"The result is {result}")  
        print(f"The percentage is {per}")    
    return render_template("match.html",result=result,per=per)
if __name__ == '__main__':
    app.run(debug=True)
