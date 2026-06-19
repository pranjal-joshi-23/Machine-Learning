import cv2
import mediapipe as mp
import time
import math
import socket

cap = cv2.VideoCapture(1)

cap.set(3, 1280)
cap.set(4, 720)

camera_horizontal_degrees = 9
camera_vertical_degrees = 3.4

degree_per_horizontal_pixel = camera_horizontal_degrees/1280
degree_per_vertical_pixel = camera_vertical_degrees/720

servo_horizontal = 90
servo_vertical = 90

servo_horizontal_to_move = 0
servo_vertical_to_move = 0

pTime = 0

mpFaceDetection = mp.solutions.face_detection

faceDetection = mpFaceDetection.FaceDetection(0.75)

# Communication part

host = '0.0.0.0'
port = 65433

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host, port))
server_socket.listen(1)

print(f"Server listening on {host}:{port}")

conn, addr = server_socket.accept()
print(f"Connected by {addr}")

while True:
    _, frame = cap.read()
    if not _:
        break

    # frame = cv2.flip(frame, 1)

    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = faceDetection.process(frameRGB)

    # Circle at center of frame
    cv2.circle(frame, (frame.shape[1]//2, frame.shape[0]//2), 4, (0, 255, 0), 2)

    if results.detections:
        for id, detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = frame.shape
            bboxX = int(bboxC.xmin * iw)
            bboxY = int(bboxC.ymin * ih)
            bboxW = int(bboxC.width * iw)
            bboxH = int(bboxC.height * ih)
            cv2.rectangle(frame, (bboxX, bboxY), (bboxX+bboxW, bboxY+bboxH), (255, 0, 255), 2)

            # Circle at center of face
            cv2.circle(frame, ((bboxX+(bboxW//2)), (bboxY+(bboxH//2))), 4, (0, 255, 0), 2)

            # Line joining both of those circles
            cv2.line(frame, (frame.shape[1]//2, frame.shape[0]//2), (bboxX+(bboxW//2), (bboxY+(bboxH)//2)), (155,155,155), 2)

            # Distance between those 2 points
            dis = math.sqrt((bboxX+(bboxW//2)-frame.shape[1]//2)**2 + ((bboxY+(bboxH)//2)-frame.shape[0]//2)**2)

            # Horizontal Dis
            horizontal_dis = (bboxX+(bboxW//2)) - frame.shape[1]//2
            cv2.line(frame, (frame.shape[1]//2, frame.shape[0]//2), (bboxX+(bboxW//2), frame.shape[0]//2), (155,155,155),2)
            servo_horizontal_to_move = abs(horizontal_dis) * degree_per_horizontal_pixel
            
            if (horizontal_dis > 0):
                if ((servo_horizontal + servo_horizontal_to_move) <= 180):
                    servo_horizontal += servo_horizontal_to_move
                    message = f"{int(servo_horizontal)}"
                    conn.sendall(message.encode())
                    time.sleep(.2)
                else:
                    print("Out of reach")
            elif (horizontal_dis < 0):
                if ((servo_horizontal - servo_horizontal_to_move) >= 0):
                    servo_horizontal -= servo_horizontal_to_move
                    message = f"{int(servo_horizontal)}"
                    conn.sendall(message.encode())
                    time.sleep(.2)
                else:
                    print("Out of reach")

            # Vertical Dis
            vertical_dis = (bboxY+(bboxH//2)) - frame.shape[0]//2
            cv2.line(frame, (bboxX+(bboxW//2), (bboxY+(bboxH)//2)), (bboxX+(bboxW//2), frame.shape[0]//2), (155, 155, 155), 2)
            servo_vertical_to_move = abs(vertical_dis) * degree_per_vertical_pixel
            
            if (vertical_dis > 0):
                if ((servo_vertical + servo_vertical_to_move) <= 180):
                    servo_vertical += servo_vertical_to_move
                    message = f"{int(servo_vertical)}"
                    conn.sendall(message.encode())
                    time.sleep(.2)
                else:
                    print("Out of reach")
            elif (vertical_dis < 0):
                if ((servo_vertical - servo_vertical_to_move) >= 0):
                    servo_vertical -= servo_vertical_to_move
                    message = f"{int(servo_vertical)}"
                    conn.sendall(message.encode())
                    time.sleep(.2)
                else:
                    print("Out of reach")

            cv2.putText(frame, f"{int(detection.score[0] * 100)}%", (bboxX, bboxY - 20), cv2.FONT_HERSHEY_COMPLEX, 1,(255, 0, 255), 2)

    # FPS Meter
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 45), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 204, 51), 2)

    cv2.imshow("Face Detection", frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()