import os
from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
def compress(sentence):
    output = ""
    flag = False
    for char in sentence:
        if char != " ":
            output += char
            flag = False
        elif not flag:
            output += " "
            flag = True
    return output
#banned character
char_ban = ["`","!","@",".","'",'"',",",")","(","&","^","#","/","|","[","]"]

# Load the YOLO models
vehicle_model = YOLO('yolov8s.pt')  # COCO model for vehicle detection
plate_model = YOLO('best.pt')  # Custom model for license plate detection

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])
# Open the video file
video_path = input()#input video path
cap = cv2.VideoCapture(video_path)
# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
# Store the highest confidence scores for each plate
confidence_map = {}
plate_map = {}
frame_skip = 3  # Process every frame
# Loop through the video frames
frame_count = 0
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # Skip frames to improve performance
        # cv2.imshow("test", frame)
        # Optionally resize the frame to speed up processing (comment out if not needed)
        # frame = cv2.resize(frame, (1000,550))

        # Step 1: Detect and track vehicles using the COCO model
        vehicle_results = vehicle_model.track(frame, persist=True)

        if vehicle_results[0].boxes.id is not None:
            vehicle_boxes = vehicle_results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = vehicle_results[0].boxes.id.cpu().numpy().astype(int)


            # Visualize the vehicle detection results
            annotated_frame = vehicle_results[0].plot()
            # cv2.imshow("nice",annotated_frame)
            # Step 2: For each detected vehicle, run the plate detection model
            for box, track_id in zip(vehicle_boxes, track_ids):
                x1, y1, x2, y2 = box
                vehicle_image = frame[y1:y2, x1:x2]

                plate_results = plate_model(vehicle_image)

                if plate_results[0].boxes.xyxy is not None:
                    plate_boxes = plate_results[0].boxes.xyxy.cpu().numpy().astype(int)

                    # Step 3: Perform OCR on the detected license plates
                    for p_box in plate_boxes:
                        px1, py1, px2, py2 = p_box
                        cv2.rectangle(annotated_frame, (x1 + px1, y1 + py1), (x1 + px2, y1 + py2), (0, 255, 0), 2)
                        plate_image = vehicle_image[int(py1):int(py2), int(px1):int(px2)]

                        # Convert the plate image to grayscale and apply bilateral filter

                        plate_image = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
                        plate_image = cv2.bilateralFilter(plate_image, 10, 20, 20)
                        # cv2.imwrite(f"gambar_plat/{track_id}_grey.jpg", plate_image)
                        # Perform OCR on the plate image

                        plate_text = reader.readtext(plate_image)
                        print(plate_text)

                        if plate_text:
                            # detected plate
                            detected_text = ""
                            for i in range(len(plate_text)-1):
                                detected_text += plate_text[i][-2] + " "
                            detected_text += plate_text[len(plate_text)-1][-2]
                             # mean of confidence by words
                            #post processing string result
                            compressed_text = compress(detected_text)
                            cnt = 0
                            idx = 0
                            for i in compressed_text:
                                if i == " ":
                                    cnt += 1
                                if cnt == 3:
                                    break
                                idx += 1
                            detected_text = compressed_text[:idx]

                            copy_detected = ""
                            for i in detected_text:
                                if i in char_ban:
                                    continue
                                copy_detected += i
                            detected_text = copy_detected
                            confidence_score = 0
                            for res in plate_text:
                                confidence_score += res[-1]
                            confidence_score /= len(plate_text)
                            #count space cause indonesian formats
                            cnt = 0
                            for i in detected_text:
                                if i == " ":
                                    cnt += 1

                            # Update the confidence map if the new confidence score is higher
                            if track_id not in confidence_map or confidence_score > confidence_map[track_id] and cnt == 2:
                                confidence_map[track_id] = confidence_score
                                plate_map[track_id] = (detected_text, (x1 + px1, y1 + py1, x1 + px2, y1 + py2))
                            print(
                                f'Detected plate text: {detected_text} with confidence: {confidence_score}')

                            if confidence_map[track_id] > 0.45:  # Display only if the score is above the threshold
                                # Place the text directly above the plate
                                cv2.putText(annotated_frame, plate_map[track_id][0], (x1 + px1, y1 + py1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                            (0, 255, 0), 2, cv2.LINE_AA)
            # Display the annotated frame
            cv2.imshow("Vehicle and Plate Detection", annotated_frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break
# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
