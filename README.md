# Indonesian License Plate Detection and Recognition

This project focuses on detecting and recognizing Indonesian vehicle license plates from video feeds using advanced computer vision techniques. The system utilizes **YOLO (You Only Look Once)** for object detection, **EasyOCR** for text recognition, and **OpenCV** for image processing and visualization.

## Features

1. **Vehicle Detection**: Detects and tracks vehicles using the YOLOv8 model trained on the COCO dataset.
2. **License Plate Detection**: Identifies license plates on detected vehicles using a custom-trained YOLO model.
3. **OCR for License Plate Recognition**: Uses EasyOCR to extract text from detected license plates.
4. **Post-Processing**: Filters detected text to handle noise, banned characters, and ensures adherence to Indonesian license plate formats.
5. **Confidence-Based Results**: Retains only high-confidence detections to improve accuracy.
6. **Real-Time Performance**: Optimized for processing video streams efficiently by skipping frames.

## Tech Stack

- **YOLOv8**: Used for vehicle and license plate detection.
- **EasyOCR**: For optical character recognition (OCR) of license plate text.
- **OpenCV**: For video frame handling, visualization, and image preprocessing.
- **Python**: Core programming language.

## How It Works

1. **Input**: A video file containing vehicles with visible license plates.
2. **Vehicle Detection**: YOLOv8 detects vehicles in each frame and tracks them across frames.
3. **License Plate Detection**: A custom-trained YOLO model detects license plates on the detected vehicles.
4. **Text Recognition**: EasyOCR extracts text from detected license plates.
5. **Post-Processing**:
   - Compresses and formats the text.
   - Removes unwanted characters.
   - Ensures text adheres to the expected format for Indonesian license plates.
6. **Output**: Annotated video frames with detected license plates and their corresponding text.

## Setup and Installation

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Install Dependencies**
   Install the required Python libraries:
   ```bash
   pip install ultralytics easyocr opencv-python-headless numpy
   ```

3. **Download Models**
   - **Vehicle Detection Model**: Download YOLOv8 pre-trained model (`yolov8s.pt`) from the [Ultralytics YOLO repository](https://github.com/ultralytics/ultralytics).
   - **License Plate Detection Model**: Use your custom YOLO model (`best.pt`).

4. **Run the Script**
   ```bash
   python <script_name>.py
   ```
   Provide the video path when prompted.

## Usage Instructions

- Place your video file in the working directory or specify its path.
- Run the script and view the annotated video frames with detected license plates and recognized text.
- Press `q` to stop the video processing at any time.

## Results and Visualization

- The script outputs annotated video frames with bounding boxes for vehicles and license plates.
- Recognized text is displayed above the license plate on the frame.

## Customization

- **Frame Skipping**: Adjust the `frame_skip` variable to control the frequency of processed frames.
- **Confidence Threshold**: Modify the confidence threshold to filter out low-confidence results.
- **Banned Characters**: Update the `char_ban` list to add/remove restricted characters.

## Limitations

- Performance depends on the quality of the input video.
- Custom YOLO model (`best.pt`) must be trained specifically for plates
- Accuracy may be affected by factors like motion blur, occlusions, or non-standard license plates.

## Acknowledgments

- **Ultralytics**: For the YOLO object detection framework.
- **EasyOCR**: For the OCR functionality.
- **OpenCV**: For video and image processing tools.

## License

This project is open-source and available under the [MIT License](LICENSE).

---

**Author:** Emery Fathan Zwageri

