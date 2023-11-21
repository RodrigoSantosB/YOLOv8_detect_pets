import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model_path = '/home/rsb6/Desktop/RoboCin/visao/Tasks/pets_detection_yolov8/runs/detect/train3/weights/best.pt'
model = YOLO(model_path)

# Open the video file
video_path = "/home/rsb6/Videos/pets.mp4"
cap = cv2.VideoCapture(video_path)


# Define the desired width and height
desired_width = 600
desired_height = 800

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        # Resize the frame to the desired width and height
        resized_frame = cv2.resize(annotated_frame, (desired_width, desired_height))

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", resized_frame)
        
        # Introduce a delay of 50 milliseconds 
        cv2.waitKey(50)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()