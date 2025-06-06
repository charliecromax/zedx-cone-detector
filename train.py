from ultralytics import YOLO
from roboflow import Roboflow
import cv2
import torch
import time

def get_dataset():
    # grab dataset from roboflow 
    rf = Roboflow(api_key="6Su2fBIyIOulThNAbAma")
    project = rf.workspace("utsma").project("fsae-cones-dataset")
    version = project.version(2)
    dataset = version.download("yolov8")
                       
def train_data():
    model = YOLO("yolov8n.pt")

    # train dataset taken from roboflow
    model.train(
    data="F:\\utsma_detect\\zedx-cone-detector\\fsae-cones-dataset-2\\data.yaml",
    epochs=20,
    imgsz=640,
    batch=8,
    device="cuda" if torch.cuda.is_available() else "cpu",
    max_det=100,
    conf=0.25,
    save=True,
    verbose=True
    )

def convert_onnx():
    model = YOLO("F:\\utsma_detect\\zedx-cone-detector\\runs\detect\\train\weights\\best.pt")
    # convert to onnx format first
    model.export(format="onnx", opset=12)
   
    # run script on main pc --> after installing tensorrt
    # trtexec --onnx=best.onnx --saveEngine=best.engine --fp16

def video_test_normal():
    # use the pytorch file with the best training results
    model = YOLO("runs/detect/train/weights/best.pt")

    # get video path
    video_path = "comp_sim.mp4"
    cap = cv2.VideoCapture(video_path)

    # Check if video is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    # Get original video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define new resolution -> only use if video file resolution is very small
    scale_factor = 4  
    original_width = int(original_width * scale_factor)
    original_height = int(original_height * scale_factor)

    # Define the video writer
    output_path = "output_video.mov"
    fourcc = cv2.VideoWriter_fourcc(*'mpv4')  # Codec for MP4 format
    out = cv2.VideoWriter(output_path, fourcc, fps, (original_width, original_height))

    # Process video frame-by-frame
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or read error.")
            break

        # Perform YOLOv8 object detection
        results = model(frame)

        # plot the bounding boxes for each frame
        for result in results:
            frame = result.plot()

        # Resize frame to new resolution -> only if it has been change, otherwise size will just stay the same
        frame_resized = cv2.resize(frame, (original_width, original_height), interpolation=cv2.INTER_LINEAR)

        out.write(frame_resized)

        cv2.imshow("YOLOv8 Detection", frame_resized)

        # q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    # display final frame count after video terminated
    print(f"Processed {frame_count} frames successfully.")

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # confirm final video resolution size as well as the file name of the output video
    print(f"Output video saved as {output_path}, Resolution: {original_width}x{original_height}")

def video_test_sped_up():
    model = YOLO("runs/detect/train/weights/best.pt")

    video_path = "track_walk.mov"  
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    # Get original video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Original Video: {frame_width}x{frame_height}, FPS: {fps}")

    # Speed factor (higher = faster video)
    speed_factor = 3

    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Best for MOV output
    output_path = "output_video.mov"  # Save as MOV
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    saved_frame_count = 0
    start = time.perf_counter()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or read error.")
            break

        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.resize(frame, (640, 640))

        # Skip frames to speed up
        if frame_count % speed_factor == 0:  # Keep only every Nth frame
            results = model(frame, imgsz=640)  # Perform YOLO detection
            frame = results[0].plot()

            out.write(frame)  # Write processed frame
            saved_frame_count += 1

            cv2.imshow("YOLOv8 Detection", frame)  # Show output (optional)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1

    end = time.perf_counter()
    elapsed = end - start
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Processed {frame_count} frames, Saved {saved_frame_count} frames (Speed factor: {speed_factor})")
    print(f"Output video saved as {output_path}")
    print(f"Average runtime per frame: {(elapsed * (10**3))/frame_count:.1f} ms")


if __name__ == "__main__":
    # get_dataset()
    train_data()
    # convert_onnx()
    # video_test_sped_up()