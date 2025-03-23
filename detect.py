from trt_detector import TRTDetector
import pyzed.sl as sl
import cv2

def zedDetector():
    # initialise Zed 
    zed = sl.Camera()
    init_params = sl.InitParameters(camera_resolution=sl.RESOLUTION.HD720, camera_fps=30)
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("ZED failed to open")
        exit()

    image_zed = sl.Mat()

    # pass in the Tensor engine that you converted from pt --> onnx --> engine
    detector = TRTDetector("best.engine")

    # run detection loop 
    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # grab image from left zed camera 
            # - left typically used for obj detection, tracking
            # - right is mainly useful for custom stereo processing (not necessary in our case)
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            # the raw image from the zed camera is format as such: (height, width, 4)
            # the 4 represents BGRA, YOLO expects BGR so we need to cut off A - alpha
            frame = image_zed.get_data()[:, :, :3]

            # call detector func from trt_detector --> store inside of boxes var
            boxes = detector.detect(frame) 

            for (x1, y1, x2, y2, conf, cls) in boxes:
                label = f"Class {cls} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("ZED + YOLOv8 (TensorRT)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    zed.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    zedDetector()