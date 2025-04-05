import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit # AUTOMATICALLY INITIALISES CUDA DRIVER
import tensorrt as trt
import time
import pyzed.sl as sl

# PATHS AND CONFIG
ENGINE_PATH = "best.engine"
INPUT_HEIGHT = 640
INPUT_WIDTH = 640
CONF_THRESH = 0.5

# TENSOR INFERENCE CLASS
class YOLOv8TRT:
    def __init__(self, engine_path):
        # TensorRT logger monitors warnings at build or during runtime
        self.logger = trt.Logger(trt.Logger.WARNING)
        # Load TensorRT engine from disk
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        # Create an execution context from the engine
        self.context = self.engine.create_execution_context()

        # Get index of input and output bindings
        # 'images' -> GPU input buffer
        # 'output0' -> GPU output buffer
        # 'images' and 'output0' are the default names when exporting onnx from YOLOv8
        self.input_binding_idx = self.engine.get_binding_index('images')
        self.output_binding_idx = self.engine.get_binding_index('output0')

        # Create input/output buffer sizes
        input_shape = self.engine.get_binding_shape(self.input_binding_idx)
        self.input_size = trt.volume(input_shape) * np.dtype(np.float32).itemsize
        output_shape = self.engine.get_binding_shape(self.output_binding_idx)
        self.output_size = trt.volume(output_shape) * np.dtype(np.float32).itemsize

        # Allocate memory on GPU for input and output
        self.device_input = cuda.mem_alloc(self.input_size)
        self.device_output = cuda.mem_alloc(self.output_size)

        # Bindings tell TensorRT what memory to use for execution 
        self.bindings = [int(self.device_input), int(self.device_output)]

        # Pinned host memory for fast GPU->CPU transfer
        self.output_host = cuda.pagelocked_empty(shape=output_shape, dtype=np.float32)

    def inference(self, input_image):
        # Copy input image to GPU
        cuda.memcpy_htod(self.device_input, input_image)
        # Run inference
        self.context.execute_v2(self.bindings)
        # Copy output back from GPU
        cuda.memcpy_dtoh(self.output_host, self.device_output)
        return self.output_host.reshape(1, 84, -1)
    
def preprocess(image):
    # Resize image to standard YOLOv8 size
    resize = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT))
    # Normalise pixel values to [0, 1]
    img = resize.astype(np.float32) / 255.0
    # Convert Numpy array to CHW (Channel, Height, Width)
    img = img.transpose(2, 0, 1)
    # Add batch dimension (1, 3, 640, 640)
    img = np.expand_dims(img, axis=0)
    # Return contiguous memory layout for Tensor
    return np.ascontiguousarray(img)

def draw_detections(img, outputs):
    detections = outputs[0]

    for i in range(detections.shape[-1]):
        conf = detections[4, i]
        if conf < CONF_THRESH:
            continue
    
        x, y, w, h = detections[0:4, i]
        x1 = int((x - w / 2) * img.shape[1])
        y1 = int((y - h / 2) * img.shape[0])
        x2 = int((x + w / 2) * img.shape[1])
        y2 = int((y + h / 2) * img.shape[0])

        class_id = int(detections[5:, i].argmax())
        label = f'ID {class_id} ({conf:.2f})'

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return img

def main():
    # Load model
    model = YOLOv8TRT(ENGINE_PATH)

    # Initialise ZED camera
    zed = sl.Camera()    
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION_HD600
    init_params.camera_fps = 30

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("ZED camera failed to open")
        return
    
    runtime_params = sl.RuntimeParameters()
    image_zed = sl.Mat()

    print("ZED + YOLOv8 Inference Running... press Q to quit")
    while True:
        # Start frame timer
        start_time = time.time()

        # Take frame from the ZED
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # Take from left camera since we need depth
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            frame = image_zed.get_data()

            # Preprocess frame and run inference
            input_tensor = preprocess(frame)
            output = model.inference(input_tensor)

            # Draw detections
            annotated = draw_detections(frame.copy(), output)

            # Display FPS
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(annotated, f"FPS: {fps:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            # Display result
            cv2.imshow("YOLOv8 + ZED X", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    zed.close()
    zed.destroyAllWindows()

if __name__ == '__main__':
    main()