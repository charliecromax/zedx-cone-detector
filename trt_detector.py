import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class TRTDetector:
    def __init__(self, engine_path, input_shape=(1, 3, 640, 640), conf_thresh=0.7):
        # path to serialised TensorRT engine
        self.engine_path = engine_path
        # input shape expected (batch, channels, height, width)
        self.input_shape = input_shape
        # minimum confidence threshold for keeping detections
        self.conf_thresh = conf_thresh
        # logger for debug output
        self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        # all class labels, model was trained on
        self.class_names = ["blue_cone", "large_orange_cone", "orange_cone", "unknown_cone", "yellow_cone"]
        self.allowed_class_ids = [0, 4] # only looking for blue and yellow

        # load engine and prepare execution context and buffers
        self.engine = self._load_engine()
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()

    def _load_engine(self):
        # deserialise the TensorRT engine from file
        with open(self.engine_path, 'rb') as f, trt.Runtime(self.TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _allocate_buffers(self):
        # allocate host and device memory buffers for inputs and outputs
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype) # Pinned mem
            device_mem = cuda.mem_alloc(host_mem.nbytes) # GPU mem
            bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
        
        return inputs, outputs, bindings, stream

    def _preprocess(self, frame):
        # resize and format frame to match yolov8 model input
        img = cv2.resize(frame, (self.input_shape[2], self.input_shape[3])) #640x640
        img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255 # convert from BGR to RGB
        img = np.expand_dims(img, axis=0) # add batch dimension
        return img

    def _postprocess(self, detections, frame):
        # convert model outputs to bbox coords filtering by confidence
        boxes = []
        for det in detections:
            if det[4] < self.conf_thresh:
                continue
            x, y, w, h = det[:4] # YOLO format: x, y, width, height
            class_scores = det[5:]
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]
            if confidence > self.conf_thresh and class_id in self.allowed_class_ids:
                # convert to (x1, y1, x2, y2)
                x1 = int((x - w / 2) * frame.shape[1])
                y1 = int((y - h / 2) * frame.shape[0])
                x2 = int((x + w / 2) * frame.shape[1])
                y2 = int((y + h / 2) * frame.shape[0])
                label = self.class_names[class_id] if class_id < len(self.class_names) else str(class_id)
                boxes.append((x1, y1, x2, y2, confidence, label))
        return boxes

    def detect(self, frame):
        # run full detection sequence: preprocess, inference, postprocess
        img = self._preprocess(frame)
        np.copyto(self.inputs[0]['host'], img.ravel())

        # copy input to device
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        # run inference on input
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        # copy output back to host
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()

        detections = self.outputs[0]['host'].reshape(-1, 85) # YOLOv8 format: (num_detections, 85)
        return self._postprocess(detections, frame)

