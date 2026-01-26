# hailo_infer.py
import numpy as np
from hailo_platform import VDevice, HailoSchedulingAlgorithm, FormatType, HEF
import time
import cv2 as cv

# Initialize device and model once (outside function)
timeout_ms = 1000
batchsize = 8

params = VDevice.create_params()
params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
vdevice = VDevice(params)

infer_model = vdevice.create_infer_model('yolox_tiny.hef')
infer_model.input().set_format_type(FormatType.FLOAT32)
infer_model.output().set_format_type(FormatType.FLOAT32)

configured_infer_model = infer_model.configure()
bindings_list = []

bindings = configured_infer_model.create_bindings()
# buffers will be set per frame
bindings_list.append(bindings)

def run_inference(frame: np.ndarray):
    #https://github.com/openvinotoolkit/open_model_zoo/blob/55761abd40abce5c4057b0dc47394afed3114d82/data/dataset_classes/coco_80cl.txt#L4
    #The output array person is index 0
    img = cv.imread("pic3.jpeg")  # shape: (H, W, 3), BGR format
    print("Original shape:", img.shape)

    # Convert to float32
    img_float = img.astype(np.float32)
    img_resized = cv.resize(frame, (416, 416))

    # Flatten to 1D for HailoRT
    frame = img_resized.ravel()
    # frame = np.zeros((416, 416, 3), dtype=np.float32).ravel()
    # frame_resized = cv.resize(frame, (832,832))
    bindings.input().set_buffer(frame.astype(np.float32))
    # allocate output buffer based on model output shape
    buffer2 = np.empty(infer_model.output().shape).astype(np.float32)
    bindings.output().set_buffer(buffer2)

    t = time.time()
    configured_infer_model.run(bindings_list, timeout_ms)
    dt = time.time() - t
    print(f"Inference took {dt:.4f} s")
    print(bindings.output().get_buffer())

    return bindings.output().get_buffer()

t = run_inference(0)
for i,ar in enumerate(t):
    if ar.size > 0:
        print(i)