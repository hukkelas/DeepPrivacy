
import cv2
import time
import numpy as np
import torch
from deep_privacy import cli
from deep_privacy.visualization import utils as vis_utils
from deep_privacy.utils import BufferlessVideoCapture
from deep_privacy.build import build_anonymizer
import os
# Configs
torch.backends.cudnn.benchmark = False
parser = cli.get_parser()
parser.add_argument("--debug", default=False, action="store_true")
parser.add_argument("-f", "--file", default=None)
args = parser.parse_args()
anonymizer, cfg = build_anonymizer(
    args.model, opts=args.opts, config_path=args.config_path,
    return_cfg=True)
if args.debug:
    anonymizer.save_debug = True

width = 1280
height = 720

if args.file is not None:
    assert os.path.isfile(args.file)
    cap = cv2.VideoCapture(args.file)
else:
    cap = BufferlessVideoCapture(0)
frames = 0
WARMUP = True
t = time.time()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, (width, height))
    frame = frame[:, :, ::-1]
    frame = anonymizer.detect_and_anonymize_images([frame])[0]
    frame = frame[:, :, ::-1]

    # Display the resulting frame
    if WARMUP and frames > 30:
        WARMUP = False
        t = time.time()
        frames = 0

    frames += 1
    delta = time.time() - t
    fps = "?"
    if delta > 1e-6:
        fps = frames / delta
    print(f"FPS: {delta:.3f}", end="\r")
    if args.debug:
        debug_im = cv2.imread(".debug/inference/im0_face0.png")
        debug_im = vis_utils.pad_im_as(debug_im, frame)
        frame = np.concatenate((frame, debug_im))
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
