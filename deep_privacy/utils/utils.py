import pathlib
import numpy as np
import cv2
try:
    from apex.amp._amp_state import _amp_state
except ImportError:
    pass


def amp_state_has_overflow():
    for loss_scaler in _amp_state.loss_scalers:
        if loss_scaler._has_overflow:
            return True
    return False


def read_im(impath: pathlib.Path, imsize: int = None):
    assert impath.is_file(),\
        f"Image path is not file: {impath}"
    im = cv2.imread(str(impath))[:, :, ::-1]
    if imsize is not None:
        im = cv2.resize(im, (imsize, imsize))
    if im.dtype == np.uint8:
        im = im.astype(np.float32) / 255
    assert im.max() <= 1 and im.min() >= 0
    return im
