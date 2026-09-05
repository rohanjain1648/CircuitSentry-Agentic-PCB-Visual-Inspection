import numpy as np
from circuitsentry.capture_client.camera import capture_frame, crop_roi


class FakeVideoCapture:
    def __init__(self, frame):
        self._frame = frame

    def read(self):
        return True, self._frame


def test_capture_frame_returns_the_frame():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    cap = FakeVideoCapture(frame)
    result = capture_frame(cap)
    assert result.shape == (100, 100, 3)


def test_crop_roi_upscales_region():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[50:100, 50:100] = 255
    cropped = crop_roi(frame, bbox=(50, 50, 50, 50), zoom=2.0)
    assert cropped.shape[0] == 100  # 50 * zoom
    assert cropped.shape[1] == 100
