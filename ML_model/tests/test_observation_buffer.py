"""ObservationBuffer: the preallocated ring gives exactly the old deque + np.stack windows."""
import collections

import numpy as np
import pytest

from exo.deploy.runtime import ObservationBuffer


class _ReferenceBuffer:
    """The previous implementation, kept here as the specification."""

    def __init__(self, window_length, num_channels):
        self.window_length = window_length
        self._buf = collections.deque(maxlen=window_length)

    def push(self, frame):
        self._buf.append(frame.astype(np.float32))

    @property
    def ready(self):
        return len(self._buf) == self.window_length

    def window(self):
        return np.stack(self._buf, axis=1)

    def reset(self):
        self._buf.clear()


@pytest.mark.parametrize("window_length", [1, 7, 300])
def test_matches_reference_through_wraps_and_resets(window_length):
    n_c = 14
    rng = np.random.default_rng(window_length)
    new, ref = ObservationBuffer(window_length, n_c), _ReferenceBuffer(window_length, n_c)
    for k in range(window_length * 4 + 3):
        if k == window_length * 2 + 1:                    # mid-run reset
            new.reset()
            ref.reset()
        frame = rng.normal(size=n_c) * 10.0 ** rng.integers(-6, 6)   # float64 in, like astype
        new.push(frame)
        ref.push(frame)
        assert new.ready == ref.ready
        w, w_ref = new.window(), ref.window()
        assert w.dtype == w_ref.dtype == np.float32
        assert w.shape == w_ref.shape and w.flags["C_CONTIGUOUS"]
        np.testing.assert_array_equal(w, w_ref)


def test_window_is_a_copy():
    buf = ObservationBuffer(3, 2)
    for v in (1.0, 2.0, 3.0):
        buf.push(np.full(2, v))
    w = buf.window()
    buf.push(np.full(2, 4.0))
    np.testing.assert_array_equal(w, [[1.0, 2.0, 3.0]] * 2)   # unchanged by the push


def test_rejects_wrong_channel_count_and_empty_window():
    buf = ObservationBuffer(3, 2)
    with pytest.raises(ValueError):
        buf.push(np.zeros(3))
    with pytest.raises(ValueError):
        buf.window()
