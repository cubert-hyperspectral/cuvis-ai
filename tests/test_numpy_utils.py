import pytest

from cuvis_ai.utils.numpy import get_shape_without_batch
from cuvis_ai.utils.test import get_np_dummy_data


@pytest.mark.parametrize(
    "ignore,expected",
    [
        (None, (3, 2, 1)),
        ([2], (3, 2, -1)),
        ((2), (3, 2, -1)),
        ((2,), (3, 2, -1)),
        ((2, 1), (3, -1, -1)),
    ],
)
def test_get_shape_without_batch(ignore, expected):
    """Ensure shape utilities drop the batch dimension and respect ignore masks."""
    data = get_np_dummy_data((4, 3, 2, 1))
    if ignore is None:
        assert get_shape_without_batch(data) == expected
    else:
        assert get_shape_without_batch(data, ignore=ignore) == expected
