import torch

import hypothesis
from hypothesis import given, strategies as st

from decofre import utils


@hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.too_slow])
@given(st.data())
def test_conf(data):
    n = data.draw(st.integers(min_value=1, max_value=1000), label="Number of classes")
    k = data.draw(st.integers(min_value=1, max_value=1000), label="Number of examples")
    predicted_draw = data.draw(
        st.lists(st.integers(min_value=0, max_value=n - 1), min_size=k, max_size=k),
        label="System output"
    )
    target_draw = data.draw(
        st.lists(st.integers(min_value=0, max_value=n - 1), min_size=k, max_size=k),
        label="Reference output"
    )
    predicted = torch.tensor(predicted_draw)
    target = torch.tensor(target_draw)
    test_target = torch.zeros((n, n), dtype=torch.int64)
    for r, k in zip(predicted, target):
        test_target[r, k] += 1
    out = utils.confusion(predicted, target, n)
    assert out.equal(test_target)
