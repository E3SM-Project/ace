import pytest

from fme.core.distributed import Distributed


@pytest.fixture(autouse=True)
def reset_distributed_singleton():
    """Reset the Distributed singleton after each test to prevent state leakage."""
    yield
    Distributed.reset()
