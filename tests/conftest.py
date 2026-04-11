import numpy as np
import pytest
from lib.records.base import ChannelView
from lib.wm.rcm import RCMEmbedder
from lib.wm.itb import ITBEmbedder
from lib.wm.lsb import LSBEmbedder
from lib.wm.pee import PEEEmbedder
from lib.wm.hs import HSEmbedder


N       = 4000
DIG_MIN = -32768
DIG_MAX =  32767
RNG     = np.random.default_rng(42)


def make_channel(signal: np.ndarray) -> ChannelView:
    return ChannelView(
        index=0, label="test",
        signal=signal,
        dig_min=DIG_MIN, dig_max=DIG_MAX,
        sample_freq=256.0,
    )


@pytest.fixture
def ch_sine():
    t = np.linspace(0, 2 * np.pi, N)
    s = (np.sin(t) * 100).astype(np.int16)
    return make_channel(s)


@pytest.fixture
def ch_rand():
    s = RNG.integers(-1000, 1000, size=N, dtype=np.int16)
    return make_channel(s)


@pytest.fixture
def wm():
    return RNG.integers(0, 2, size=64, dtype=np.uint8)


# ── алгоритмы и их конфиги для round-trip параметризации ──────────────────

ALGO_CASES = [
    pytest.param(RCMEmbedder,  {},                          id="rcm-default"),
    pytest.param(RCMEmbedder,  {"block_len": 4},            id="rcm-block4"),
    pytest.param(RCMEmbedder,  {"shuffle": True, "key": "k"}, id="rcm-shuffle"),
    pytest.param(ITBEmbedder,  {},                          id="itb-default"),
    pytest.param(ITBEmbedder,  {"redundancy": 3},           id="itb-redundancy3"),
    pytest.param(LSBEmbedder,  {},                          id="lsb-default"),
    pytest.param(LSBEmbedder,  {"block_len": 4},            id="lsb-block4"),
    pytest.param(LSBEmbedder,  {"shuffle": True, "key": "k"}, id="lsb-shuffle"),
    pytest.param(PEEEmbedder,  {},                          id="pee-default"),
    pytest.param(PEEEmbedder,  {"shuffle": True, "key": "k"}, id="pee-shuffle"),
    pytest.param(HSEmbedder,   {},                          id="hs-default"),
    pytest.param(HSEmbedder,   {"shuffle": True, "key": "k"}, id="hs-shuffle"),
]

REVERSIBLE = [RCMEmbedder, ITBEmbedder, PEEEmbedder, HSEmbedder]
