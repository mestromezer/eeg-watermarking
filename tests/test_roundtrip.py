import numpy as np
import pytest
from conftest import ALGO_CASES, REVERSIBLE, make_channel
from lib.wm.embedder import CantEmbed, InvalidConfig


@pytest.mark.parametrize("cls,kw", ALGO_CASES)
def test_ber_zero(cls, kw, ch_sine, wm):
    """Извлечённый ЦВЗ совпадает с встроенным."""
    emb  = cls(**kw)
    res  = emb.embed(ch_sine, wm)
    extr = emb.extract(make_channel(res.carrier), len(wm))
    assert np.array_equal(extr.extracted, wm)


@pytest.mark.parametrize("cls,kw", ALGO_CASES)
def test_carrier_length_unchanged(cls, kw, ch_sine, wm):
    emb = cls(**kw)
    res = emb.embed(ch_sine, wm)
    assert len(res.carrier) == len(ch_sine.signal)


@pytest.mark.parametrize("cls,kw", ALGO_CASES)
def test_extracted_length(cls, kw, ch_sine, wm):
    emb  = cls(**kw)
    res  = emb.embed(ch_sine, wm)
    extr = emb.extract(make_channel(res.carrier), len(wm))
    assert len(extr.extracted) == len(wm)


@pytest.mark.parametrize("cls,kw", ALGO_CASES)
def test_bps_positive(cls, kw, ch_sine, wm):
    emb = cls(**kw)
    res = emb.embed(ch_sine, wm)
    assert res.bps > 0


@pytest.mark.parametrize("cls,kw", ALGO_CASES)
def test_embed_psnr_positive(cls, kw, ch_sine, wm):
    emb = cls(**kw)
    res = emb.embed(ch_sine, wm)
    assert res.embed_psnr > 0


@pytest.mark.parametrize("cls", REVERSIBLE)
def test_reversible_restore(cls, ch_sine, wm):
    """Обратимые алгоритмы точно восстанавливают оригинал."""
    emb  = cls()
    res  = emb.embed(ch_sine, wm)
    extr = emb.extract(make_channel(res.carrier), len(wm), orig_signal=ch_sine.signal)
    assert np.array_equal(extr.restored, ch_sine.signal)
    assert extr.restore_psnr == float("inf")


@pytest.mark.parametrize("cls,kw", ALGO_CASES)
def test_all_zeros_wm(cls, kw, ch_sine):
    wm_zeros = np.zeros(64, dtype=np.uint8)
    emb  = cls(**kw)
    res  = emb.embed(ch_sine, wm_zeros)
    extr = emb.extract(make_channel(res.carrier), len(wm_zeros))
    assert np.array_equal(extr.extracted, wm_zeros)


@pytest.mark.parametrize("cls,kw", ALGO_CASES)
def test_all_ones_wm(cls, kw, ch_sine):
    wm_ones = np.ones(64, dtype=np.uint8)
    emb  = cls(**kw)
    res  = emb.embed(ch_sine, wm_ones)
    extr = emb.extract(make_channel(res.carrier), len(wm_ones))
    assert np.array_equal(extr.extracted, wm_ones)


def test_cant_embed_too_long(ch_sine):
    """ЦВЗ длиннее контейнера → CantEmbed."""
    from lib.wm.lsb import LSBEmbedder
    emb    = LSBEmbedder(allow_partial=False)
    wm_big = np.ones(len(ch_sine.signal) * 10, dtype=np.uint8)
    with pytest.raises(CantEmbed):
        emb.embed(ch_sine, wm_big)


def test_allow_partial(ch_sine):
    """allow_partial=True не бросает, встроено меньше запрошенного."""
    from lib.wm.lsb import LSBEmbedder
    emb    = LSBEmbedder(allow_partial=True)
    wm_big = np.ones(len(ch_sine.signal) * 10, dtype=np.uint8)
    res    = emb.embed(ch_sine, wm_big)
    assert len(res.watermark) < len(wm_big)


def test_invalid_config():
    from lib.wm.lsb import LSBEmbedder
    from lib.wm.pee import PEEEmbedder
    with pytest.raises(InvalidConfig):
        LSBEmbedder(block_len=0)
    with pytest.raises(InvalidConfig):
        LSBEmbedder(block_len=9)
    with pytest.raises(InvalidConfig):
        PEEEmbedder(block_len=4)


def test_algo_params_keys():
    """algo_params содержит block_len и redundancy для всех алгоритмов."""
    from lib.wm.rcm import RCMEmbedder
    from lib.wm.itb import ITBEmbedder
    from lib.wm.lsb import LSBEmbedder
    from lib.wm.pee import PEEEmbedder
    for emb in [RCMEmbedder(), ITBEmbedder(), LSBEmbedder(), PEEEmbedder()]:
        p = emb.algo_params()
        assert "block_len"  in p
        assert "redundancy" in p
