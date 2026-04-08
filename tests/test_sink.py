import csv
import tempfile
from pathlib import Path

import numpy as np
import pytest
from conftest import make_channel
from lib.metrics.sink import CSVMetricSink
from lib.wm.rcm import RCMEmbedder
from lib.wm.lsb import LSBEmbedder


def _run_embed(emb, ch, wm):
    sink_path = Path(tempfile.mktemp(suffix=".csv"))
    emb._metric_sink = CSVMetricSink(sink_path)
    res = emb.embed(ch, wm)
    return sink_path


@pytest.fixture
def ch(ch_sine):
    return ch_sine


@pytest.fixture
def wm():
    return np.ones(32, dtype=np.uint8)


def test_param_block_len_written(ch, wm):
    emb  = RCMEmbedder(metric_sink=None)
    path = _run_embed(emb, ch, wm)
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert rows[0]["param_block_len"] == "1"


def test_mixed_algos_no_lost_columns(ch, wm):
    """RCM + LSB в один файл — оба param-столбца присутствуют в каждой строке."""
    path = Path(tempfile.mktemp(suffix=".csv"))
    sink = CSVMetricSink(path)

    RCMEmbedder(metric_sink=sink).embed(ch, wm)
    LSBEmbedder(metric_sink=sink).embed(ch, wm)

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert len(rows) == 2
    # оба param присутствуют у каждой строки
    for row in rows:
        assert "param_block_len"  in row
        assert "param_redundancy" in row
    # LSB-специфичный столбец есть, у RCM-строки он пустой
    assert "param_lowest_bit" in rows[1]
    assert rows[0].get("param_lowest_bit", "") == ""


def test_rcm_specific_param_written(ch, wm):
    path = Path(tempfile.mktemp(suffix=".csv"))
    sink = CSVMetricSink(path)
    RCMEmbedder(metric_sink=sink).embed(ch, wm)
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert rows[0]["param_rcm_shift"] == "1"
