from lib.records import load_record
from lib.wm import RCMEmbedder
from lib.metrics import CSVMetricSink
from lib.utils import WatermarkGenerator
import logging

rec  = load_record("./data/S001R01.edf")
wm   = WatermarkGenerator(seed="key").bits(50000)
algo = RCMEmbedder(block_len=4, log_level=logging.DEBUG, metric_sink=CSVMetricSink("out/metrics.csv"))

for idx in [0, 1, 5]:
    cv     = rec.channel_view(idx)
    result = algo.embed(cv, wm, filename=rec.path.name)
    rec.update_channel(idx, result.carrier)

rec.save("S001R01_wm.edf")