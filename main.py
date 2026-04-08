import logging
from pathlib import Path
from lib.logging_setup import setup_logging
from lib.records import load_record
from lib.wm import RCMEmbedder, ITBEmbedder
from lib.metrics import CSVMetricSink
from lib.utils import WatermarkGenerator

setup_logging(logging.INFO)

DATA_DIR = Path("./data")
OUT_DIR  = Path("./out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

sink = CSVMetricSink(OUT_DIR / "metrics.csv")
wm   = WatermarkGenerator(seed="key").bits(50000)

ALGOS = [
    RCMEmbedder(block_len=1, log_level=logging.INFO, metric_sink=sink, allow_partial=True),
    RCMEmbedder(block_len=4, log_level=logging.INFO, metric_sink=sink, allow_partial=True),
    ITBEmbedder(log_level=logging.INFO, metric_sink=sink, allow_partial=True),
]

CHANNELS = [22,24,30,32,34,36,38,41,9,11,13,42,47,49,51,53,55,61,62,63]

if __name__=='__main__':
    for algo in ALGOS:
        for edf_path in sorted(DATA_DIR.glob("*.edf")):
            rec = load_record(edf_path)

            for idx in CHANNELS:
                cv     = rec.channel_view(idx)
                result = algo.embed(cv, wm, source=rec)
                # rec.update_channel(idx, result.carrier)

            out_name = f"{edf_path.stem}_{algo.codename}_bl{algo.algo_params().get('block_len', 1)}{edf_path.suffix}"