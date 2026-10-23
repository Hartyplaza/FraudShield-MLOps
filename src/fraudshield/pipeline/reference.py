"""
Stage 8: drift reference. Scores an unseen stream (the period right after the
training data) exactly as the API does, and saves how every feature and the
score were spread. Live traffic is compared with this profile.

The profile holds bin edges and shares only, no transactions.

    uv run python -m fraudshield.pipeline.reference
"""

import logging

from fraudshield.monitoring.drift import build_reference, save_reference
from fraudshield.monitoring.stream import score_stream
from fraudshield.pipeline.config import ensure_parent, stage_args

logger = logging.getLogger(__name__)


def reference_stream_cfg(cfg: dict) -> dict:
    return {**cfg["generator"], **cfg["monitoring"]["reference"]}


def main() -> None:
    cfg = stage_args("Build the drift reference profile from an unseen stream.")
    paths = cfg["paths"]

    stream = score_stream(paths["model_dir"], reference_stream_cfg(cfg))
    total = len(stream.features) + stream.unparseable
    reference = build_reference(stream.features, stream.scores, stream.threshold,
                                unparseable_rate=stream.unparseable / total)
    # The API reads its window and limits from here, so the file is all it needs
    reference["limits"] = {k: v for k, v in cfg["monitoring"].items() if k not in ("reference", "drill")}
    save_reference(reference, ensure_parent(paths["reference"]))
    logger.info("Reference built from %d scored transactions (flag rate %.4f, unparseable %.4f)",
                reference["n"], reference["flag_rate"], reference["unparseable_rate"])


if __name__ == "__main__":
    main()
