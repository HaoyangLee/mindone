#!/usr/bin/env python
import argparse
import os
import sys
# sys.path.append("../")
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from infer_engine.utils.export import LiteConverter
from infer_engine.utils.logger import setup_logger


def main():
    _logger = setup_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--names", nargs="+", help="convert the model from MindIR to Mindspore Lite MindIR with the given name"
    )
    parser.add_argument("--root", default="models/mindir", help="Convert all MindIR in root folder")
    args = parser.parse_args()

    lite_converter = LiteConverter(target="ascend")

    if args.names:
        for name in args.names:
            lite_converter(name, model_save_path="./models_256/lite", mindir_root="./models_256/mindir")
    elif args.root:
        names = sorted(os.listdir(args.root))
        names = [x.replace(".mindir", "") for x in names if x.endswith(".mindir")]
        names = [x.replace("_graph", "") for x in names]
        _logger.info(f"converting {names}")
        for name in names:
            lite_converter(name, model_save_path="./models_256/lite", mindir_root="./models_256/mindir")


if __name__ == "__main__":
    main()
