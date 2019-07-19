import os
import sys
import argparse
import pickle

import json

print("System Paths:", sys.path)
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from collections import defaultdict

from maskrcnn_benchmark.config import cfg


def plot(output, cfg):
    validation_set = cfg.DATASETS.VAL
    save_dir = cfg.OUTPUT_DIR
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    its = list(sorted(output.keys()))
    print("Number of iterations saved:", len(its))
    print("Datasets", output[its[0]].keys())

    for part in ['AP', 'AP50']:
        server = defaultdict(list)
        for i in its:
            for mode in validation_set:
                outset = output[i][mode][part]
                server[mode].append((i, outset))
        scatter(server, part, save_dir)


def scatter(server, part, save_dir):
    file_path = os.path.join(save_dir, part + "map.pdf")

    # colors = ['r-', 'b', 'g']
    modes = list(server.keys())

    print("All Training Modes", modes)
    for i, mode in enumerate(modes):
        its, acc = zip(*(server[mode]))
        # print(its, acc, mode)
        dict = {"iter": its, "AP": acc}
        json_path = os.path.join(save_dir, part + ".json")
        with open(json_path, 'w') as file:
            json.dump(dict, file, indent=2)
        plt.plot(its, acc)  # .split("_")[1])
    # plt.title(("Average Precision:"))
    pylab.rcParams['font.size'] = 11
    plt.xlabel("Iterationen")
    if part == 'AP':
        plt.ylabel(r'AP$_{50 \dots 95}^{box}$')
    else:
        plt.ylabel(r'AP$_{50}^{box}$')
    plt.savefig(file_path, dvi=1000)
    plt.close()
    print("SAVED PDF OF RESULTS", file_path)


def getCGF(args):
    from maskrcnn_benchmark.config import cfg
    cfg.merge_from_file(args.config)
    cfg.OUTPUT_DIR = os.path.join("output", cfg.OUTPUT_DIR, "val")
    assert os.path.exists(cfg.OUTPUT_DIR)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config",
        default="/home/daniel.rose/maskrcnn-benchmark/configs/sidewalk.fpn.bs1.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        "--home",
        default="/home/daniel.rose/maskrcnn-benchmark",
        metavar="FILE",
        help="path to root directory",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    with open('tmp_result.pkl', 'rb') as handle:
        output = pickle.load(handle)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    plot(output, cfg)
