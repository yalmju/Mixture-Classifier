"""Paired second-seed confirmation for the recovery-aware composition loss."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(r"S:\Google Drive\내 드라이브\github\Mixture Classifier")
sys.path.insert(0, str(ROOT / "documentation/scripts"))

import tune_mlp_composition_92 as tm
import tune_recovery_loss_92 as rl


def main():
    X, Y, signal, mixrows, blanks = tm.load_data()
    tm.ratio_key = lambda c: tuple(__import__("numpy").asarray(c, float))
    folds = tm.grouped_folds(mixrows)
    configs = [
        {"name": "R00", "recovery_weight": 0.0, "bias_weight": 0.0},
        {"name": "R03", "recovery_weight": 0.1, "bias_weight": 0.0},
    ]
    seed = 27100
    results = []
    for config in configs:
        started = time.time()
        print("START", config, flush=True)
        metrics, *_ = rl.run(
            config, seed, X, Y, signal, mixrows, blanks, folds
        )
        results.append({"config": config, "seed": seed, "metrics": metrics})
        print(
            "DONE {name} all={all_mae:.3f} binary={binary_mae:.3f} "
            "tight={tight:.2f} bias={bias:.2f} sec={seconds:.1f}".format(
                name=config["name"],
                all_mae=metrics["all"]["mae_pp"],
                binary_mae=metrics["binary"]["mae_pp"],
                tight=metrics["all"]["within80_120_pct"],
                bias=metrics["all"]["macro_abs_mean_bias_pct"],
                seconds=time.time() - started,
            ),
            flush=True,
        )
    output = tm.OUT / "recovery_loss_paired_seed27100.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(output, flush=True)


if __name__ == "__main__":
    main()
