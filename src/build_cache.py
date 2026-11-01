"""One-off: materialise the resized 256x256 caches used by training and eval."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import data

N_FIT = 3000   # train images actually used to fit the region model

if __name__ == "__main__":
    train, val = data.splits()
    test = data.test_names()
    rng = np.random.RandomState(data.SEED)
    fit = [train[i] for i in rng.permutation(len(train))[:N_FIT]]
    print(f"train split {len(train)} | val split {len(val)} | "
          f"test {len(test)} | fit subset {len(fit)}", flush=True)
    data.build_cache("fit", fit, "tr")
    data.build_cache("val", val, "tr")
    data.build_cache("test", test, "te")
    json.dump({"n_train_split": len(train), "n_val_split": len(val),
               "n_fit": len(fit), "n_test": len(test)},
              open(os.path.join(data.CACHE, "splits.json"), "w"), indent=2)
    print("cache done", flush=True)
