"""Writes every kernel directory in oct21/ from the master scripts and plan.json
(the oct3/build_kernels.py scheme). Run after editing a master:
    python oct21/build_kernels.py
"""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "oct21")
RUNNER = "kaggle_oct3_runner/runner.py"
IMNET = "kaggle_imnet_fog/imnet_fog.py"
IMNET_MODELS = {"resnet50-v1": "resnet50_v1", "resnet50-v2": "resnet50_v2", "efficientnet-b0": "efficientnet_b0",
                "vit-b-16": "vit_b_16", "convnext-tiny": "convnext_tiny", "swin-t": "swin_t"}


def build(account, name):
    if name.startswith("imnet-fog-"):
        master, model = IMNET, IMNET_MODELS[name[len("imnet-fog-"):].removesuffix("-r2")]
        src = open(os.path.join(ROOT, master)).read()
        line = 'MODEL = "resnet50_v1"'
        assert src.count(line) == 1
        src = src.replace(line, f'MODEL = "{model}"')
        datasets, desc = ["sautkin/imagenet1kvalid"], f"model={model}"
    else:
        seed = int(re.fullmatch(r"blur-joint-s(\d+)", name).group(1))
        master = RUNNER
        src = open(os.path.join(ROOT, master)).read()
        for old, new in (('SEED = int(os.environ.get("RUN_SEED", "2"))', f'SEED = int(os.environ.get("RUN_SEED", "{seed}"))'),
                         ('JOB = "adam_main"', 'JOB = "blur_joint"')):
            assert src.count(old) == 1, old
            src = src.replace(old, new)
        datasets, desc = [], f"seed={seed} job=blur_joint"
    d = os.path.join(HERE, name)
    os.makedirs(d, exist_ok=True)
    for f in os.listdir(d):
        if f.endswith(".py"):
            os.remove(os.path.join(d, f))
    code_file = os.path.basename(master)
    open(os.path.join(d, code_file), "w").write(src)
    meta = dict(id=f"{account}/{name}", title=name, code_file=code_file, language="python",
                kernel_type="script", is_private=True, enable_gpu=True, enable_internet=True,
                dataset_sources=datasets, competition_sources=[], kernel_sources=[])
    json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=2)
    return f"{account:<11} {name:<28} {code_file:<14} {desc}"


def main():
    plan = json.load(open(os.path.join(HERE, "plan.json")))
    for key, queue in plan.items():
        account = key.split(":")[0]          # "account3:r2" is a second queue of account3
        for name in queue:
            print(build(account, name))


if __name__ == "__main__":
    main()
