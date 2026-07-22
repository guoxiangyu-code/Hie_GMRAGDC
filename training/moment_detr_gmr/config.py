from __future__ import annotations

import os
import shutil
from pathlib import Path

import yaml
from easydict import EasyDict

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "configs" / "moment_detr_gmr"

class BaseOptions:
    """Load the minimal Moment-DETR-GMR configuration stack."""

    def __init__(self, model: str, dataset: str, feature: str, resume: str | None = None):
        self.model = model
        self.dataset = dataset
        self.feature = feature
        self.resume = resume
        self.opt = {}

    @property
    def option(self):
        if not self.opt:
            raise RuntimeError("option is empty. Did you run parse()?")
        return self.opt

    def update(self, yaml_file: Path) -> None:
        with yaml_file.open("r", encoding="utf-8") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        if data:
            self.opt.update(data)

    def parse(self) -> None:
        cfgs = [
            CONFIG_ROOT / "base.yml",
            CONFIG_ROOT / "feature" / f"{self.feature}.yml",
            CONFIG_ROOT / "model" / f"{self.model}.yml",
            CONFIG_ROOT / "dataset" / f"{self.dataset}.yml",
        ]
        for cfg in cfgs:
            if not cfg.exists():
                raise FileNotFoundError(f"Missing config file: {cfg}")
            self.update(cfg)

        self.opt = EasyDict(self.opt)

        results_root = Path(self.opt.results_dir)
        if not results_root.is_absolute():
            results_root = REPO_ROOT / results_root

        if self.resume:
            results_dir = results_root / self.model / f"{self.dataset}_finetune" / self.feature
        else:
            results_dir = results_root / self.model / self.dataset / self.feature

        self.opt.results_dir = str(results_dir)
        self.opt.ckpt_filepath = str(results_dir / self.opt.ckpt_filename)
        self.opt.train_log_filepath = str(results_dir / self.opt.train_log_filename)
        self.opt.eval_log_filepath = str(results_dir / self.opt.eval_log_filename)

        # Prefer the released Soccer-GMR feature bundle when it is mounted in
        # the workspace.  Fall back to the layout documented by this repo.
        released_root = REPO_ROOT / "Soccer-GMR" / "feature" / "standard"
        feature_root = (
            released_root
            if released_root.exists()
            else REPO_ROOT / "features" / self.dataset
        )
        self.opt.v_feat_dirs = [
            str(feature_root / "clip"),
            str(feature_root / "slowfast"),
        ]
        self.opt.t_feat_dir = str(feature_root / "clip_text")
        self.opt.a_feat_dirs = None
        self.opt.a_feat_types = None
        self.opt.t_feat_dir_pretrain_eval = None

    def clean_and_makedirs(self, overwrite: bool = False) -> None:
        if "results_dir" not in self.opt:
            raise RuntimeError("results_dir is not set. Did you run parse()?")
        if overwrite and os.path.exists(self.opt.results_dir):
            shutil.rmtree(self.opt.results_dir)
        os.makedirs(self.opt.results_dir, exist_ok=True)
