import pprint
import sys
from tqdm import tqdm, trange
import numpy as np
import os
from collections import defaultdict
from models.flash_vtg_gmr.utils.basic_utils import AverageMeter

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from training.flash_vtg_gmr.config import TestOptions
from training.flash_vtg_gmr.dataset import (
    StartEndDataset,
    start_end_collate,
    prepare_batch_inputs,
)
from training.flash_vtg_gmr.postprocessing import PostProcessorDETR
from models.flash_vtg_gmr.standalone_eval.eval import eval_submission
from models.flash_vtg_gmr.utils.basic_utils import save_jsonl, save_json

import nncore
from nncore.ops import temporal_iou

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def evaluate_unified_gmr(submission, opt, gt_data):
    """Run the repository-wide GMR evaluator and require complete split coverage."""
    from eval.eval_main import evaluate_gmr
    from eval.normalization import normalize_ground_truth

    gt, _ = normalize_ground_truth(gt_data, None, drop_empty_gt=False)
    pred_qids = {row.get("qid") for row in submission}
    gt_qids = {row.get("qid") for row in gt}
    if pred_qids != gt_qids:
        raise ValueError(
            "Strict GMR evaluation requires complete qid coverage: "
            f"pred={len(pred_qids)}, gt={len(gt_qids)}, "
            f"missing={len(gt_qids - pred_qids)}, extra={len(pred_qids - gt_qids)}"
        )

    return evaluate_gmr(
        submission,
        gt,
        k_list=(1, 3, 5),
        max_pred_windows=10,
        cls_thresholds=tuple(getattr(opt, "gmr_cls_thresholds", (0.4, 0.6))),
        gmiou_cls_threshold=float(getattr(opt, "gmiou_cls_threshold", 0.4)),
        map_num_workers=8,
        verbose=bool(getattr(opt, "debug", False)),
    )


def attach_unified_gmr_metrics(metrics, submission, opt, gt_data):
    """Attach prefixed strict metrics without changing legacy Flash-VTG keys."""
    unified = evaluate_unified_gmr(submission, opt, gt_data)
    metrics["GMR-unified"] = unified
    for name, value in unified["brief"].items():
        if isinstance(value, (int, float)):
            metrics["brief"][f"GMR-{name}"] = value
    return metrics


def combine_two_stage_existence(
    stage1_exist,
    zero_probability,
    candidate_strength,
    opt,
):
    """High-recall stage one with independent rescue and high-confidence veto."""
    if zero_probability is None:
        return stage1_exist
    verifier_exist = 1.0 - zero_probability
    if stage1_exist is None:
        return verifier_exist

    combined = torch.maximum(stage1_exist, verifier_exist)
    loose = float(getattr(opt, "exist_loose_thd", 0.35))
    rescue = float(getattr(opt, "zero_rescue_thd", 0.45))
    veto = float(getattr(opt, "zero_veto_thd", 0.65))
    weak = float(getattr(opt, "zero_weak_candidate_thd", 0.35))

    both_empty = (stage1_exist < loose) & (zero_probability >= (1.0 - rescue))
    strong_veto = (
        (stage1_exist >= loose)
        & (zero_probability >= veto)
        & (candidate_strength < weak)
    )
    conservative = torch.minimum(stage1_exist, verifier_exist)
    return torch.where(both_empty | strong_veto, conservative, combined)


def post_processing_mr_nms(mr_res, nms_thd, max_before_nms, max_after_nms, nms_type):
    mr_res_after_nms = []
    for e in mr_res:
        bnd = torch.tensor(e["pred_relevant_windows"])
        for i in range(bnd.size(0)):
            max_idx = bnd[i:, -1].argmax(dim=0)
            bnd = nncore.swap_element(bnd, i, max_idx + i)
            iou = temporal_iou(bnd[i, None, :-1], bnd[i + 1:, :-1])[0]

            if nms_type == 'normal':
                bnd[i + 1:, -1][iou >= nms_thd] = 0
            elif nms_type == 'linear':
                bnd[i + 1:, -1] *= 1 - iou
            else:
                raise ValueError(f"Unknown nms_type: {nms_type}")

        _, inds = bnd[:, -1].sort(descending=True)
        bnd = bnd[inds]
        e["pred_relevant_windows"] = bnd.tolist()

        mr_res_after_nms.append(e)
    return mr_res_after_nms


def eval_epoch_post_processing(submission, opt, gt_data, save_submission_filename):
    # IOU_THDS = (0.5, 0.7)
    logger.info("Saving/Evaluating before nms results")
    submission_path = os.path.join(opt.results_dir, save_submission_filename)
    save_jsonl(submission, submission_path)

    shared_qids = set()
    gt_aligned = []

    if opt.eval_split_name in ["val"]:  # since test_public has no GT
        metrics = eval_submission(
            submission,
            gt_data,
            verbose=opt.debug,
            match_number=not opt.debug,
            full_only=opt.eval_full_only,
            mr_only=opt.mr_only,
        )
        metrics = attach_unified_gmr_metrics(metrics, submission, opt, gt_data)

        save_metrics_path = submission_path.replace(".jsonl", "_metrics.json")
        save_json(metrics, save_metrics_path, save_pretty=True, sort_keys=False)
        latest_file_paths = [submission_path, save_metrics_path]
    else:
        metrics = None
        latest_file_paths = [
            submission_path,
        ]

    if opt.nms_thd != -1:
        logger.info("[MR] Performing nms with nms_thd {}".format(opt.nms_thd))
        submission_after_nms = post_processing_mr_nms(
            submission,
            nms_thd=opt.nms_thd,
            max_before_nms=opt.max_before_nms,
            max_after_nms=opt.max_after_nms,
            nms_type=opt.nms_type,
        )

        logger.info("Saving/Evaluating nms results")
        submission_nms_path = submission_path.replace(
            ".jsonl", "_nms_thd_{}.jsonl".format(opt.nms_thd)
        )
        save_jsonl(submission_after_nms, submission_nms_path)
        if opt.eval_split_name == "val":
            metrics_nms = eval_submission(
                submission_after_nms,
                gt_data,
                verbose=opt.debug,
                match_number=not opt.debug,
                full_only=opt.eval_full_only,
                mr_only=opt.mr_only,
            )
            metrics_nms = attach_unified_gmr_metrics(
                metrics_nms, submission_after_nms, opt, gt_data
            )
            save_metrics_nms_path = submission_nms_path.replace(
                ".jsonl", "_metrics.json"
            )
            save_json(
                metrics_nms, save_metrics_nms_path, save_pretty=True, sort_keys=False
            )
            latest_file_paths += [submission_nms_path, save_metrics_nms_path]
        else:
            metrics_nms = None
            latest_file_paths = [
                submission_nms_path,
            ]
    else:
        metrics_nms = None
    return metrics, metrics_nms, latest_file_paths

# for HL
@torch.no_grad()
def compute_hl_results(
    model, eval_loader, opt, epoch_i=None, criterion=None, tb_writer=None
):
    model.eval()
    if criterion:
        assert eval_loader.dataset.load_labels
        criterion.eval()

    loss_meters = defaultdict(AverageMeter)
    write_tb = tb_writer is not None and epoch_i is not None

    mr_res = []

    topk = 5  # top-5 map

    video_ap_collected = []
    for batch in tqdm(eval_loader, desc="compute st ed scores"):
        query_meta = batch[0]

        model_inputs, targets = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory)

        if targets is not None:
            targets["label"] = batch[0]
            bsz = int(model_inputs["src_vid"].shape[0])
            targets["fps"] = torch.full((bsz,), 1 / opt.clip_length, device=opt.device)
        else:
            targets = {}

        outputs = model(**model_inputs, targets=targets)

        preds = outputs["saliency_scores"].clone().detach()

        for meta, pred in zip(query_meta, preds):
            pred = pred
            label = meta["label"]  # raw label

            video_ap = []
            # Follow the UMT code "https://github.com/TencentARC/UMT/blob/main/datasets/tvsum.py"

            if opt.dset_name in ["tvsum"]:
                for i in range(20):
                    pred = pred.cpu()
                    cur_pred = pred[: len(label)]
                    inds = torch.argsort(cur_pred, descending=True, dim=-1)

                    # video_id = self.get_video_id(idx)
                    cur_label = torch.Tensor(label)[:, i]
                    cur_label = torch.where(cur_label > cur_label.median(), 1.0, 0.0)

                    cur_label = cur_label[inds].tolist()[:topk]

                    # if (num_gt := sum(cur_label)) == 0:
                    num_gt = sum(cur_label)
                    if num_gt == 0:
                        video_ap.append(0)
                        continue

                    hits = ap = rec = 0
                    prc = 1

                    for j, gt in enumerate(cur_label):
                        hits += gt

                        _rec = hits / num_gt
                        _prc = hits / (j + 1)

                        ap += (_rec - rec) * (prc + _prc) / 2
                        rec, prc = _rec, _prc

                    video_ap.append(ap)

            elif opt.dset_name in ["youtube_uni"]:
                cur_pred = pred[: len(label)]
                # if opt.dset_name == "tvsum_sfc":
                cur_pred = cur_pred.cpu()
                inds = torch.argsort(cur_pred, descending=True, dim=-1)

                cur_label = torch.Tensor(label).squeeze()[inds].tolist()

                num_gt = sum(cur_label)
                if num_gt == 0:
                    video_ap.append(0)
                    continue

                hits = ap = rec = 0
                prc = 1

                for j, gt in enumerate(cur_label):
                    hits += gt

                    _rec = hits / num_gt
                    _prc = hits / (j + 1)

                    ap += (_rec - rec) * (prc + _prc) / 2
                    rec, prc = _rec, _prc

                video_ap.append(float(ap))
            else:
                print("No such dataset")
                exit(-1)

            video_ap_collected.append(video_ap)

    mean_ap = np.mean(video_ap_collected)
    submmission = dict(mAP=round(mean_ap, 5))

    # tensorboard writer
    if write_tb and criterion:
        for k, v in loss_meters.items():
            tb_writer.add_scalar("Eval/{}".format(k), v.avg, epoch_i + 1)

    return submmission, loss_meters

# for MR
@torch.no_grad()
def compute_mr_results(
    model, eval_loader, opt, epoch_i=None, criterion=None, tb_writer=None
):
    model.eval()
    if criterion:
        assert eval_loader.dataset.load_labels
        criterion.eval()

    loss_meters = defaultdict(AverageMeter)
    write_tb = tb_writer is not None and epoch_i is not None

    mr_res = []
    for batch in tqdm(eval_loader, desc="compute st ed scores"):
        query_meta = batch[0]

        model_inputs, targets = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory)

        if targets is not None:
            targets["label"] = batch[0]
            bsz = int(model_inputs["src_vid"].shape[0])
            targets["fps"] = torch.full((bsz,), 1 / opt.clip_length, device=opt.device)
        else:
            targets = {}
        outputs = model(**model_inputs, targets=targets)

        # Optional two-stage GMR calibration.  Stage one is deliberately loose;
        # the independent zero verifier may rescue it, or veto only when both
        # P(null) and weak candidate evidence agree.
        pred_exist_stage1_scores = None
        if getattr(opt, "use_exist_head", False) and ("pred_exist_logits" in outputs):
            pred_exist_stage1_scores = torch.sigmoid(
                outputs["pred_exist_logits"]
            ).detach().cpu()
        pred_zero_scores = None
        if "pred_zero_logits" in outputs:
            pred_zero_scores = torch.sigmoid(
                outputs["pred_zero_logits"]
            ).detach().cpu()

        boundary_out = outputs.get("_out", {}).get("boundary", None)
        if boundary_out is not None and boundary_out.numel() > 0:
            candidate_strength = boundary_out[:, 2].detach().max().cpu().reshape(1)
        else:
            candidate_strength = torch.zeros(1)
        pred_exist_scores = combine_two_stage_existence(
            pred_exist_stage1_scores,
            pred_zero_scores,
            candidate_strength,
            opt,
        )
        if pred_exist_scores is not None and boundary_out is not None:
            thd = float(getattr(opt, "exist_gate_thd", 0.5))
            mult = torch.where(
                pred_exist_scores >= thd,
                torch.ones_like(pred_exist_scores),
                pred_exist_scores,
            )
            # Boundary decoding currently assumes an inference batch size of one.
            boundary_out = boundary_out.clone()
            boundary_out[:, 2] = boundary_out[:, 2] * float(mult[0])

        if opt.span_loss_type == "l1":
            _bnd = boundary_out if boundary_out is not None else outputs["_out"]["boundary"]
            scores = _bnd[:, 2]
            pred_spans = _bnd[:, :2].unsqueeze(0)
            _saliency_scores = outputs["_out"]["saliency"].unsqueeze(0)

            saliency_scores = []
            valid_vid_lengths = outputs["_out"]["video_msk"].sum(1).cpu().tolist()
            for j in range(len(valid_vid_lengths)):
                ss = _saliency_scores[j, : int(valid_vid_lengths[j])].tolist()
                ss = [float(f"{e:.3f}") for e in ss]
                saliency_scores.append(ss)
        else:
            bsz, n_queries = outputs["pred_spans"].shape[
                :2
            ]  # # (bsz, #queries, max_v_l *2)
            pred_spans_logits = outputs["pred_spans"].view(
                bsz, n_queries, 2, opt.max_v_l
            )
            pred_span_scores, pred_spans = F.softmax(pred_spans_logits, dim=-1).max(
                -1
            )  # 2 * (bsz, #queries, 2)
            scores = torch.prod(pred_span_scores, 2)  # (bsz, #queries)
            pred_spans[:, 1] += 1
            pred_spans *= opt.clip_length

        # compose predictions
        for idx, (meta, spans, score) in enumerate(
            zip(query_meta, pred_spans.cpu(), scores.cpu())
        ):
            spans_src = boundary_out if boundary_out is not None else outputs["_out"]["boundary"]
            spans = torch.clamp(spans_src, 0, meta["duration"])
            cur_ranked_preds = spans.tolist()
            cur_ranked_preds = [
                [float(f"{e:.3f}") for e in row] for row in cur_ranked_preds
            ]
            cur_query_pred = dict(
                qid=meta["qid"],
                query=meta["query"],
                vid=meta["vid"],
                pred_relevant_windows=cur_ranked_preds,
            )
            # Only include saliency outputs when running HL-style evaluation.
            # For MR-only/GMR usage, GT typically has no saliency fields, so omit this to keep submission minimal.
            if not getattr(opt, "mr_only", False):
                cur_query_pred["pred_saliency_scores"] = saliency_scores[idx]
            if pred_exist_scores is not None:
                cur_query_pred["pred_exist_score"] = float(f"{float(pred_exist_scores[idx]):.3f}")
            if pred_exist_stage1_scores is not None:
                cur_query_pred["pred_exist_score_stage1"] = float(
                    f"{float(pred_exist_stage1_scores[idx]):.3f}"
                )
            if pred_zero_scores is not None:
                cur_query_pred["pred_zero_score"] = float(
                    f"{float(pred_zero_scores[idx]):.3f}"
                )
            mr_res.append(cur_query_pred)

        loss_dict = {k: v for k, v in outputs.items() if 'loss' in k}
        losses = sum(loss_dict.values())
        loss_dict["loss_overall"] = float(losses)  # for logging only
        for k, v in loss_dict.items():
            loss_meters[k].update(
                float(v)
            )

    if write_tb and len(loss_meters) != 1:
        for k, v in loss_meters.items():
            tb_writer.add_scalar("Eval/{}".format(k), v.avg, epoch_i + 1)

    if opt.dset_name in ["hl"]:
        post_processor = PostProcessorDETR(
            clip_length=opt.clip_length,
            min_ts_val=0,
            max_ts_val=150,
            min_w_l=2,
            max_w_l=150,
            move_window_method="left",
            process_func_names=("clip_ts", "round_multiple"),
        )
    elif opt.dset_name in ["charadesSTA"]:
        if opt.v_feat_dim == 4096:  # vgg
            post_processor = PostProcessorDETR(
                clip_length=opt.clip_length,
                min_ts_val=0,
                max_ts_val=360,
                min_w_l=12,
                max_w_l=360,
                move_window_method="left",
                process_func_names=("clip_ts", "round_multiple"),
            )
        else:
            post_processor = PostProcessorDETR(
                clip_length=opt.clip_length,
                min_ts_val=0,
                max_ts_val=150,
                min_w_l=2,
                max_w_l=60,
                move_window_method="left",
                process_func_names=("clip_ts", "round_multiple"),
            )
    else:
        post_processor = PostProcessorDETR(
            clip_length=opt.clip_length,
            min_ts_val=0,
            max_ts_val=50000,
            min_w_l=0,
            max_w_l=50000,
            move_window_method="left",
            process_func_names=(["round_multiple"]),
        )

    mr_res = post_processor(mr_res)
    return mr_res, loss_meters


def get_eval_res(model, eval_loader, opt, epoch_i, criterion, tb_writer):
    """compute and save query and video proposal embeddings"""
    eval_res, eval_loss_meters = compute_mr_results(
        model, eval_loader, opt, epoch_i, criterion, tb_writer
    )  # list(dict)
    return eval_res, eval_loss_meters


def eval_epoch(
    model,
    eval_dataset,
    opt,
    save_submission_filename,
    epoch_i=None,
    criterion=None,
    tb_writer=None,
):
    logger.info("Generate submissions")
    model.eval()
    if criterion is not None and eval_dataset.load_labels:
        criterion.eval()
    else:
        criterion = None

    if opt.dset_name == "tacos":
        shuffle = True
    else:
        shuffle = False

    eval_loader = DataLoader(
        eval_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.eval_bsz,
        num_workers=opt.num_workers,
        shuffle=shuffle,
        pin_memory=opt.pin_memory,
    )

    # tvsum
    if opt.dset_name in ["tvsum", "youtube_uni"]:
        metrics, eval_loss_meters = compute_hl_results(
            model, eval_loader, opt, epoch_i, criterion, tb_writer
        )

        # to match original save format
        submission = [{"brief": metrics}]
        submission_path = os.path.join(opt.results_dir, "latest_metric.jsonl")
        save_jsonl(submission, submission_path)

        return submission[0], submission[0], eval_loss_meters, [submission_path]

    else:
        submission, eval_loss_meters = get_eval_res(
            model, eval_loader, opt, epoch_i, criterion, tb_writer
        )

        if opt.dset_name in ["charadesSTA", "tacos", "nlq"]:
            new_submission = []
            for s in submission:
                s.pop("pred_saliency_scores", None)
                new_submission.append(s)
            submission = new_submission

        metrics, metrics_nms, latest_file_paths = eval_epoch_post_processing(
            submission, opt, eval_dataset.data, save_submission_filename
        )
        return metrics, metrics_nms, eval_loss_meters, latest_file_paths


def setup_model(opt):
    """setup model/optimizer/scheduler and load checkpoints when needed"""
    logger.info("setup model/optimizer/scheduler")
    from models.flash_vtg_gmr.model import build_model1
    model, criterion = build_model1(opt)

    if bool(getattr(opt, "freeze_parent", False)):
        for parameter in model.parameters():
            parameter.requires_grad = False
        if bool(getattr(opt, "use_quality_head", False)) and not bool(
            getattr(opt, "freeze_quality_head", False)
        ):
            for parameter in model.quality_head.parameters():
                parameter.requires_grad = True
        if bool(getattr(opt, "use_independent_zero_head", False)):
            for parameter in model.zero_verifier_head.parameters():
                parameter.requires_grad = True
        if not any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError(
                "--freeze_parent requires at least one trainable Quality/Zero head"
            )
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)
        criterion.to(opt.device)

    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad],
            "lr": opt.lr,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=opt.lr, weight_decay=opt.wd)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_drop, gamma=0.5)
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-4)

    if opt.resume_adapter is not None:
        logger.info(f"Load adapter checkpoint from {opt.resume_adapter}")
        # Flash-VTG checkpoints contain argparse.Namespace metadata.  PyTorch
        # 2.6+ defaults to weights_only=True, which cannot deserialize that
        # legacy but locally produced/trusted format.
        adapter_checkpoint = torch.load(opt.resume_adapter, weights_only=False)
        adapter_state_dict = {k: v for k, v in adapter_checkpoint['state_dict'].items() if k.startswith('adapter')}
        model.load_state_dict(adapter_state_dict, strict=False)

    if opt.resume is not None:
        logger.info(f"Load checkpoint from {opt.resume}")
        checkpoint = torch.load(opt.resume, map_location="cpu", weights_only=False)

        from collections import OrderedDict

        state = checkpoint.get("model", checkpoint.get("state_dict"))
        if state is None:
            raise KeyError("Checkpoint must contain 'model' or 'state_dict'")
        if any(k.startswith("module.") for k in state.keys()):
            new_state_dict = OrderedDict()
            for k, v in state.items():
                name = k[7:] if k.startswith("module.") else k
                new_state_dict[name] = v
            state = new_state_dict

        if bool(getattr(opt, "allow_head_init", False)):
            incompatible = model.load_state_dict(state, strict=False)
            allowed_prefixes = []
            if bool(getattr(opt, "use_quality_head", False)):
                allowed_prefixes.append("quality_head.")
            if bool(getattr(opt, "use_independent_zero_head", False)):
                allowed_prefixes.append("zero_verifier_head.")
            disallowed_missing = [
                key
                for key in incompatible.missing_keys
                if not any(key.startswith(prefix) for prefix in allowed_prefixes)
            ]
            if disallowed_missing or incompatible.unexpected_keys:
                raise RuntimeError(
                    "Checkpoint mismatch outside requested new heads: "
                    f"missing={disallowed_missing}, "
                    f"unexpected={incompatible.unexpected_keys}"
                )
            logger.info(
                "Initialized requested new head keys: %s",
                incompatible.missing_keys,
            )
        else:
            model.load_state_dict(state, strict=True)
        if opt.resume_all:
            optimizer.load_state_dict(checkpoint["optimizer"])
            # Older Flash runs called StepLR.step(loss_tensor), which serialized
            # differentiable tensors into both optimizer LR fields and scheduler
            # counters.  Normalize those values before exact continuation.
            for group in optimizer.param_groups:
                for key in ("lr", "initial_lr"):
                    value = group.get(key)
                    if torch.is_tensor(value):
                        group[key] = float(value.detach().cpu().item())
            scheduler_state = dict(checkpoint["lr_scheduler"])
            scheduler_state["last_epoch"] = int(checkpoint["epoch"]) + 1
            scheduler_state["_last_lr"] = [
                float(value.detach().cpu().item()) if torch.is_tensor(value)
                else float(value)
                for value in scheduler_state.get(
                    "_last_lr", [group["lr"] for group in optimizer.param_groups]
                )
            ]
            lr_scheduler.load_state_dict(scheduler_state)
            opt.start_epoch = checkpoint["epoch"] + 1
    else:
        logger.warning(
            "If you intend to evaluate the model, please specify --resume with ckpt path"
        )

    return model, criterion, optimizer, lr_scheduler


def start_inference(train_opt=None, split=None, splitfile=None):
    if train_opt is not None:
        opt = TestOptions().parse(train_opt.a_feat_dir)
    else:
        opt = TestOptions().parse()
    if split is not None:
        opt.eval_split_name = split
    if splitfile is not None:
        opt.eval_path = splitfile

    opt.cfg = nncore.Config.from_file(opt.config)

    print(opt.eval_split_name)
    print(opt.eval_path)
    logger.info("Setup config, data and model...")

    cudnn.benchmark = True
    cudnn.deterministic = False

    assert opt.eval_path is not None
    if opt.eval_split_name == "val":
        loadlabel = True
    else:
        loadlabel = False

    eval_dataset = StartEndDataset(
        dset_name=opt.dset_name,
        data_path=opt.eval_path,
        v_feat_dirs=opt.v_feat_dirs,
        q_feat_dir=opt.t_feat_dir,
        q_feat_type=opt.q_feat_type,
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        ctx_mode=opt.ctx_mode,
        data_ratio=opt.data_ratio,
        normalize_v=not opt.no_norm_vfeat,
        normalize_t=not opt.no_norm_tfeat,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        load_labels=loadlabel,  # opt.eval_split_name == "val",
        span_loss_type=opt.span_loss_type,
        txt_drop_ratio=0,
        dset_domain=opt.dset_domain,
        mr_only=opt.mr_only,
        # Strict GMR evaluation requires predictions for every query, including
        # null queries.  Plain Flash-VTG uses its maximum window score as the
        # existence proxy in eval/eval_main.py.
        keep_empty_gt=True,
    )
    model, criterion, _, _ = setup_model(opt)
    save_submission_filename = "hl_{}_submission.jsonl".format(opt.eval_split_name)

    logger.info("Starting inference...")
    with torch.no_grad():
        metrics_no_nms, metrics_nms, eval_loss_meters, latest_file_paths = eval_epoch(
            model, eval_dataset, opt, save_submission_filename, criterion=criterion
        )
    if opt.eval_split_name == "val":
        logger.info(
            "metrics_no_nms {}".format(
                pprint.pformat(metrics_no_nms["brief"], indent=4)
            )
        )
    if metrics_nms is not None:
        logger.info(
            "metrics_nms {}".format(pprint.pformat(metrics_nms["brief"], indent=4))
        )


if __name__ == "__main__":
    start_inference()
