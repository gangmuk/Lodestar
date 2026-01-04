#!/usr/bin/env python3

"""
Model Explainability Suite for RL-based Request Router
-----------------------------------------------------
This script loads a trained policy from a model directory and performs
systematic explainability analyses:

- Dimension inference from a reference tensor dataset (or CLI overrides)
- PDP/ICE for per-pod features, KV hit ratios, and request features
- Gradient and Integrated Gradients saliency attribution
- Pod permutation (symmetry/invariance) tests
- Counterfactual (minimal change) analysis to flip routing decisions
- Two-way interaction heatmaps

Outputs are saved under: <final_model_dir>/xai_report

Usage example:
  python model_explainability.py \
    --final_model_dir \
      /users/gangmuk/projects/aibrix-gangmuk/benchmarks/data-driven-routing/routing-agent-service/training_data/SharingRatio71%/all/final_model-data_replaced-processed-linear_simple-lr_0.001-ttft_weight_2.0-ttftslo_1000-avgtpotslo_50 \
    --search_data_roots \
      /users/gangmuk/projects/aibrix-gangmuk/benchmarks/data-driven-routing/routing-agent-service/training_data/SharingRatio71%/all

If a reference tensor dataset (tensor_dataset.pt) is found under search roots,
dimensions are inferred automatically. Otherwise, you can pass overrides:
  --num_pods 7 --pod_feat_dim 10 --kv_dim 1 --req_feat_dim 3
"""

import os
import sys
import json
import math
import glob
import argparse
import random
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch  # type: ignore
import torch.nn.functional as F  # type: ignore
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages  # type: ignore


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))

# Ensure agent code modules are importable
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from logger import logger  # uses existing project logger
import simpler_contextual_bandit as scb


def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _find_reference_tensor(search_roots: Optional[str]) -> Optional[str]:
    if not search_roots:
        return None
    candidates = []
    for root in str(search_roots).split(":"):
        if not root or not os.path.exists(root):
            continue
        # Common locations: encoded_data/**/tensor_dataset.pt or batch_*/(train/)?tensor_dataset.pt
        patterns = [
            os.path.join(root, "**", "tensor_dataset.pt"),
            os.path.join(root, "encoded_data", "**", "tensor_dataset.pt"),
        ]
        for pat in patterns:
            candidates.extend(glob.glob(pat, recursive=True))
    # Prefer the most recently modified
    if candidates:
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]
    return None


def _load_model_config(final_model_dir: str) -> Dict[str, Any]:
    cfg_path = os.path.join(final_model_dir, "model_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        return cfg
    logger.warning(f"model_config.json not found in {final_model_dir}, using minimal defaults")
    return {
        "hidden_dim": 64,
        "weight_initialization": "xavier",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "entropy_bonus_factor": 0.02,
        "batch_size": 64,
        "explore": False,
        "exploration_rate": 0.0,
    }


def _infer_dims_from_tensor(tensor_path: str) -> Tuple[int, int, int, int]:
    data = torch.load(tensor_path, map_location="cpu")
    pod = data["pod_features_with_staleness"]
    kv = data["kv_hit_ratios"]
    req = data["request_features"]
    num_pods = int(pod.shape[1])
    pod_feat_dim = int(pod.shape[2])
    kv_dim = int(kv.shape[2])
    req_feat_dim = int(req.shape[1])
    return num_pods, pod_feat_dim, kv_dim, req_feat_dim


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_agent(model_cfg: Dict[str, Any], num_pods: int, pod_feat_dim: int, kv_dim: int, req_feat_dim: int, final_model_dir: str) -> scb.SimplifiedContextualBandit:
    state_dim = {
        "pod_features": pod_feat_dim,
        "kv_hit_ratios": kv_dim,
        "request_features": req_feat_dim,
        "num_pods": num_pods,
    }
    H = dict(model_cfg)
    if "learning_rate" not in H:
        H["learning_rate"] = 1e-3
    if "weight_decay" not in H:
        H["weight_decay"] = 1e-4
    if "entropy_bonus_factor" not in H:
        H["entropy_bonus_factor"] = 0.02
    if "hidden_dim" not in H:
        H["hidden_dim"] = 64
    if "weight_initialization" not in H:
        H["weight_initialization"] = "xavier"
    agent = scb.SimplifiedContextualBandit(state_dim, num_pods, H, final_model_dir=final_model_dir)
    agent.load(final_model_dir)
    agent.policy.eval()
    return agent


def _to_batch(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        return x.unsqueeze(0)
    return x


def _forward_probs(agent: scb.SimplifiedContextualBandit, pod: torch.Tensor, kv: torch.Tensor, req: torch.Tensor) -> torch.Tensor:
    pod = _to_batch(pod).to(_device())
    kv = _to_batch(kv).to(_device())
    req = _to_batch(req).to(_device())
    with torch.no_grad():
        probs = agent.policy(pod, kv, req)
    return probs


def _forward_scores(agent: scb.SimplifiedContextualBandit, pod: torch.Tensor, kv: torch.Tensor, req: torch.Tensor) -> torch.Tensor:
    # Reconstruct raw scores before softmax, using the model internals
    pod = _to_batch(pod).to(_device())
    kv = _to_batch(kv).to(_device())
    req = _to_batch(req).to(_device())
    batch_size, num_pods = pod.shape[0], pod.shape[1]
    combined = torch.cat([pod, kv], dim=2)
    exp_req = req.unsqueeze(1).expand(-1, num_pods, -1)
    full = torch.cat([combined, exp_req], dim=2)
    flat = full.view(batch_size * num_pods, -1)
    with torch.no_grad():
        scores = agent.policy.pod_scorer(flat).view(batch_size, num_pods)
    return scores


def _feature_ranges_from_reference(tensor_path: str) -> Dict[str, np.ndarray]:
    data = torch.load(tensor_path, map_location="cpu")
    pod = data["pod_features_with_staleness"].numpy()
    kv = data["kv_hit_ratios"].numpy()
    req = data["request_features"].numpy()

    # Compute per-dimension min/max
    ranges = {
        "pod_min": np.min(pod, axis=(0, 1)),
        "pod_max": np.max(pod, axis=(0, 1)),
        "kv_min": np.min(kv, axis=(0, 1)),
        "kv_max": np.max(kv, axis=(0, 1)),
        "req_min": np.min(req, axis=0),
        "req_max": np.max(req, axis=0),
    }
    return ranges


def _load_feature_names(reference_tensor: Optional[str], pod_feat_dim: int, kv_dim: int, req_feat_dim: int) -> Dict[str, list]:
    # Strict mode: require metadata.json and matching dimensions
    assert reference_tensor and os.path.exists(reference_tensor), "reference_tensor not found; cannot infer feature names"
    meta_path = os.path.join(os.path.dirname(reference_tensor), "metadata.json")
    assert os.path.exists(meta_path), f"metadata.json not found next to reference tensor: {meta_path}"

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    # Pod feature names (without staleness in metadata). If tensor has +1, it is staleness.
    pod_from_meta = meta.get('pod_features_list')
    assert isinstance(pod_from_meta, list) and len(pod_from_meta) > 0, "Invalid pod_features_list in metadata.json"
    if len(pod_from_meta) == pod_feat_dim:
        pod_names = list(pod_from_meta)
    elif len(pod_from_meta) + 1 == pod_feat_dim:
        pod_names = list(pod_from_meta) + ["staleness"]
    else:
        raise AssertionError(
            f"Pod feature dim mismatch: tensor={pod_feat_dim}, metadata={len(pod_from_meta)} (expected equal or +1 for staleness)"
        )

    # Request feature names must match exactly
    req_from_meta = meta.get('numeric_request_features')
    assert isinstance(req_from_meta, list), "numeric_request_features missing in metadata.json"
    assert len(req_from_meta) == req_feat_dim, f"Request feature dim mismatch: tensor={req_feat_dim}, metadata={len(req_from_meta)}"
    req_names = list(req_from_meta)

    # KV names: deterministic
    kv_names = ["kv_hit_ratio"] if kv_dim == 1 else [f"kv_feat_{i}" for i in range(kv_dim)]

    return {"pod": pod_names, "kv": kv_names, "req": req_names}


def _select_base_samples(tensor_path: Optional[str], num_samples: int, num_pods: int, pod_feat_dim: int, kv_dim: int, req_feat_dim: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if tensor_path and os.path.exists(tensor_path):
        data = torch.load(tensor_path, map_location="cpu")
        pod_all = data["pod_features_with_staleness"]
        kv_all = data["kv_hit_ratios"]
        req_all = data["request_features"]
        n = pod_all.shape[0]
        idx = torch.randperm(n)[: min(num_samples, n)]
        return pod_all[idx].to(_device()), kv_all[idx].to(_device()), req_all[idx].to(_device())

    # Synthetic fallback: zero-mean unit-range features
    pod = torch.zeros((num_samples, num_pods, pod_feat_dim), dtype=torch.float32)
    kv = torch.zeros((num_samples, num_pods, kv_dim), dtype=torch.float32)
    req = torch.zeros((num_samples, req_feat_dim), dtype=torch.float32)
    return pod.to(_device()), kv.to(_device()), req.to(_device())


def gradient_saliency(agent: scb.SimplifiedContextualBandit, pod: torch.Tensor, kv: torch.Tensor, req: torch.Tensor, topk: int = 10, target_pod: int = 0) -> Dict[str, Any]:
    pod = pod.clone().detach().requires_grad_(True)
    kv = kv.clone().detach().requires_grad_(True)
    req = req.clone().detach().requires_grad_(True)
    probs = agent.policy(pod, kv, req)
    # FIX: explain a consistent scalar - probability of a fixed target pod
    target_pod = int(target_pod)
    chosen = probs[:, target_pod]
    chosen.sum().backward()

    # FIX: focus on target pod's feature channels to align with PDP interpretation
    pod_grad_mean = pod.grad[:, target_pod, :].mean(dim=0).detach().cpu().numpy()
    kv_grad_mean = kv.grad[:, target_pod, :].mean(dim=0).detach().cpu().numpy()
    req_grad_mean = req.grad.mean(dim=0).detach().cpu().numpy()

    # Magnitudes for top-k ranking (absolute gradients) restricted to target pod
    pod_importance = pod.grad[:, target_pod, :].abs().mean(dim=0).detach().cpu().numpy()
    kv_importance = kv.grad[:, target_pod, :].abs().mean(dim=0).detach().cpu().numpy()
    req_importance = req.grad.abs().mean(dim=0).detach().cpu().numpy()

    def _topk(arr, k):
        idx = np.argsort(-arr)[: min(k, len(arr))]
        return [(int(i), float(arr[i])) for i in idx]

    result = {
        "pod_feat_importance_topk": _topk(pod_importance, topk),
        "kv_feat_importance": _topk(kv_importance, kv_importance.shape[0]),
        "req_feat_importance_topk": _topk(req_importance, topk),
        # Signed gradients for coloring by direction
        "pod_grad_mean": pod_grad_mean.tolist(),
        "kv_grad_mean": kv_grad_mean.tolist(),
        "req_grad_mean": req_grad_mean.tolist(),
    }
    return result


def integrated_gradients(agent: scb.SimplifiedContextualBandit, pod: torch.Tensor, kv: torch.Tensor, req: torch.Tensor, steps: int = 32, target_pod: int = 0) -> Dict[str, Any]:
    # Baseline: zeros
    pod0 = torch.zeros_like(pod)
    kv0 = torch.zeros_like(kv)
    req0 = torch.zeros_like(req)

    # Deltas from baseline
    d_pod = (pod - pod0).detach()
    d_kv = (kv - kv0).detach()
    d_req = (req - req0).detach()

    # Accumulate signed gradients along the path (per-sample), then average
    B = pod.shape[0]
    pod_grad_sum = torch.zeros((B, pod.shape[2]), device=_device())
    kv_grad_sum = torch.zeros((B, kv.shape[2]), device=_device())
    req_grad_sum = torch.zeros((B, req.shape[1]), device=_device())

    for alpha in np.linspace(0.0, 1.0, steps, endpoint=True):
        pod_alpha = (pod0 + alpha * d_pod).clone().detach().requires_grad_(True)
        kv_alpha = (kv0 + alpha * d_kv).clone().detach().requires_grad_(True)
        req_alpha = (req0 + alpha * d_req).clone().detach().requires_grad_(True)
        probs = agent.policy(pod_alpha, kv_alpha, req_alpha)
        chosen = probs[:, int(target_pod)]
        grads = torch.autograd.grad(chosen.sum(), [pod_alpha, kv_alpha, req_alpha], allow_unused=True)
        # Restrict to target pod channels
        g_pod = grads[0][:, int(target_pod), :] if grads[0] is not None else torch.zeros_like(pod_grad_sum)
        g_kv = grads[1][:, int(target_pod), :] if grads[1] is not None else torch.zeros_like(kv_grad_sum)
        g_req = grads[2] if grads[2] is not None else torch.zeros_like(req_grad_sum)
        pod_grad_sum = pod_grad_sum + g_pod
        kv_grad_sum = kv_grad_sum + g_kv
        req_grad_sum = req_grad_sum + g_req

    # Riemann approximation: average gradient along path times delta
    # Per-sample delta restricted to target pod
    d_pod_tp = d_pod[:, int(target_pod), :]
    d_kv_tp = d_kv[:, int(target_pod), :]
    # Compute per-sample attribution then average across batch
    pod_ig_signed = ((pod_grad_sum / steps) * d_pod_tp).mean(dim=0)
    kv_ig_signed = ((kv_grad_sum / steps) * d_kv_tp).mean(dim=0)
    req_ig_signed = ((req_grad_sum / steps) * d_req).mean(dim=0)

    # Also compute absolute-IG magnitude like the previous version (average of |grad| along path)
    pod_grad_abs_sum = torch.zeros((B, pod.shape[2]), device=_device())
    kv_grad_abs_sum = torch.zeros((B, kv.shape[2]), device=_device())
    req_grad_abs_sum = torch.zeros((B, req.shape[1]), device=_device())
    for alpha in np.linspace(0.0, 1.0, steps, endpoint=True):
        pod_alpha = (pod0 + alpha * d_pod).clone().detach().requires_grad_(True)
        kv_alpha = (kv0 + alpha * d_kv).clone().detach().requires_grad_(True)
        req_alpha = (req0 + alpha * d_req).clone().detach().requires_grad_(True)
        probs = agent.policy(pod_alpha, kv_alpha, req_alpha)
        chosen = probs[:, int(target_pod)]
        grads = torch.autograd.grad(chosen.sum(), [pod_alpha, kv_alpha, req_alpha], allow_unused=True)
        g_pod = grads[0][:, int(target_pod), :].abs() if grads[0] is not None else torch.zeros_like(pod_grad_abs_sum)
        g_kv = grads[1][:, int(target_pod), :].abs() if grads[1] is not None else torch.zeros_like(kv_grad_abs_sum)
        g_req = grads[2].abs() if grads[2] is not None else torch.zeros_like(req_grad_abs_sum)
        pod_grad_abs_sum = pod_grad_abs_sum + g_pod
        kv_grad_abs_sum = kv_grad_abs_sum + g_kv
        req_grad_abs_sum = req_grad_abs_sum + g_req

    pod_ig_abs = ((pod_grad_abs_sum / steps).mean(dim=0)).detach().cpu().numpy()
    kv_ig_abs = ((kv_grad_abs_sum / steps).mean(dim=0)).detach().cpu().numpy()
    req_ig_abs = ((req_grad_abs_sum / steps).mean(dim=0)).detach().cpu().numpy()

    return {
        # Preserve previous key for magnitude chart compatibility
        "pod_feat_ig": pod_ig_abs.tolist(),
        "kv_feat_ig": kv_ig_abs.tolist(),
        "req_feat_ig": req_ig_abs.tolist(),
        # Add signed IG for coloring/direction
        "pod_feat_ig_signed": pod_ig_signed.detach().cpu().numpy().tolist(),
        "kv_feat_ig_signed": kv_ig_signed.detach().cpu().numpy().tolist(),
        "req_feat_ig_signed": req_ig_signed.detach().cpu().numpy().tolist(),
    }


def permutation_symmetry_test(agent: scb.SimplifiedContextualBandit, pod: torch.Tensor, kv: torch.Tensor, req: torch.Tensor, trials: int = 20) -> Dict[str, Any]:
    n = pod.shape[1]
    agree = 0
    mean_l1 = 0.0
    for _ in range(trials):
        perm = torch.randperm(n)
        inv = torch.argsort(perm)

        probs_orig = _forward_probs(agent, pod, kv, req)[0].detach().cpu().numpy()
        pod_perm = pod[:, perm, :]
        kv_perm = kv[:, perm, :]
        probs_perm = _forward_probs(agent, pod_perm, kv_perm, req)[0].detach().cpu().numpy()

        # Undo permutation on probs_perm for comparison
        probs_perm_unperm = probs_perm[inv]
        agree += int(np.argmax(probs_orig) == np.argmax(probs_perm_unperm))
        mean_l1 += float(np.mean(np.abs(probs_orig - probs_perm_unperm)))
    return {
        "match_rate": agree / trials,
        "mean_l1_diff": mean_l1 / trials,
    }


def pdp_ice(agent: scb.SimplifiedContextualBandit, pod: torch.Tensor, kv: torch.Tensor, req: torch.Tensor, ranges: Dict[str, np.ndarray], num_points: int = 25, num_contexts: int = 16, target_pod: int = 0) -> Dict[str, Any]:
    # Use a subset of contexts
    ctx_idx = torch.randperm(pod.shape[0])[: min(num_contexts, pod.shape[0])]
    pod_ctx = pod[ctx_idx].clone()
    kv_ctx = kv[ctx_idx].clone()
    req_ctx = req[ctx_idx].clone()

    pod_feat_dim = pod.shape[2]
    kv_dim = kv.shape[2]
    req_dim = req.shape[1]

    pdp_results = {"pod": {}, "kv": {}, "req": {}}

    # Per-pod features PDP/ICE (vary feature for target_pod)
    for f in range(pod_feat_dim):
        lo, hi = float(ranges["pod_min"][f]), float(ranges["pod_max"][f])
        grid = np.linspace(lo, hi, num_points)
        ice_curves = []
        for c in range(pod_ctx.shape[0]):
            probs_curve = []
            for v in grid:
                pod_tmp = pod_ctx[c:c+1].clone()
                pod_tmp[:, target_pod, f] = float(v)
                probs = _forward_probs(agent, pod_tmp, kv_ctx[c:c+1], req_ctx[c:c+1])[0].detach().cpu().numpy()
                probs_curve.append(probs[target_pod])
            ice_curves.append(probs_curve)
        pdp = np.mean(np.array(ice_curves), axis=0)
        pdp_results["pod"][f"feat_{f}"] = {
            "grid": grid.tolist(),
            "pdp": pdp.tolist(),
        }

    # KV PDP/ICE (vary kv for target_pod)
    for f in range(kv_dim):
        lo, hi = float(ranges["kv_min"][f]), float(ranges["kv_max"][f])
        grid = np.linspace(lo, hi, num_points)
        ice_curves = []
        for c in range(kv_ctx.shape[0]):
            probs_curve = []
            for v in grid:
                kv_tmp = kv_ctx[c:c+1].clone()
                kv_tmp[:, target_pod, f] = float(v)
                probs = _forward_probs(agent, pod_ctx[c:c+1], kv_tmp, req_ctx[c:c+1])[0].detach().cpu().numpy()
                probs_curve.append(probs[target_pod])
            ice_curves.append(probs_curve)
        pdp = np.mean(np.array(ice_curves), axis=0)
        pdp_results["kv"][f"feat_{f}"] = {
            "grid": grid.tolist(),
            "pdp": pdp.tolist(),
        }

    # Request features PDP/ICE (vary global request feature)
    for f in range(req_dim):
        lo, hi = float(ranges["req_min"][f]), float(ranges["req_max"][f])
        grid = np.linspace(lo, hi, num_points)
        ice_curves = []
        for c in range(req_ctx.shape[0]):
            probs_curve = []
            for v in grid:
                req_tmp = req_ctx[c:c+1].clone()
                req_tmp[:, f] = float(v)
                probs = _forward_probs(agent, pod_ctx[c:c+1], kv_ctx[c:c+1], req_tmp)[0].detach().cpu().numpy()
                probs_curve.append(probs[target_pod])
            ice_curves.append(probs_curve)
        pdp = np.mean(np.array(ice_curves), axis=0)
        pdp_results["req"][f"feat_{f}"] = {
            "grid": grid.tolist(),
            "pdp": pdp.tolist(),
        }

    return pdp_results


def relative_pdp(agent: scb.SimplifiedContextualBandit, pod: torch.Tensor, kv: torch.Tensor, req: torch.Tensor, ranges: Dict[str, np.ndarray], f_idx: int, num_points: int = 25, num_contexts: int = 16, target_pod: int = 0) -> Dict[str, Any]:
    # Keep sum across pods constant for feature f_idx: raise target by Δ, lower others by Δ/(n-1)
    ctx_idx = torch.randperm(pod.shape[0])[: min(num_contexts, pod.shape[0])]
    pod_ctx = pod[ctx_idx].clone()
    kv_ctx = kv[ctx_idx].clone()
    req_ctx = req[ctx_idx].clone()

    lo, hi = float(ranges["pod_min"][f_idx]), float(ranges["pod_max"][f_idx])
    grid = np.linspace(lo, hi, num_points)
    ice_curves = []
    for c in range(pod_ctx.shape[0]):
        base = pod_ctx[c:c+1].clone()
        base_val = float(base[:, target_pod, f_idx].item())
        probs_curve = []
        for v in grid:
            pod_tmp = base.clone()
            delta = float(v - base_val)
            pod_tmp[:, target_pod, f_idx] = float(v)
            if pod_tmp.shape[1] > 1:
                delta_each = -delta / float(pod_tmp.shape[1] - 1)
                # apply to others with clipping
                for j in range(pod_tmp.shape[1]):
                    if j == target_pod:
                        continue
                    cur = float(pod_tmp[:, j, f_idx].item())
                    newv = cur + delta_each
                    newv = max(lo, min(hi, newv))
                    pod_tmp[:, j, f_idx] = float(newv)
            probs = _forward_probs(agent, pod_tmp, kv_ctx[c:c+1], req_ctx[c:c+1])[0].detach().cpu().numpy()
            probs_curve.append(probs[target_pod])
        ice_curves.append(probs_curve)
    pdp = np.mean(np.array(ice_curves), axis=0)
    return {"grid": grid.tolist(), "pdp": pdp.tolist()}


def ale_one_feature(agent: scb.SimplifiedContextualBandit, pod: torch.Tensor, kv: torch.Tensor, req: torch.Tensor, f_idx: int, bins: int = 10, target_pod: int = 0) -> Dict[str, Any]:
    # Accumulated Local Effects for target pod feature f_idx
    x = pod[:, target_pod, f_idx].detach().cpu().numpy()
    if x.size == 0:
        return {"bin_centers": [], "ale": []}
    qs = np.quantile(x, np.linspace(0.0, 1.0, bins + 1))
    effects = []
    centers = []
    for i in range(bins):
        a, b = float(qs[i]), float(qs[i+1])
        mask = (x >= a) & (x <= b)
        idx = np.where(mask)[0]
        if idx.size == 0:
            effects.append(0.0)
            centers.append((a + b) / 2.0)
            continue
        with torch.no_grad():
            pod_a = pod[idx].clone(); pod_b = pod[idx].clone()
            pod_a[:, target_pod, f_idx] = a
            pod_b[:, target_pod, f_idx] = b
        p_a = _forward_probs(agent, pod_a, kv[idx], req[idx])[:, target_pod].detach().cpu().numpy()
        p_b = _forward_probs(agent, pod_b, kv[idx], req[idx])[:, target_pod].detach().cpu().numpy()
        effects.append(float(np.mean(p_b - p_a)))
        centers.append((a + b) / 2.0)
    ale_vals = np.cumsum(effects)
    ale_vals = ale_vals - np.mean(ale_vals)
    return {"bin_centers": centers, "ale": ale_vals.tolist()}


def rank_effect(agent: scb.SimplifiedContextualBandit, pod: torch.Tensor, kv: torch.Tensor, req: torch.Tensor, f_idx: int, target_pod: int = 0) -> Dict[str, Any]:
    # Relationship between target pod's rank by feature and its probability
    with torch.no_grad():
        probs = _forward_probs(agent, pod, kv, req)[:, target_pod].detach().cpu().numpy()
    vals = pod[:, :, f_idx].detach().cpu().numpy()
    ranks = []
    for i in range(vals.shape[0]):
        order = np.argsort(vals[i])  # ascending: 0 is lowest
        rank_pos = int(np.where(order == target_pod)[0][0])
        ranks.append(rank_pos + 1)  # 1..n
    ranks = np.array(ranks)
    uniq = np.unique(ranks)
    means = []
    for r in uniq:
        means.append(float(np.mean(probs[ranks == r])))
    return {"ranks": uniq.tolist(), "mean_prob": means}

def counterfactual_min_change(agent: scb.SimplifiedContextualBandit, pod: torch.Tensor, kv: torch.Tensor, req: torch.Tensor, ranges: Dict[str, np.ndarray], steps: int = 50, step_size: float = 0.1) -> Dict[str, Any]:
    # One sample analysis (use first sample)
    pod0 = pod[:1].clone().detach().to(_device())
    kv0 = kv[:1].clone().detach().to(_device())
    req0 = req[:1].clone().detach().to(_device())

    with torch.no_grad():
        base_probs = agent.policy(pod0, kv0, req0)[0]
        a = int(torch.argmax(base_probs).item())
    results = {}

    num_pods = base_probs.shape[0]
    for b in range(num_pods):
        if b == a:
            continue

        pod_cf = pod0.clone().detach()
        kv_cf = kv0.clone().detach()
        req_cf = req0.clone().detach()

        success = False
        total_delta_l2 = 0.0
        for _ in range(steps):
            pod_cf.requires_grad_(True)
            kv_cf.requires_grad_(True)
            req_cf.requires_grad_(True)

            # Score margin s_b - s_a
            combined = torch.cat([pod_cf, kv_cf], dim=2)
            exp_req = req_cf.unsqueeze(1).expand(-1, pod_cf.shape[1], -1)
            full = torch.cat([combined, exp_req], dim=2)
            flat = full.view(1 * pod_cf.shape[1], -1)
            scores = agent.policy.pod_scorer(flat).view(1, pod_cf.shape[1])
            margin = scores[0, b] - scores[0, a]
            loss = -margin  # ascend margin

            pod_g, kv_g, req_g = torch.autograd.grad(loss, [pod_cf, kv_cf, req_cf], allow_unused=True)
            if pod_g is None:
                pod_g = torch.zeros_like(pod_cf)
            if kv_g is None:
                kv_g = torch.zeros_like(kv_cf)
            if req_g is None:
                req_g = torch.zeros_like(req_cf)

            with torch.no_grad():
                # Only allow modifying target pod features and global request features
                pod_grad = torch.zeros_like(pod_cf)
                pod_grad[:, b:b+1, :] = pod_g[:, b:b+1, :]
                kv_grad = torch.zeros_like(kv_cf)
                kv_grad[:, b:b+1, :] = kv_g[:, b:b+1, :]
                req_grad = req_g

                # Normalize gradients to step_size
                def _step(x, g, lo, hi):
                    norm = torch.norm(g)
                    if norm > 0:
                        x = x + step_size * g / norm
                    return torch.max(torch.min(x, hi), lo)

                pod_lo = torch.tensor(ranges["pod_min"], device=_device()).view(1, 1, -1).expand_as(pod_cf)
                pod_hi = torch.tensor(ranges["pod_max"], device=_device()).view(1, 1, -1).expand_as(pod_cf)
                kv_lo = torch.tensor(ranges["kv_min"], device=_device()).view(1, 1, -1).expand_as(kv_cf)
                kv_hi = torch.tensor(ranges["kv_max"], device=_device()).view(1, 1, -1).expand_as(kv_cf)
                req_lo = torch.tensor(ranges["req_min"], device=_device()).view(1, -1)
                req_hi = torch.tensor(ranges["req_max"], device=_device()).view(1, -1)

                pod_prev = pod_cf.clone()
                kv_prev = kv_cf.clone()
                req_prev = req_cf.clone()

                pod_cf = _step(pod_cf, pod_grad, pod_lo, pod_hi).detach()
                kv_cf = _step(kv_cf, kv_grad, kv_lo, kv_hi).detach()
                req_cf = _step(req_cf, req_grad, req_lo, req_hi).detach()

                total_delta_l2 += float(torch.norm(pod_cf - pod_prev) + torch.norm(kv_cf - kv_prev) + torch.norm(req_cf - req_prev))

                probs = agent.policy(pod_cf, kv_cf, req_cf)[0]
                if int(torch.argmax(probs).item()) == b:
                    success = True
                    break

        results[f"to_pod_{b}"] = {
            "success": success,
            "total_delta_l2": total_delta_l2,
        }
    return results


def interaction_heatmap(agent: scb.SimplifiedContextualBandit, pod: torch.Tensor, kv: torch.Tensor, req: torch.Tensor, ranges: Dict[str, np.ndarray], target_pod: int = 0, f1: int = 0, f2: int = 1, num_points: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Use first context
    pod0 = pod[:1].clone()
    kv0 = kv[:1].clone()
    req0 = req[:1].clone()
    x_vals = np.linspace(float(ranges["pod_min"][f1]), float(ranges["pod_max"][f1]), num_points)
    y_vals = np.linspace(float(ranges["pod_min"][f2]), float(ranges["pod_max"][f2]), num_points)
    Z = np.zeros((num_points, num_points), dtype=np.float32)
    for i, xv in enumerate(x_vals):
        for j, yv in enumerate(y_vals):
            pod_tmp = pod0.clone()
            pod_tmp[:, target_pod, f1] = float(xv)
            pod_tmp[:, target_pod, f2] = float(yv)
            Z[j, i] = _forward_probs(agent, pod_tmp, kv0, req0)[0, target_pod].item()
    return x_vals, y_vals, Z


def save_pdp_plots(pdp_res: Dict[str, Any], out_dir: str, prefix: str, target_pod: int):
    # Deprecated in favor of direct-to-PDF rendering
    pass


def save_heatmap(x_vals: np.ndarray, y_vals: np.ndarray, Z: np.ndarray, out_dir: str, title: str, fname: str):
    # Deprecated in favor of direct-to-PDF rendering
    pass


def collate_pngs_to_pdf(out_dir: str, pdf_filename: str = "xai_plots.pdf", n_cols: int = 4, n_rows: int = 2, single_page: bool = True) -> Optional[str]:
    # Deprecated; no longer collating PNGs. All charts are rendered directly to PDF.
    return None


def _add_title_page(pdf: PdfPages, title: str, subtitle: str = ""):
    fig = plt.figure(figsize=(11, 8.5))
    plt.axis('off')
    plt.text(0.5, 0.6, title, ha='center', va='center', fontsize=24, weight='bold')
    if subtitle:
        plt.text(0.5, 0.5, subtitle, ha='center', va='center', fontsize=12)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def _add_saliency_page(pdf: PdfPages, grad_res: Dict[str, Any], names: Dict[str, list], topk: int = 20):
    pod_top = grad_res.get("pod_feat_importance_topk", [])
    req_top = grad_res.get("req_feat_importance_topk", [])
    kv_top = grad_res.get("kv_feat_importance", [])

    combined = []
    # Pod features
    for i, v in pod_top:
        i = int(i)
        label = names["pod"][i] if i < len(names["pod"]) else f"pod_feat_{i}"
        combined.append((f"pod:{label}", float(v), "pod"))
    # Request features
    for i, v in req_top:
        i = int(i)
        label = names["req"][i] if i < len(names["req"]) else f"req_feat_{i}"
        combined.append((f"req:{label}", float(v), "req"))
    # KV features
    for i, v in kv_top:
        i = int(i)
        label = names["kv"][i] if i < len(names["kv"]) else f"kv_feat_{i}"
        combined.append((f"kv:{label}", float(v), "kv"))

    # Sort and take topk
    combined.sort(key=lambda x: x[1], reverse=True)
    combined = combined[:topk]

    if not combined:
        fig = plt.figure(figsize=(11, 4))
        plt.axis('off')
        plt.text(0.5, 0.5, "No saliency data", ha='center', va='center')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        return

    # Determine sign using saved mean gradients
    pod_grad_mean = np.array(grad_res.get("pod_grad_mean", []))
    kv_grad_mean = np.array(grad_res.get("kv_grad_mean", []))
    req_grad_mean = np.array(grad_res.get("req_grad_mean", []))

    labels = []
    values = []
    colors = []
    for name, mag, typ in combined:
        labels.append(name)
        values.append(mag)
        if typ == 'pod':
            # extract index
            base = name.split(':',1)[1]
            try:
                idx = names['pod'].index(base)
            except ValueError:
                idx = None
            sign = np.sign(pod_grad_mean[idx]) if idx is not None and idx < len(pod_grad_mean) else 0.0
        elif typ == 'req':
            base = name.split(':',1)[1]
            try:
                idx = names['req'].index(base)
            except ValueError:
                idx = None
            sign = np.sign(req_grad_mean[idx]) if idx is not None and idx < len(req_grad_mean) else 0.0
        else:
            base = name.split(':',1)[1]
            try:
                idx = names['kv'].index(base)
            except ValueError:
                idx = None
            sign = np.sign(kv_grad_mean[idx]) if idx is not None and idx < len(kv_grad_mean) else 0.0
        colors.append('tab:blue' if sign >= 0 else 'tab:red')

    fig = plt.figure(figsize=(11, 5.5))
    ax = plt.gca()
    ax.bar(range(len(values)), values, color=colors)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_title("Gradient Saliency (combined: pod, kv, request)")
    ax.set_ylabel("|dP/dx|")
    ax.grid(True, alpha=0.3)
    # Legend
    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor='tab:blue', label='positive (+)'), Patch(facecolor='tab:red', label='negative (−)')]
    ax.legend(handles=legend_elems, loc='upper right', fontsize=8)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def _add_ig_page(pdf: PdfPages, ig_res: Dict[str, Any], names: Dict[str, list], topk: int = 20):
    pod = ig_res.get("pod_feat_ig", [])
    kv = ig_res.get("kv_feat_ig", [])
    req = ig_res.get("req_feat_ig", [])

    combined = []
    for i, v in enumerate(pod):
        label = names["pod"][i] if i < len(names["pod"]) else f"pod_feat_{i}"
        combined.append((f"pod:{label}", float(v), "pod"))
    for i, v in enumerate(kv):
        label = names["kv"][i] if i < len(names["kv"]) else f"kv_feat_{i}"
        combined.append((f"kv:{label}", float(v), "kv"))
    for i, v in enumerate(req):
        label = names["req"][i] if i < len(names["req"]) else f"req_feat_{i}"
        combined.append((f"req:{label}", float(v), "req"))

    # Sort desc and take topk
    combined.sort(key=lambda x: x[1], reverse=True)
    combined = combined[:topk]

    if not combined:
        fig = plt.figure(figsize=(11, 4))
        plt.axis('off')
        plt.text(0.5, 0.5, "No IG data", ha='center', va='center')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        return

    # For IG, sign is the attribution sign
    # For IG, use signed IG values to color, magnitudes for bar heights
    pod_ig_signed = np.array(ig_res.get("pod_feat_ig_signed", []))
    kv_ig_signed = np.array(ig_res.get("kv_feat_ig_signed", []))
    req_ig_signed = np.array(ig_res.get("req_feat_ig_signed", []))

    labels = []
    values = []
    colors = []
    for name, mag, typ in combined:
        labels.append(name)
        values.append(mag)
        if typ == 'pod':
            base = name.split(':',1)[1]
            try:
                idx = names['pod'].index(base)
            except ValueError:
                idx = None
            sign = np.sign(pod_ig_signed[idx]) if idx is not None and idx < len(pod_ig_signed) else 0.0
        elif typ == 'req':
            base = name.split(':',1)[1]
            try:
                idx = names['req'].index(base)
            except ValueError:
                idx = None
            sign = np.sign(req_ig_signed[idx]) if idx is not None and idx < len(req_ig_signed) else 0.0
        else:
            base = name.split(':',1)[1]
            try:
                idx = names['kv'].index(base)
            except ValueError:
                idx = None
            sign = np.sign(kv_ig_signed[idx]) if idx is not None and idx < len(kv_ig_signed) else 0.0
        colors.append('tab:blue' if sign >= 0 else 'tab:red')

    fig = plt.figure(figsize=(11, 5.5))
    ax = plt.gca()
    ax.bar(range(len(values)), values, color=colors)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_title("Integrated Gradients (combined: pod, kv, request)")
    ax.set_ylabel("attribution")
    ax.grid(True, alpha=0.3)
    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor='tab:blue', label='positive (+)'), Patch(facecolor='tab:red', label='negative (−)')]
    ax.legend(handles=legend_elems, loc='upper right', fontsize=8)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def _add_symmetry_page(pdf: PdfPages, sym_res: Dict[str, Any]):
    fig = plt.figure(figsize=(11, 8.5))
    plt.axis('off')
    title = "Permutation Symmetry Test"
    text = (
        f"match_rate: {sym_res.get('match_rate', 'NA')}\n"
        f"mean_l1_diff: {sym_res.get('mean_l1_diff', 'NA')}\n\n"
        "Higher match_rate and lower mean_l1_diff indicate invariance to pod ordering."
    )
    plt.text(0.5, 0.7, title, ha='center', va='center', fontsize=16, weight='bold')
    plt.text(0.1, 0.5, text, ha='left', va='center', fontsize=12)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def _add_counterfactuals_page(pdf: PdfPages, cf_res: Dict[str, Any]):
    # Parse to_pod_* entries
    entries = []
    for k, v in cf_res.items():
        if k.startswith("to_pod_"):
            try:
                pod_id = int(k.split("_")[-1])
            except Exception:
                continue
            entries.append((pod_id, v.get("total_delta_l2", float('inf')), bool(v.get("success", False))))
    entries.sort(key=lambda x: x[0])

    fig, ax = plt.subplots(1, 1, figsize=(11, 6))
    if entries:
        xs = [f"pod_{e[0]}" for e in entries]
        ys = [e[1] for e in entries]
        colors = ['tab:green' if e[2] else 'tab:red' for e in entries]
        ax.bar(range(len(ys)), ys, color=colors)
        ax.set_xticks(range(len(ys)))
        ax.set_xticklabels(xs, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel("Total ΔL2 to flip")
        ax.set_title("Counterfactual Distance to Flip to Each Pod (green=success, red=failed within budget)")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No counterfactual data", ha='center', va='center')
        ax.axis('off')
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def _add_pdp_pages(pdf: PdfPages, pdp_res: Dict[str, Any], target_pod: int, names: Dict[str, list], n_cols: int = 4, n_rows: int = 2):
    # Combine all groups into one sequence and render in unified grids
    combined = []
    for group in ["pod", "kv", "req"]:
        for feat_name, obj in pdp_res.get(group, {}).items():
            try:
                feat_idx = int(str(feat_name).split('_')[-1])
            except Exception:
                feat_idx = None
            pretty_name = feat_name
            if feat_idx is not None:
                if group == 'pod' and feat_idx < len(names['pod']):
                    pretty_name = names['pod'][feat_idx]
                elif group == 'kv' and feat_idx < len(names['kv']):
                    pretty_name = names['kv'][feat_idx]
                elif group == 'req' and feat_idx < len(names['req']):
                    pretty_name = names['req'][feat_idx]
            combined.append((group, pretty_name, obj))

    if not combined:
        return

    per_page = n_cols * n_rows
    for i in range(0, len(combined), per_page):
        chunk = combined[i:i + per_page]
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 8.5))
        axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        for ax, (group, pretty_name, obj) in zip(axes_flat, chunk):
            grid = obj["grid"]
            pdp = obj["pdp"]
            ax.plot(grid, pdp)
            ax.set_title(f"{group}:{pretty_name}", fontsize=8)
            ax.set_xlabel("value", fontsize=8)
            ax.set_ylabel(f"P(pod{target_pod})", fontsize=8)
            ax.grid(True, alpha=0.3)
        for ax in axes_flat[len(chunk):]:
            ax.axis('off')
        fig.suptitle(f"PDPs (all groups), target pod={target_pod}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)


def _add_interaction_page(pdf: PdfPages, x_vals: np.ndarray, y_vals: np.ndarray, Z: np.ndarray, title: str):
    fig = plt.figure(figsize=(11, 8.5))
    ax = plt.gca()
    im = ax.imshow(Z, origin='lower', aspect='auto', extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]], cmap='viridis')
    plt.colorbar(im, ax=ax, label='P(target pod)')
    ax.set_xlabel('feature 1')
    ax.set_ylabel('feature 2')
    ax.set_title(title)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def _add_relative_pdp_page(pdf: PdfPages, rel_obj: Dict[str, Any], title: str):
    if not rel_obj:
        return
    fig = plt.figure(figsize=(11, 5.5))
    ax = plt.gca()
    ax.plot(rel_obj.get("grid", []), rel_obj.get("pdp", []))
    ax.set_title(title)
    ax.set_xlabel("value (sum-conserving across pods)")
    ax.set_ylabel("P(target pod)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def _add_ale_page(pdf: PdfPages, ale_obj: Dict[str, Any], title: str):
    if not ale_obj:
        return
    fig = plt.figure(figsize=(11, 5.5))
    ax = plt.gca()
    ax.plot(ale_obj.get("bin_centers", []), ale_obj.get("ale", []))
    ax.set_title(title)
    ax.set_xlabel("feature value (quantile bins)")
    ax.set_ylabel("Accumulated local effect")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def _add_rank_effect_page(pdf: PdfPages, rank_obj: Dict[str, Any], title: str):
    if not rank_obj:
        return
    fig = plt.figure(figsize=(11, 5.5))
    ax = plt.gca()
    ranks = rank_obj.get("ranks", [])
    means = rank_obj.get("mean_prob", [])
    ax.plot(ranks, means, marker='o')
    ax.set_title(title)
    ax.set_xlabel("target pod rank by feature (1=lowest)")
    ax.set_ylabel("mean P(target pod)")
    ax.set_xticks(ranks)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def _add_findings_pages(pdf: PdfPages, grad_res: Dict[str, Any], ig_res: Dict[str, Any], sym_res: Dict[str, Any], cf_res: Dict[str, Any], pdp_res: Dict[str, Any], names: Dict[str, list]):
    # Page 1: Saliency + IG summary
    def _fmt_top(items, label_list):
        lines = []
        for idx, val in items:
            idx = int(idx)
            label = label_list[idx] if idx < len(label_list) else f"feat_{idx}"
            lines.append(f"- {label}: {val:.3f}")
        return "\n".join(lines)

    pod_sal = grad_res.get("pod_feat_importance_topk", [])[:8]
    req_sal = grad_res.get("req_feat_importance_topk", [])[:3]
    kv_sal = grad_res.get("kv_feat_importance", [])

    pod_ig = ig_res.get("pod_feat_ig", [])
    req_ig = ig_res.get("req_feat_ig", [])
    kv_ig = ig_res.get("kv_feat_ig", [])

    fig = plt.figure(figsize=(11, 8.5))
    plt.axis('off')
    text = [
        "EXPLAINABILITY FINDINGS",
        "",
        "Saliency (local sensitivity):",
        _fmt_top(pod_sal, names["pod"]) or "- (no data)",
        "",
        "Request Saliency:",
        _fmt_top(req_sal, names["req"]) or "- (no data)",
        "",
        "KV Saliency:",
        _fmt_top(kv_sal, names["kv"]) or "- (no data)",
        "",
        "Integrated Gradients (robust attribution):",
        f"- Top pod feature: {names['pod'][int(np.argmax(pod_ig))] if len(pod_ig)>0 else 'NA'}",
        f"- KV importance: {kv_ig[0]:.3f}" if len(kv_ig)>0 else "- KV importance: NA",
        f"- Request max attribution: {names['req'][int(np.argmax(req_ig))] if len(req_ig)>0 else 'NA'}",
    ]
    plt.text(0.05, 0.95, "\n".join(text), va='top', ha='left', fontsize=11)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # Page 2: PDP trends summary
    def _trend(p):
        if not p:
            return "flat"
        if abs(p[-1]-p[0]) < 1e-3:
            return "flat"
        return "increasing" if (p[-1] > p[0]) else "decreasing"

    pod_lines = []
    for k, obj in pdp_res.get('pod', {}).items():
        try:
            idx = int(str(k).split('_')[-1])
            name = names['pod'][idx] if idx < len(names['pod']) else k
        except Exception:
            name = k
        pod_lines.append(f"- {name}: {_trend(obj.get('pdp', []))}")

    kv_lines = []
    for k, obj in pdp_res.get('kv', {}).items():
        try:
            idx = int(str(k).split('_')[-1])
            name = names['kv'][idx] if idx < len(names['kv']) else k
        except Exception:
            name = k
        kv_lines.append(f"- {name}: {_trend(obj.get('pdp', []))}")

    req_lines = []
    for k, obj in pdp_res.get('req', {}).items():
        try:
            idx = int(str(k).split('_')[-1])
            name = names['req'][idx] if idx < len(names['req']) else k
        except Exception:
            name = k
        req_lines.append(f"- {name}: {_trend(obj.get('pdp', []))}")

    fig = plt.figure(figsize=(11, 8.5))
    plt.axis('off')
    lines = ["PDP Trend Summary (target pod)", "", "Pod features:"] + pod_lines + ["", "KV features:"] + kv_lines + ["", "Request features:"] + req_lines
    plt.text(0.05, 0.95, "\n".join(lines), va='top', ha='left', fontsize=11)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # Page 3: Symmetry + Counterfactuals
    entries = []
    for k, v in cf_res.items():
        if k.startswith("to_pod_"):
            try:
                pod_id = int(k.split("_")[-1])
            except Exception:
                continue
            entries.append((pod_id, v.get("total_delta_l2", float('inf')), bool(v.get("success", False))))
    entries.sort(key=lambda x: x[1])
    success_rate = np.mean([e[2] for e in entries]) if entries else 0.0
    min_delta = entries[0][1] if entries else float('nan')
    mean_delta = float(np.mean([e[1] for e in entries])) if entries else float('nan')

    fig = plt.figure(figsize=(11, 8.5))
    plt.axis('off')
    text = [
        "Permutation Symmetry:",
        f"- match_rate: {sym_res.get('match_rate','NA')}",
        f"- mean_l1_diff: {sym_res.get('mean_l1_diff','NA')}",
        "",
        "Counterfactual Robustness:",
        f"- success_rate (flip achieved): {success_rate:.2f}",
        f"- min ΔL2 to flip: {min_delta:.2f}",
        f"- mean ΔL2 to flip: {mean_delta:.2f}",
        "",
        "Interpretation:",
        "- High symmetry indicates no position bias.",
        "- Larger ΔL2 implies locally robust decisions (harder to flip).",
    ]
    plt.text(0.05, 0.95, "\n".join(text), va='top', ha='left', fontsize=11)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Explainability for RL router policy")
    parser.add_argument("--final_model_dir", required=True, help="Path to trained model directory containing policy.pth")
    # parser.add_argument("--search_data_roots", default="", help="Colon-separated roots to search for tensor_dataset.pt")
    # parser.add_argument("--reference_tensor", default="", help="Explicit path to tensor_dataset.pt (overrides search)")
    parser.add_argument("--num_pods", type=int, default=0, help="Override num pods if no reference found")
    parser.add_argument("--pod_feat_dim", type=int, default=0, help="Override pod feature dim if no reference found")
    parser.add_argument("--kv_dim", type=int, default=0, help="Override kv feature dim if no reference found")
    parser.add_argument("--req_feat_dim", type=int, default=0, help="Override request feature dim if no reference found")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=32, help="Number of contexts for analyses")
    args = parser.parse_args()

    set_all_seeds(args.seed)

    final_model_dir = os.path.abspath(args.final_model_dir)
    # assert os.path.exists(os.path.join(final_model_dir, "policy.pth")), f"policy.pth not found under {final_model_dir}"
    reference_tensor = f"{final_model_dir}/encoded_data/batch_1/tensor_dataset.pt"
    if reference_tensor:
        logger.info(f"Using reference tensor: {reference_tensor}")
        num_pods, pod_feat_dim, kv_dim, req_feat_dim = _infer_dims_from_tensor(reference_tensor)
    else:
        logger.warning("No reference tensor found; using CLI overrides")
        assert args.num_pods > 0 and args.pod_feat_dim > 0 and args.kv_dim > 0 and args.req_feat_dim > 0, \
            "When no reference dataset is found, you must supply --num_pods --pod_feat_dim --kv_dim --req_feat_dim"
        num_pods, pod_feat_dim, kv_dim, req_feat_dim = args.num_pods, args.pod_feat_dim, args.kv_dim, args.req_feat_dim

    model_cfg = _load_model_config(final_model_dir)
    agent = _build_agent(model_cfg, num_pods, pod_feat_dim, kv_dim, req_feat_dim, final_model_dir)

    # Prepare contexts and ranges
    pod_ctx, kv_ctx, req_ctx = _select_base_samples(reference_tensor, args.samples, num_pods, pod_feat_dim, kv_dim, req_feat_dim)
    if reference_tensor:
        ranges = _feature_ranges_from_reference(reference_tensor)
    else:
        # Generic unit ranges
        ranges = {
            "pod_min": np.zeros(pod_feat_dim),
            "pod_max": np.ones(pod_feat_dim),
            "kv_min": np.zeros(kv_dim),
            "kv_max": np.ones(kv_dim),
            "req_min": np.zeros(req_feat_dim),
            "req_max": np.ones(req_feat_dim),
        }

    # Load feature names from metadata if available
    feature_names = _load_feature_names(reference_tensor, pod_feat_dim, kv_dim, req_feat_dim)

    out_dir = os.path.join(final_model_dir, "xai_report")
    os.makedirs(out_dir, exist_ok=True)

    # 1) Gradient saliency
    grad_res = gradient_saliency(agent, pod_ctx, kv_ctx, req_ctx, topk=10)
    with open(os.path.join(out_dir, "saliency_topk.json"), "w") as f:
        json.dump(grad_res, f, indent=2)

    # 2) Integrated Gradients
    ig_res = integrated_gradients(agent, pod_ctx, kv_ctx, req_ctx, steps=32)
    with open(os.path.join(out_dir, "integrated_gradients.json"), "w") as f:
        json.dump(ig_res, f, indent=2)

    # 3) Symmetry test
    sym_res = permutation_symmetry_test(agent, pod_ctx[:1], kv_ctx[:1], req_ctx[:1], trials=30)
    with open(os.path.join(out_dir, "symmetry_test.json"), "w") as f:
        json.dump(sym_res, f, indent=2)

    # 4) PDP/ICE for a representative target pod
    target_pod = 0
    pdp_res = pdp_ice(agent, pod_ctx, kv_ctx, req_ctx, ranges, num_points=25, num_contexts=min(16, pod_ctx.shape[0]), target_pod=target_pod)
    with open(os.path.join(out_dir, "pdp_ice.json"), "w") as f:
        json.dump(pdp_res, f, indent=2)

    # 5) Counterfactual minimal changes
    cf_res = counterfactual_min_change(agent, pod_ctx, kv_ctx, req_ctx, ranges, steps=60, step_size=0.15)
    with open(os.path.join(out_dir, "counterfactuals.json"), "w") as f:
        json.dump(cf_res, f, indent=2)

    # 6) Interaction heatmap between top-2 important pod features
    top_pod_feats = [idx for idx, _ in grad_res.get("pod_feat_importance_topk", [])]
    interaction_payload = None
    if len(top_pod_feats) >= 2:
        f1, f2 = top_pod_feats[0], top_pod_feats[1]
        x_vals, y_vals, Z = interaction_heatmap(agent, pod_ctx, kv_ctx, req_ctx, ranges, target_pod=target_pod, f1=f1, f2=f2, num_points=40)
        interaction_payload = (x_vals, y_vals, Z, f1, f2)

    # 7) Faithful analyses focused on prefill_tokens (only if present)
    prefill_name = "prefill_tokens"
    prefill_idx = None
    if prefill_name in feature_names.get('pod', []):
        prefill_idx = feature_names['pod'].index(prefill_name)
    rel_pdp_obj, ale_obj, rank_obj = {}, {}, {}
    if prefill_idx is not None:
        rel_pdp_obj = relative_pdp(agent, pod_ctx, kv_ctx, req_ctx, ranges, f_idx=int(prefill_idx), num_points=25, num_contexts=min(16, pod_ctx.shape[0]), target_pod=target_pod)
        ale_obj = ale_one_feature(agent, pod_ctx, kv_ctx, req_ctx, f_idx=int(prefill_idx), bins=10, target_pod=target_pod)
        rank_obj = rank_effect(agent, pod_ctx, kv_ctx, req_ctx, f_idx=int(prefill_idx), target_pod=target_pod)

    # Direct-to-PDF report (vector quality)
    report_pdf = os.path.join(out_dir, "xai_report.pdf")
    with PdfPages(report_pdf) as pdf:
        _add_title_page(pdf, "Routing Policy Explainability Report", os.path.basename(final_model_dir))
        _add_saliency_page(pdf, grad_res, feature_names, topk=20)
        _add_ig_page(pdf, ig_res, feature_names, topk=20)
        _add_symmetry_page(pdf, sym_res)
        _add_pdp_pages(pdf, pdp_res, target_pod, feature_names, n_cols=4, n_rows=2)
        if interaction_payload is not None:
            x_vals, y_vals, Z, f1, f2 = interaction_payload
            f1_name = feature_names['pod'][f1] if f1 < len(feature_names['pod']) else f"feat {f1}"
            f2_name = feature_names['pod'][f2] if f2 < len(feature_names['pod']) else f"feat {f2}"
            _add_interaction_page(pdf, x_vals, y_vals, Z, title=f"Interaction heatmap (pod{target_pod} {f1_name} vs {f2_name})")
        # New faithful explainability pages (only if prefill exists)
        if prefill_idx is not None:
            _add_relative_pdp_page(pdf, rel_pdp_obj, title=f"Relative PDP (sum-conserving) for pod{target_pod}:{prefill_name}")
            _add_ale_page(pdf, ale_obj, title=f"ALE for pod{target_pod}:{prefill_name}")
            _add_rank_effect_page(pdf, rank_obj, title=f"Rank-based effect for pod{target_pod}:{prefill_name}")
        _add_counterfactuals_page(pdf, cf_res)
        # New: narrative findings pages
        _add_findings_pages(pdf, grad_res, ig_res, sym_res, cf_res, pdp_res, feature_names)

    logger.info(f"Saved explainability report PDF: {report_pdf}")
    logger.info(f"Saved explainability artifacts to {out_dir}")


if __name__ == "__main__":
    main()



