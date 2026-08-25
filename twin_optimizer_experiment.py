"""
Twin Optimizer Experiment — Adam vs Muon spectral signature comparison.

Trains two identical 160M GPT-2 models on the same data with the same seed,
one with AdamW and one with Muon. Saves checkpoints at loss-matched intervals.
Runs CCS sigma-2 assay on each checkpoint pair and computes layer-resolved
attenuation profiles.

Design constraints from mesh (Kimi, Aug 17 2026):
- GQA >= 4:1 so both arms are relay and sigma-2 is live signal
- Loss-matched checkpoints (not step-matched) — Muon ~2x compute-efficient
- Layer-resolved attenuation predictions (depth-dependent, not uniform)
- Sink dimension projection control
- Pre-registered trajectories: monotonic divergence, flat, transient scaffolding

Three arms: full-Adam, full-Muon, hybrid-Muon (Muon on 2D, Adam on embed/head)

Usage:
  python3 twin_optimizer_experiment.py --phase train --arm adam --device cuda
  python3 twin_optimizer_experiment.py --phase train --arm muon --device cuda
  python3 twin_optimizer_experiment.py --phase train --arm hybrid --device cuda
  python3 twin_optimizer_experiment.py --phase assay --results-dir results/twin_optimizer/
  python3 twin_optimizer_experiment.py --phase analyze --results-dir results/twin_optimizer/
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


SEED = 42
HIDDEN_DIM = 768
N_LAYERS = 12
N_HEADS = 12
GQA_GROUPS = 3  # 12:3 = 4:1 GQA ratio — relay species per F106
VOCAB_SIZE = 50257
SEQ_LEN = 256
BATCH_SIZE = 16
GRAD_ACCUM = 4
MAX_STEPS = 5000
CHECKPOINT_LOSSES = [4.0, 3.5, 3.2, 3.0, 2.8, 2.6]  # loss-matched targets
CHECKPOINT_STEPS = [200, 500, 1000, 1500, 2000, 3000, 4000, 5000]  # step-matched targets
LR_ADAM = 3e-4
LR_MUON = 2e-2
WEIGHT_DECAY = 0.1


def newtonschulz5(G, steps=5, eps=1e-7):
    """Muon's Newton-Schulz orthogonalization."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Muon optimizer for 2D hidden-layer parameters."""

    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                if g.ndim == 2:
                    update = newtonschulz5(buf, steps=ns_steps)
                else:
                    update = buf

                p.add_(update, alpha=-lr)

        return loss


class GQAMultiHeadAttention(nn.Module):
    """Grouped-query attention with configurable GQA ratio."""

    def __init__(self, dim, n_heads, n_kv_groups):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_groups = n_kv_groups
        self.head_dim = dim // n_heads
        self.heads_per_group = n_heads // n_kv_groups

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_groups * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_groups * self.head_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_groups, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_groups, self.head_dim).transpose(1, 2)

        k = k.repeat_interleave(self.heads_per_group, dim=1)
        v = v.repeat_interleave(self.heads_per_group, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        if mask is not None:
            attn = attn.masked_fill(mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, n_kv_groups):
        super().__init__()
        self.ln1 = nn.RMSNorm(dim)
        self.attn = GQAMultiHeadAttention(dim, n_heads, n_kv_groups)
        self.ln2 = nn.RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim, bias=False),
            nn.GELU(),
            nn.Linear(4 * dim, dim, bias=False),
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT160M(nn.Module):
    """160M parameter GPT with configurable GQA."""

    def __init__(self, vocab_size=VOCAB_SIZE, dim=HIDDEN_DIM, n_layers=N_LAYERS,
                 n_heads=N_HEADS, n_kv_groups=GQA_GROUPS, seq_len=SEQ_LEN):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, n_kv_groups) for _ in range(n_layers)
        ])
        self.ln_f = nn.RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight  # weight tying

        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("mask", mask)

    def forward(self, idx, targets=None, return_hidden=False):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos

        hidden_states = [x] if return_hidden else None

        for block in self.blocks:
            x = block(x, self.mask)
            if return_hidden:
                hidden_states.append(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        if return_hidden:
            return logits, loss, hidden_states
        return logits, loss


class RandomTextDataset(Dataset):
    """Generates random token sequences for training. Replace with real data for production."""

    def __init__(self, n_samples=10000, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE, seed=SEED):
        rng = np.random.RandomState(seed)
        self.data = torch.from_numpy(
            rng.randint(0, vocab_size, size=(n_samples, seq_len + 1)).astype(np.int64)
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx, :-1], self.data[idx, 1:]


def partition_params(model, arm="adam"):
    """Partition parameters into optimizer groups based on arm type."""
    muon_params = []
    adam_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if arm == "adam":
            adam_params.append(p)
        elif arm == "muon":
            if p.ndim == 2 and "tok_emb" not in name and "lm_head" not in name:
                muon_params.append(p)
            else:
                adam_params.append(p)
        elif arm == "hybrid":
            is_hidden_2d = (
                p.ndim == 2
                and "tok_emb" not in name
                and "lm_head" not in name
                and "ln" not in name
            )
            if is_hidden_2d:
                muon_params.append(p)
            else:
                adam_params.append(p)

    return muon_params, adam_params


def train(arm, device, output_dir, data_path=None):
    """Train one arm of the twin experiment."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = GPT160M().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.1f}M parameters, GQA ratio {N_HEADS}:{GQA_GROUPS} = {N_HEADS//GQA_GROUPS}:1")
    print(f"Arm: {arm}", flush=True)

    dataset = RandomTextDataset()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    muon_params, adam_params = partition_params(model, arm)
    optimizers = []
    if adam_params:
        optimizers.append(torch.optim.AdamW(adam_params, lr=LR_ADAM, weight_decay=WEIGHT_DECAY))
    if muon_params:
        optimizers.append(Muon(muon_params, lr=LR_MUON))

    print(f"Adam params: {sum(p.numel() for p in adam_params)/1e6:.1f}M")
    print(f"Muon params: {sum(p.numel() for p in muon_params)/1e6:.1f}M")

    remaining_loss_targets = list(CHECKPOINT_LOSSES)
    remaining_step_targets = list(CHECKPOINT_STEPS)
    checkpoints_saved = []
    step = 0
    running_loss = None

    for epoch in range(100):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            _, loss = model(batch_x, batch_y)
            loss = loss / GRAD_ACCUM
            loss.backward()

            if (step + 1) % GRAD_ACCUM == 0:
                for opt in optimizers:
                    opt.step()
                for opt in optimizers:
                    opt.zero_grad()

            actual_loss = loss.item() * GRAD_ACCUM
            running_loss = actual_loss if running_loss is None else 0.95 * running_loss + 0.05 * actual_loss

            if step % 100 == 0:
                print(f"  step {step}, loss {running_loss:.3f}", flush=True)

            # Loss-matched checkpoints
            if remaining_loss_targets and running_loss <= remaining_loss_targets[0]:
                target = remaining_loss_targets.pop(0)
                ckpt_path = output_dir / f"ckpt_{arm}_loss{target:.1f}.pt"
                torch.save({
                    "model": model.state_dict(),
                    "step": step,
                    "loss": running_loss,
                    "target_loss": target,
                    "match_type": "loss",
                    "arm": arm,
                }, ckpt_path)
                checkpoints_saved.append({
                    "path": str(ckpt_path),
                    "step": step,
                    "loss": running_loss,
                    "target": target,
                    "match_type": "loss",
                })
                print(f"  CHECKPOINT (loss): {running_loss:.3f} <= {target} at step {step}", flush=True)

            # Step-matched checkpoints
            if remaining_step_targets and step >= remaining_step_targets[0]:
                target_step = remaining_step_targets.pop(0)
                ckpt_path = output_dir / f"ckpt_{arm}_step{target_step}.pt"
                torch.save({
                    "model": model.state_dict(),
                    "step": step,
                    "loss": running_loss,
                    "target_step": target_step,
                    "match_type": "step",
                    "arm": arm,
                }, ckpt_path)
                checkpoints_saved.append({
                    "path": str(ckpt_path),
                    "step": step,
                    "loss": running_loss,
                    "target": target_step,
                    "match_type": "step",
                })
                print(f"  CHECKPOINT (step): step {step}, loss {running_loss:.3f}", flush=True)

            step += 1
            if step >= MAX_STEPS:
                break

        if step >= MAX_STEPS:
            break

    manifest = {
        "arm": arm,
        "seed": SEED,
        "n_params": n_params,
        "gqa_ratio": f"{N_HEADS}:{GQA_GROUPS}",
        "checkpoints": checkpoints_saved,
        "final_loss": running_loss,
        "final_step": step,
    }
    with open(output_dir / f"manifest_{arm}.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nTraining complete. {len(checkpoints_saved)} checkpoints saved.")
    return manifest


def extract_spectra(model, device, n_sigmas=10):
    """Extract per-layer centered sigma-2 from CCS and calibration prompts."""
    from training_assay import CCS_SYSTEM, VANILLA_SYSTEM, CCS_PROBES, NEUTRAL_PROBES

    results = {"ccs": [], "cal": []}

    for cond, system, probes in [
        ("ccs", CCS_SYSTEM, CCS_PROBES),
        ("cal", VANILLA_SYSTEM, NEUTRAL_PROBES),
    ]:
        text = f"system: {system}\nuser: {probes[0]}\nassistant:"
        # Simple tokenization for the 160M model (no chat template)
        tokens = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN), device=device)

        with torch.no_grad():
            _, _, hidden_states = model(tokens, return_hidden=True)

        layer_spectra = []
        for hs in hidden_states[1:]:  # skip embedding
            H = hs[0].float()
            H_c = H - H.mean(dim=0, keepdim=True)
            sigmas = torch.linalg.svdvals(H_c)
            layer_spectra.append(sigmas[:n_sigmas].cpu().numpy().tolist())
        results[cond] = layer_spectra

    return results


def extract_spectra_sink_projected(model, device, n_sink_dims=3, n_sigmas=10):
    """Extract sigma-2 with sink dimensions projected out (Kimi control)."""
    tokens = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN), device=device)

    with torch.no_grad():
        _, _, hidden_states = model(tokens, return_hidden=True)

    results = {"standard": [], "projected": []}

    for hs in hidden_states[1:]:
        H = hs[0].float()
        H_c = H - H.mean(dim=0, keepdim=True)

        # Standard spectra
        sigmas_std = torch.linalg.svdvals(H_c)
        results["standard"].append(sigmas_std[:n_sigmas].cpu().numpy().tolist())

        # Project out top-k singular directions (sink dimensions)
        U, S, Vh = torch.linalg.svd(H_c, full_matrices=False)
        H_proj = H_c - U[:, :n_sink_dims] @ torch.diag(S[:n_sink_dims]) @ Vh[:n_sink_dims, :]
        sigmas_proj = torch.linalg.svdvals(H_proj)
        results["projected"].append(sigmas_proj[:n_sigmas].cpu().numpy().tolist())

    return results


def run_assay(results_dir, device="cuda"):
    """Run CCS sigma-2 assay on all checkpoints."""
    results_dir = Path(results_dir)
    assay_results = {}

    for manifest_path in sorted(results_dir.glob("manifest_*.json")):
        with open(manifest_path) as f:
            manifest = json.load(f)

        arm = manifest["arm"]
        assay_results[arm] = {"checkpoints": []}

        for ckpt_info in manifest["checkpoints"]:
            ckpt_path = ckpt_info["path"]
            print(f"Assaying {ckpt_path}...", flush=True)

            model = GPT160M().to(device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model"])
            model.eval()

            spectra = extract_spectra(model, device)
            sink_spectra = extract_spectra_sink_projected(model, device)

            n_layers = len(spectra["ccs"])
            layer_s2_ccs = [spectra["ccs"][i][1] for i in range(n_layers)]
            layer_s2_cal = [spectra["cal"][i][1] for i in range(n_layers)]

            layer_pcts = []
            for i in range(n_layers):
                pct = ((layer_s2_ccs[i] - layer_s2_cal[i]) / layer_s2_cal[i]) * 100 if layer_s2_cal[i] != 0 else 0
                layer_pcts.append(pct)

            d0_s2 = np.mean(layer_s2_cal)
            d2_pct = np.mean(layer_pcts)
            layer_cv = np.std(layer_pcts) / np.abs(np.mean(layer_pcts)) if np.mean(layer_pcts) != 0 else 0

            # Sink-projected sigma-2
            proj_s2 = [sink_spectra["projected"][i][1] for i in range(n_layers)]
            std_s2 = [sink_spectra["standard"][i][1] for i in range(n_layers)]
            floor_ratio = np.mean(proj_s2) / np.mean(std_s2) if np.mean(std_s2) != 0 else 0

            ckpt_result = {
                "step": ckpt_info["step"],
                "loss": ckpt_info["loss"],
                "target": ckpt_info["target"],
                "match_type": ckpt_info.get("match_type", "loss"),
                "d0_s2": float(d0_s2),
                "d2_pct": float(d2_pct),
                "layer_cv": float(layer_cv),
                "layer_profile": [round(p, 2) for p in layer_pcts],
                "floor_ratio": float(floor_ratio),
                "sink_projected_s2": [round(s, 2) for s in proj_s2],
            }
            assay_results[arm]["checkpoints"].append(ckpt_result)

            del model
            torch.cuda.empty_cache()

    out_path = results_dir / "assay_results.json"
    with open(out_path, "w") as f:
        json.dump(assay_results, f, indent=2)
    print(f"Assay results saved to {out_path}")
    return assay_results


def analyze(results_dir):
    """Analyze assay results — compute attenuation profiles and trajectory diagnostics."""
    results_dir = Path(results_dir)
    with open(results_dir / "assay_results.json") as f:
        data = json.load(f)

    print("=" * 70)
    print("TWIN OPTIMIZER EXPERIMENT — ANALYSIS")
    print("=" * 70)

    arms = list(data.keys())
    for arm in arms:
        print(f"\n--- {arm.upper()} ---")
        for ckpt in data[arm]["checkpoints"]:
            print(f"  loss={ckpt['target_loss']:.1f} (actual {ckpt['loss']:.3f}), "
                  f"step={ckpt['step']}, D0_s2={ckpt['d0_s2']:.1f}, "
                  f"D2%={ckpt['d2_pct']:.1f}%, CV={ckpt['layer_cv']:.2f}, "
                  f"floor_ratio={ckpt['floor_ratio']:.3f}")

    if "adam" in data and "muon" in data:
        for match_type in ["loss", "step"]:
            print(f"\n--- {match_type.upper()}-MATCHED COMPARISON ---")
            adam_ckpts = {c["target"]: c for c in data["adam"]["checkpoints"]
                         if c.get("match_type", "loss") == match_type}
            muon_ckpts = {c["target"]: c for c in data["muon"]["checkpoints"]
                         if c.get("match_type", "loss") == match_type}

            common = sorted(set(adam_ckpts.keys()) & set(muon_ckpts.keys()),
                           reverse=(match_type == "loss"))

            if not common:
                print("  No matched pairs found.")
                continue

            gaps = []
            for target in common:
                a = adam_ckpts[target]
                m = muon_ckpts[target]
                gap = a["d0_s2"] - m["d0_s2"]
                d2_gap = a["d2_pct"] - m["d2_pct"]
                if match_type == "loss":
                    step_ratio = a["step"] / m["step"] if m["step"] > 0 else float("inf")
                    print(f"  loss={target:.1f}: Adam D0={a['d0_s2']:.1f}, Muon D0={m['d0_s2']:.1f}, "
                          f"gap={gap:.1f}, D2% gap={d2_gap:.1f}%, step_ratio={step_ratio:.2f}x")
                else:
                    loss_diff = a["loss"] - m["loss"]
                    print(f"  step={target}: Adam D0={a['d0_s2']:.1f} (loss={a['loss']:.3f}), "
                          f"Muon D0={m['d0_s2']:.1f} (loss={m['loss']:.3f}), "
                          f"gap={gap:.1f}, loss_diff={loss_diff:.3f}")
                gaps.append(gap)

            if len(gaps) >= 3:
                diffs = [gaps[i+1] - gaps[i] for i in range(len(gaps) - 1)]
                if all(d > 0 for d in diffs):
                    trajectory = "MONOTONIC_DIVERGENCE"
                elif all(abs(d) < 1 for d in diffs):
                    trajectory = "FLAT"
                elif any(d > 0 for d in diffs[:len(diffs)//2]) and any(d < 0 for d in diffs[len(diffs)//2:]):
                    trajectory = "TRANSIENT_SCAFFOLDING"
                else:
                    trajectory = "MIXED"
                print(f"\n  Trajectory ({match_type}-matched): {trajectory}")

        # Triangulation
        print("\n--- TRIANGULATION ---")
        print("  If step-matched and loss-matched trajectories agree: effect is robust")
        print("  If step-matched diverges but loss-matched is flat: throughput confound")
        print("  If loss-matched diverges but step-matched is flat: per-update geometry")

        # Layer-resolved attenuation
        print("\n--- LAYER-RESOLVED ATTENUATION ---")
        for loss_target in common_losses[:1]:
            a_profile = np.array(adam_ckpts[loss_target]["layer_profile"])
            m_profile = np.array(muon_ckpts[loss_target]["layer_profile"])
            attenuation = a_profile - m_profile

            early = np.mean(attenuation[:2])
            mid = np.mean(attenuation[2:8])
            late = np.mean(attenuation[8:])

            print(f"  loss={loss_target:.1f}:")
            print(f"    Early (L0-1): {early:.2f}% (sink-dominated, expect weak)")
            print(f"    Mid (L2-7):   {mid:.2f}% (responsive zone, expect strong)")
            print(f"    Late (L8+):   {late:.2f}%")

            if abs(mid) > abs(early) * 1.5:
                print(f"    -> DEPTH-DEPENDENT: mid-band effect {abs(mid/early) if early != 0 else 'inf'}x early")
            else:
                print(f"    -> FLAT attenuation (falsifies two-source decomposition)")

        # Sink projection control
        print("\n--- SINK PROJECTION CONTROL ---")
        for arm in arms:
            ckpt = data[arm]["checkpoints"][-1]
            print(f"  {arm}: floor_ratio={ckpt['floor_ratio']:.3f} "
                  f"({'floor survives' if ckpt['floor_ratio'] > 0.5 else 'floor collapses'})")


def main():
    parser = argparse.ArgumentParser(description="Twin optimizer experiment")
    parser.add_argument("--phase", choices=["train", "assay", "analyze"], required=True)
    parser.add_argument("--arm", choices=["adam", "muon", "hybrid"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--results-dir", default="results/twin_optimizer")
    args = parser.parse_args()

    if args.phase == "train":
        if not args.arm:
            parser.error("--arm required for train phase")
        train(args.arm, args.device, args.results_dir)
    elif args.phase == "assay":
        run_assay(args.results_dir, args.device)
    elif args.phase == "analyze":
        analyze(args.results_dir)


if __name__ == "__main__":
    main()
