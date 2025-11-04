"""sample nanoEBM

Usage:
    python sample.py checkpoint=out_ebt/ckpt_step_1000.pt
    python sample.py checkpoint=out_ebt/final.pt max_new_tokens=500 prompt="HAMLET:"
    # Thinking mode (iterative refinement)
    python sample.py checkpoint=out_ebt/ckpt_step_1000.pt use_thinking=true think_steps=4 topk=64
    # Thinking + sampling (stabilizes and reduces repetition)
    python sample.py checkpoint=out_ebt/final.pt use_thinking=true think_steps=4 topk=64 sample=true sample_temp=1.2 sample_top_p=0.9
    # Baseline transformer checkpoint
    python sample.py checkpoint=out_ebt/run_baseline/final.pt model_mode=lm sample=true sample_temp=1.1
"""
import chz
import torch
import torch.nn.functional as F
from nanoebm.config import ModelConfig
from nanoebm.model import EBM
from nanoebm.lm import AutoregressiveLM
from nanoebm.data import CharDataset


def find_latest_checkpoint(base_dir: str = "out_ebt") -> str:
    """Find the latest checkpoint by looking for the newest run_* directory."""
    import os
    import glob

    # Look for run_* directories
    run_dirs = glob.glob(os.path.join(base_dir, "run_*"))
    if not run_dirs:
        raise FileNotFoundError(f"No run_* directories found in {base_dir}")

    # Sort by modification time (newest first)
    run_dirs.sort(key=os.path.getmtime, reverse=True)
    latest_dir = run_dirs[0]

    # Check for final.pt in the latest directory
    checkpoint_path = os.path.join(latest_dir, "final.pt")
    if os.path.exists(checkpoint_path):
        return checkpoint_path

    # Fallback: look for any .pt file in the latest directory
    pt_files = glob.glob(os.path.join(latest_dir, "*.pt"))
    if pt_files:
        pt_files.sort(key=os.path.getmtime, reverse=True)
        return pt_files[0]

    raise FileNotFoundError(f"No .pt files found in {latest_dir}")


@chz.chz
class SampleConfig:
    """Configuration for sampling from a trained EBM model"""
    checkpoint: str | None = None  # None = auto-detect latest checkpoint
    data_path: str = "shakespeare.txt"  # Path to training data (for vocab)
    prompt: str = "ROMEO:"  # Text prompt to start generation
    max_new_tokens: int = 200  # Number of tokens to generate
    
    # EBM parameters
    mode: str = "think"  # Sampling mode: 'fast' (System 1), 'think' (System 2)
    think_steps: int = 4       # Number of refinement steps when thinking
    topk: int | None = 50      # Restrict to top-k tokens (None = use all vocab)

    # Sampling parameters
    sample: bool = False  # Sample from distribution vs greedy
    sample_temp: float = 1.0  # Temperature for sampling
    model_mode: str | None = None  # Optional override: 'ebm' or 'lm'


@torch.no_grad()
def decode(idx, itos):
    return "".join(itos[i] for i in idx.tolist())


def main(cfg: SampleConfig):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Auto-detect latest checkpoint if not specified
    checkpoint = cfg.checkpoint
    if checkpoint is None:
        checkpoint = find_latest_checkpoint()
        print(f"Auto-detected latest checkpoint: {checkpoint}")
    else:
        print(f"Loading checkpoint: {checkpoint}")

    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    model_cfg = ModelConfig(**ckpt["config"]["model"])
    ckpt_train_cfg = ckpt["config"].get("train", {})
    ckpt_mode = ckpt_train_cfg.get("mode", "ebm").lower()
    override_mode = cfg.model_mode.lower() if cfg.model_mode else None
    sample_mode = override_mode or ckpt_mode
    if sample_mode in {"gpt", "transformer"}:
        sample_mode = "lm"
    if sample_mode not in {"ebm", "lm"}:
        raise ValueError(f"Unsupported model_mode '{sample_mode}'. Use 'ebm' or 'lm'.")
    if override_mode and sample_mode != ckpt_mode:
        raise ValueError(
            f"Checkpoint trained as '{ckpt_mode}', cannot override to '{override_mode}'."
        )

    # Initialize model and load weights
    if sample_mode == "ebm":
        model = EBM(model_cfg).to(device)
    else:
        model = AutoregressiveLM(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"Loaded model from step {ckpt['step']} (mode={sample_mode})")

    # Load dataset for vocabulary and decoding
    ds = CharDataset(cfg.data_path, block_size=model_cfg.block_size, split="train")
    stoi, itos = ds.stoi, ds.itos
    if len(stoi) != model_cfg.vocab_size:
        print(
            f"Warning: dataset vocab_size ({len(stoi)}) != model vocab_size ({model_cfg.vocab_size}). "
            "Ensure you are using the same data file used for training."
        )

    # Encode prompt (filter unknown chars for robustness)
    print(f"\nPrompt: {cfg.prompt!r}")
    known = [c for c in cfg.prompt if c in stoi]
    dropped = [c for c in cfg.prompt if c not in stoi]
    if dropped:
        print(f"[warn] Dropping {len(dropped)} unknown chars from prompt: {repr(''.join(dropped))}")
    idx = torch.tensor([[stoi[c] for c in known]], dtype=torch.long, device=device)

    temperature = cfg.sample_temp if cfg.sample else 1.0
    top_k = cfg.topk

    # Generate based on mode
    if sample_mode == "ebm":
        if cfg.mode == "fast":
            print("Generating with System 1 (fast mode)...")
            out = model.generate(
                idx.clone(),
                max_new_tokens=cfg.max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                use_thinking=False
            )
        elif cfg.mode == "think":
            print(f"Generating with System 2 (thinking mode, steps={cfg.think_steps})...")
            out = model.generate(
                idx.clone(),
                max_new_tokens=cfg.max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                use_thinking=True,
                think_steps=cfg.think_steps
            )
        else:
            raise ValueError(f"Unknown mode: {cfg.mode}. Use 'fast', 'think'")
    else:
        if cfg.mode == "think":
            print("[warn] 'think' mode not available for baseline LM; using standard autoregressive decoding.")
        else:
            print("Generating with baseline transformer (autoregressive LM)...")
        out = model.generate(
            idx.clone(),
            max_new_tokens=cfg.max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    # Decode and print
    txt = decode(out[0], itos)
    print("\n" + "="*80)
    print(txt)
    print("="*80)


if __name__ == "__main__":
    config = chz.entrypoint(SampleConfig)
    main(config)
