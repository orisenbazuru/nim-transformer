import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
CHECKPOINT_CANDIDATES = [
    ROOT / "checkpoints" / "nim_transformer_move_model.pt",
    ROOT / "nim_transformer_checkpoint.pt",
]
OUT = ROOT / "docs" / "model_weights.json"


def tensor_to_list(value):
    return value.detach().cpu().tolist()


def main():
    checkpoint_path = next((path for path in CHECKPOINT_CANDIDATES if path.exists()), None)
    if checkpoint_path is None:
        candidates = "\n".join(f"  - {path}" for path in CHECKPOINT_CANDIDATES)
        raise FileNotFoundError(f"No transformer checkpoint found. Expected one of:\n{candidates}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint["model_state_dict"]
    payload = {
        "metadata": {
            "checkpoint": str(checkpoint_path.relative_to(ROOT)),
            "architecture": checkpoint.get("architecture"),
            "seed": checkpoint.get("seed"),
            "epochs": checkpoint.get("epochs"),
            "history": checkpoint.get("history", []),
            "note": "Weights exported for the static GitHub Pages Nim transformer app.",
        },
        "weights": {name: tensor_to_list(value) for name, value in state.items()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
