"""Weight-file helpers for nrs-large (DLNR): load a state dict from .pth / .pt / .safetensors, strip the
DataParallel 'module.' prefix, and fingerprint the weights independently of the file format."""
import hashlib
import os


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def strip_module(sd):
    if sd and all(k.startswith("module.") for k in sd):
        return {k[len("module."):]: v for k, v in sd.items()}, True
    return sd, False


def load_state(path):
    """Returns (state_dict without 'module.', info dict). Accepts a plain state dict, {'model': ...} (our full-state
    checkpoints) or {'state_dict': ...}; .safetensors files are read with safetensors."""
    import torch
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd, wrapped = load_file(path), "safetensors"
    else:
        ck = torch.load(path, map_location="cpu", weights_only=True)
        wrapped = "plain"
        if isinstance(ck, dict) and "model" in ck and isinstance(ck["model"], dict):
            ck, wrapped = ck["model"], "model"
        elif isinstance(ck, dict) and "state_dict" in ck:
            ck, wrapped = ck["state_dict"], "state_dict"
        sd = ck
    sd, had_module = strip_module(sd)
    return sd, {"file": os.path.abspath(path), "file_sha256": sha256_file(path), "container": wrapped,
                "had_module_prefix": had_module, "n_tensors": len(sd)}


def weights_fingerprint(sd):
    """sha256 over (key, dtype, shape, bytes) in key order; same weights -> same value in any file format."""
    import torch
    h = hashlib.sha256()
    for k in sorted(sd):
        t = sd[k]
        if not torch.is_tensor(t):
            continue
        t = t.detach().cpu().contiguous()
        h.update(k.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()
