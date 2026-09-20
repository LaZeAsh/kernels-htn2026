"""One-time eager Triton compiler metadata for staged experiments."""

_seen = set()


def log_compiler_once(label, kernel, config=None):
    if label in _seen:
        return
    _seen.add(label)
    ptx = kernel.asm["ptx"]
    fields = [
        f"compiler {label}",
        f"regs={kernel.n_regs}",
        f"spills={kernel.n_spills}",
        f"shared={kernel.metadata.shared}",
        f"wgmma={'wgmma.mma_async' in ptx}",
    ]
    if config is not None:
        fields.extend((f"rows={config.kwargs['ROWS']}",
                       f"warps={config.num_warps}"))
    print(" ".join(fields), flush=True)
