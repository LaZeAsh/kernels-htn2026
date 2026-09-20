"""Once-per-process eager compiler metadata for the staged large MLP."""

_seen = set()


def log_compiler_once(label, kernel):
    if label in _seen:
        return
    _seen.add(label)
    print(
        f"compiler {label} regs={kernel.n_regs} spills={kernel.n_spills} "
        f"shared={kernel.metadata.shared} "
        f"wgmma={'wgmma.mma_async' in kernel.asm['ptx']}",
        flush=True,
    )
