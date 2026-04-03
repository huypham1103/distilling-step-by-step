import os
from typing import Dict, List, Tuple


def _capability_to_arch(capability: Tuple[int, int]) -> str:
    return f"sm_{capability[0]}{capability[1]}"


def _normalize_arch_list(arch_list: List[str]) -> List[str]:
    normalized = []
    for arch in arch_list:
        if not arch:
            continue
        arch = arch.strip()
        if arch.startswith("compute_"):
            arch = "sm_" + arch.split("_", 1)[1]
        normalized.append(arch)
    return normalized


def collect_torch_runtime_info(torch_module) -> Dict[str, object]:
    info: Dict[str, object] = {
        "torch_version": getattr(torch_module, "__version__", "unknown"),
        "cuda_version": getattr(getattr(torch_module, "version", None), "cuda", None),
        "cuda_available": bool(torch_module.cuda.is_available()),
        "device_type": "cpu",
        "device_name": "cpu",
        "capability": None,
        "arch": None,
        "supported_arches": [],
        "bf16_supported": False,
    }

    if hasattr(torch_module.cuda, "get_arch_list"):
        try:
            info["supported_arches"] = _normalize_arch_list(torch_module.cuda.get_arch_list())
        except Exception:
            info["supported_arches"] = []

    if info["cuda_available"]:
        info["device_type"] = "cuda"
        try:
            info["device_name"] = torch_module.cuda.get_device_name(0)
        except Exception:
            info["device_name"] = "cuda"
        try:
            capability = torch_module.cuda.get_device_capability(0)
            info["capability"] = capability
            info["arch"] = _capability_to_arch(capability)
        except Exception:
            info["capability"] = None
            info["arch"] = None
        if hasattr(torch_module.cuda, "is_bf16_supported"):
            try:
                info["bf16_supported"] = bool(torch_module.cuda.is_bf16_supported())
            except Exception:
                info["bf16_supported"] = False
        return info

    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        info["device_type"] = "mps"
        info["device_name"] = "mps"
    return info


def validate_cuda_runtime(info: Dict[str, object]) -> None:
    if info["device_type"] != "cuda":
        return

    arch = info.get("arch")
    supported_arches = info.get("supported_arches") or []
    if arch and supported_arches and arch not in supported_arches:
        supported_summary = ", ".join(supported_arches)
        raise RuntimeError(
            "The installed PyTorch build does not include CUDA kernels for "
            f"{info['device_name']} ({arch}). Supported architectures in this build: "
            f"{supported_summary}. Install a PyTorch build compiled for this GPU or switch "
            "to a compatible GPU/runtime."
        )


def adapt_args_for_runtime(args, info: Dict[str, object]) -> List[str]:
    adjustments: List[str] = []

    if getattr(args, "bf16", False) and getattr(args, "fp16", False):
        if info.get("bf16_supported"):
            args.fp16 = False
            adjustments.append("Disabled fp16 because bf16 is enabled and supported on this device.")
        else:
            args.bf16 = False
            adjustments.append("Disabled bf16 because this device/runtime does not support bf16; keeping fp16.")

    if getattr(args, "bf16", False) and not info.get("bf16_supported", False):
        args.bf16 = False
        adjustments.append("Disabled bf16 because this device/runtime does not support bf16.")

    if info["device_type"] != "cuda":
        if getattr(args, "bf16", False):
            args.bf16 = False
            adjustments.append(f"Disabled bf16 because the active device is {info['device_type']}.")
        if getattr(args, "fp16", False):
            args.fp16 = False
            adjustments.append(f"Disabled fp16 because the active device is {info['device_type']}.")

    return adjustments


def format_runtime_summary(info: Dict[str, object]) -> str:
    parts = [
        f"device={info['device_type']}",
        f"name={info['device_name']}",
        f"torch={info['torch_version']}",
    ]
    if info.get("cuda_version"):
        parts.append(f"cuda={info['cuda_version']}")
    if info.get("arch"):
        parts.append(f"arch={info['arch']}")
    if info.get("bf16_supported"):
        parts.append("bf16_supported=True")
    return "Runtime: " + ", ".join(parts)


def configure_runtime(args, torch_module) -> Tuple[Dict[str, object], List[str]]:
    info = collect_torch_runtime_info(torch_module)
    validate_cuda_runtime(info)
    adjustments = adapt_args_for_runtime(args, info)

    if info["device_type"] != "cuda":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    return info, adjustments
