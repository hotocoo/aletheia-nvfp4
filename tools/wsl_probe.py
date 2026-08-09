"""Capability probe for the WSL2 side: does this box actually reach NVFP4?"""

import importlib.metadata as md

import torch

print("torch      :", torch.__version__, "| cuda", torch.version.cuda,
      "| available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device     :", torch.cuda.get_device_name(), torch.cuda.get_device_capability())

try:
    import transformer_engine.pytorch as te  # noqa: F401
    # The distribution name depends on how TE got here: `transformer_engine_cu13` for NVIDIA's
    # prebuilt wheel, plain `transformer_engine` for a source build. Asking for the wrong one
    # raises PackageNotFoundError and makes a perfectly working install look absent.
    for _dist in ("transformer_engine", "transformer_engine_cu13", "transformer_engine_cu12"):
        try:
            print("transformer_engine:", md.version(_dist), f"({_dist})")
            break
        except md.PackageNotFoundError:
            continue
    try:
        from transformer_engine.pytorch.quantization import (
            check_fp8_support, check_mxfp8_support, check_nvfp4_support)
    except Exception:
        from transformer_engine.pytorch.fp8 import (
            check_fp8_support, check_mxfp8_support, check_nvfp4_support)
    for name, fn in (("NVFP4", check_nvfp4_support), ("MXFP8", check_mxfp8_support),
                     ("FP8", check_fp8_support)):
        ok, why = fn()
        print(f"{name:6s}: {ok}  {'' if ok else why}")
    from transformer_engine.common.recipe import NVFP4BlockScaling
    print("NVFP4BlockScaling:", NVFP4BlockScaling())
except Exception as exc:
    print("transformer_engine UNAVAILABLE:", type(exc).__name__, exc)

try:
    import triton
    print("triton     :", triton.__version__, "-> torch.compile available")
except Exception as exc:
    print("triton missing:", exc)
