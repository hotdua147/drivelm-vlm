"""Small compatibility layer for transformers 4.x and 5.x."""
from __future__ import annotations


def _version() -> tuple:
    import transformers
    parts = []
    for x in transformers.__version__.split(".")[:2]:
        digits = "".join(ch for ch in x if ch.isdigit())
        parts.append(int(digits or 0))
    return tuple(parts)


def dtype_kwargs(dtype) -> dict:
    """``dtype=`` since transformers 4.56 (``torch_dtype`` is deprecated, removed in 5.x)."""
    return {"dtype": dtype} if _version() >= (4, 56) else {"torch_dtype": dtype}


def set_image_pixels(processor, min_pixels: int, max_pixels: int) -> dict:
    """Limit the image resolution of a Qwen-VL processor, whatever the transformers version.

    Qwen2/2.5/3-VL image processors store the pixel budget either as ``size =
    {"shortest_edge": min_pixels, "longest_edge": max_pixels}`` or as ``min_pixels`` /
    ``max_pixels`` attributes. Returns what was set, so it can be logged.
    """
    ip = getattr(processor, "image_processor", processor)
    done = {}
    size = getattr(ip, "size", None)
    if isinstance(size, dict) and ("longest_edge" in size or "shortest_edge" in size):
        ip.size = {**size, "shortest_edge": min_pixels, "longest_edge": max_pixels}
        done["size"] = dict(ip.size)
    elif size is not None and (hasattr(size, "longest_edge") or hasattr(size, "shortest_edge")):
        # transformers 5 stores size as an object with attributes (SizeDict)
        try:
            size.shortest_edge, size.longest_edge = min_pixels, max_pixels
            done["size"] = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
        except Exception:
            pass
    if not done:
        done["note"] = (f"no pixel-budget field found on {type(ip).__name__}; images are already resized "
                        "to data.cell before the processor, so the token count is still bounded")
    for attr, val in (("min_pixels", min_pixels), ("max_pixels", max_pixels)):
        if hasattr(ip, attr):
            setattr(ip, attr, val)
            done[attr] = val
    return done
