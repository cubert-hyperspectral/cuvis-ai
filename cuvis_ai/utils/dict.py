from typing import Any


def remove_prefix_str(s, prefix):
    if s.startswith(prefix):
        return s[len(prefix) :]
    return s


def remove_prefix(d: dict[str, Any], prefix, keep_only=False):
    return {
        remove_prefix_str(k, prefix): v
        for k, v in d.items()
        if k.startswith(prefix) or not keep_only
    }


def add_prefix(d: dict[str, Any], prefix):
    return {prefix + k: v for k, v in d.items()}
