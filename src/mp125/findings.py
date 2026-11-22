"""Edit values inside reports/findings.yaml WITHOUT rewriting the file.

yaml.safe_dump would drop the comments and could turn the key `on`/`'North'`
into something else, so we change only the value part of the line.
"""
import re
from pathlib import Path

LINE = re.compile(r"^(\s*)(['\"]?)(\w+)\2:\s*(.*?)\s*(#.*)?$")


def _fmt(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f'"{v}"'
    return repr(v)


def set_values(path, updates):
    """updates = {("monitoring", "psi", "stops"): 1.23, ...}. Returns paths not found."""
    lines = Path(path).read_text().splitlines()
    stack, out, seen = [], [], set()
    for line in lines:
        m = LINE.match(line) if line.strip() and not line.lstrip().startswith("#") else None
        if m:
            indent, quote, key, value, comment = m.groups()
            while stack and stack[-1][0] >= len(indent):
                stack.pop()
            full = tuple(k for _, k in stack) + (key,)
            if full in updates:
                new = f"{indent}{quote}{key}{quote}: {_fmt(updates[full])}"
                line = new + (f"   {comment}" if comment else "")
                seen.add(full)
            stack.append((len(indent), key))
        out.append(line)
    Path(path).write_text("\n".join(out) + "\n")
    return [p for p in updates if p not in seen]
