from pathlib import Path
import importlib.util

spec = importlib.util.find_spec("basicsr")
if spec is None or not spec.submodule_search_locations:
    raise RuntimeError("BasicSR is not installed.")

basicsr_dir = Path(next(iter(spec.submodule_search_locations)))
target = basicsr_dir / "data" / "degradations.py"

old = (
    "from torchvision.transforms.functional_tensor "
    "import rgb_to_grayscale"
)
new = (
    "from torchvision.transforms.functional "
    "import rgb_to_grayscale"
)

if not target.exists():
    raise FileNotFoundError(target)

text = target.read_text(encoding="utf-8")

if old in text:
    backup = target.with_suffix(".py.bak")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
    target.write_text(text.replace(old, new), encoding="utf-8")
    print(f"Patched: {target}")
elif new in text:
    print(f"Already patched: {target}")
else:
    raise RuntimeError(f"Expected import not found in {target}")
