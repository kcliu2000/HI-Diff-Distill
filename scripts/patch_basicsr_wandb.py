from pathlib import Path
import importlib.util


spec = importlib.util.find_spec("basicsr")

if spec is None or not spec.submodule_search_locations:
    raise RuntimeError("BasicSR is not installed.")

basicsr_dir = Path(
    next(iter(spec.submodule_search_locations))
)

target = basicsr_dir / "utils" / "logger.py"

if not target.exists():
    raise FileNotFoundError(target)


old_code = "wandb_id = wandb.util.generate_id()"

new_code = (
    "wandb_id = "
    "__import__('uuid').uuid4().hex[:8]"
)


text = target.read_text(
    encoding="utf-8"
)


if old_code in text:
    backup = target.with_suffix(
        ".py.before_wandb_patch"
    )

    if not backup.exists():
        backup.write_text(
            text,
            encoding="utf-8",
        )
        print(f"Backup created: {backup}")

    patched_text = text.replace(
        old_code,
        new_code,
    )

    target.write_text(
        patched_text,
        encoding="utf-8",
    )

    print(f"Patched: {target}")

elif new_code in text:
    print(f"Already patched: {target}")

else:
    print("Expected line was not found.")
    print()
    print("Lines containing generate_id or wandb_id:")

    for line_number, line in enumerate(
        text.splitlines(),
        start=1,
    ):
        if (
            "generate_id" in line
            or "wandb_id" in line
        ):
            print(
                f"{line_number}: {line}"
            )

    raise RuntimeError(
        "BasicSR logger structure is different. "
        "No changes were applied."
    )
