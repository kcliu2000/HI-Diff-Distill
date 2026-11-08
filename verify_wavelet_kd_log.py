from pathlib import Path
import math
import re


log_root = Path(
    "experiments/"
    "Student-Lite-S2-OutputWaveletKD-Smoke"
)

logs = list(log_root.glob("*.log"))

if not logs:
    raise FileNotFoundError(
        f"No logs found in {log_root}"
    )

latest_log = max(
    logs,
    key=lambda path: path.stat().st_mtime,
)

pattern = re.compile(
    r"iter:\s*(?P<iteration>\d+).*?"
    r"l_pix:\s*(?P<pix>[\d.eE+-]+).*?"
    r"l_pix_diff:\s*(?P<pix_diff>[\d.eE+-]+).*?"
    r"l_output_kd_raw:\s*(?P<out_raw>[\d.eE+-]+).*?"
    r"l_output_kd:\s*(?P<out>[\d.eE+-]+).*?"
    r"l_wavelet_ll:\s*(?P<ll>[\d.eE+-]+).*?"
    r"l_wavelet_lh:\s*(?P<lh>[\d.eE+-]+).*?"
    r"l_wavelet_hl:\s*(?P<hl>[\d.eE+-]+).*?"
    r"l_wavelet_hh:\s*(?P<hh>[\d.eE+-]+).*?"
    r"l_wavelet_kd_raw:\s*(?P<wav_raw>[\d.eE+-]+).*?"
    r"l_wavelet_kd:\s*(?P<wav>[\d.eE+-]+).*?"
    r"l_total:\s*(?P<total>[\d.eE+-]+)"
)

text = latest_log.read_text(
    encoding="utf-8",
    errors="ignore",
)

records = {}

for match in pattern.finditer(text):
    data = match.groupdict()
    iteration = int(data.pop("iteration"))

    if iteration in (1, 10, 20):
        records[iteration] = {
            key: float(value)
            for key, value in data.items()
        }

if set(records) != {1, 10, 20}:
    raise RuntimeError(
        f"Missing target iterations: "
        f"{sorted({1, 10, 20} - set(records))}"
    )

print("Log:", latest_log)
print()

for iteration in (1, 10, 20):
    values = records[iteration]

    expected_out = 0.5 * values["out_raw"]

    expected_wav_raw = (
        values["ll"]
        + 2.0 * values["lh"]
        + 2.0 * values["hl"]
        + 2.0 * values["hh"]
    )

    expected_wav = 0.5 * expected_wav_raw

    expected_total = (
        values["pix"]
        + values["pix_diff"]
        + values["out"]
        + values["wav"]
    )

    checks = {
        "Output KD alpha": math.isclose(
            values["out"],
            expected_out,
            rel_tol=3e-3,
            abs_tol=3e-6,
        ),
        "Wavelet sub-band sum": math.isclose(
            values["wav_raw"],
            expected_wav_raw,
            rel_tol=3e-3,
            abs_tol=3e-6,
        ),
        "Wavelet beta": math.isclose(
            values["wav"],
            expected_wav,
            rel_tol=3e-3,
            abs_tol=3e-6,
        ),
        "Total loss": math.isclose(
            values["total"],
            expected_total,
            rel_tol=3e-3,
            abs_tol=3e-6,
        ),
    }

    print(f"Iteration {iteration}")

    for name, passed in checks.items():
        print(
            f"  {name}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    if not all(checks.values()):
        raise RuntimeError(
            f"Verification failed at "
            f"iteration {iteration}"
        )

print()
print("Wavelet KD loss verification: PASS")
