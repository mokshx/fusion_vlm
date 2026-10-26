"""Record the exact hardware/software environment next to every run."""

import os
import platform
import subprocess
import sys

import psutil


def _sh(cmd: list[str], stderr_too: bool = False) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return (r.stdout + (r.stderr if stderr_too else "")).strip()


def collect(server_bin: str = "llama-server") -> dict:
    info = {
        "hostname": platform.node(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cpu_count_logical": psutil.cpu_count(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_gb": psutil.virtual_memory().total / 2**30,
        "load_avg_1m": os.getloadavg()[0],
    }
    out = _sh([server_bin, "--version"], stderr_too=True)  # llama.cpp prints version to stderr
    info["llama_cpp_version"] = " | ".join(l for l in out.splitlines() if l.startswith(("version", "built")))
    if platform.system() == "Darwin":
        info["cpu"] = _sh(["sysctl", "-n", "machdep.cpu.brand_string"])
        info["perf_cores"] = _sh(["sysctl", "-n", "hw.perflevel0.physicalcpu"])
        info["eff_cores"] = _sh(["sysctl", "-n", "hw.perflevel1.physicalcpu"])
        batt = _sh(["pmset", "-g", "batt"])
        info["power_source"] = "AC" if "AC Power" in batt else "battery" if batt else ""
        info["low_power_mode"] = "lowpowermode 1" in _sh(["pmset", "-g"]).replace("  ", " ")
    else:
        info["cpu"] = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo")
                            if l.startswith(("model name", "Model"))), "")
        if os.path.exists("/etc/nv_tegra_release"):
            info["jetson_l4t"] = open("/etc/nv_tegra_release").readline().strip()
            info["jetson_power_mode"] = _sh(["nvpmodel", "-q"])
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name()
    except ImportError:
        pass
    return info
