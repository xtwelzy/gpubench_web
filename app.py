from __future__ import annotations

import os
import platform
import threading
import time
import random
import subprocess
import shutil
import glob
import re
import csv, io, datetime
from dataclasses import dataclass, asdict
from collections import deque
from typing import Optional, Dict, Any

from flask import Flask, render_template, jsonify, request, send_file

# ===================== NVML bootstrap =====================
_NV_OK = False
_NV_ERR: Optional[str] = None
_HANDLE = None

def _hint_nvml_path_windows() -> Optional[str]:
    p = os.environ.get("NVML_DLL")
    if p and os.path.isfile(p): return p
    try:
        nvsmi = shutil.which("nvidia-smi")
        if nvsmi:
            cand = os.path.join(os.path.dirname(nvsmi), "nvml.dll")
            if os.path.isfile(cand): return cand
    except Exception:
        pass
    for cand in (
        r"C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll",
        r"C:\Windows\System32\nvml.dll",
    ):
        if os.path.isfile(cand): return cand
    try:
        for path in glob.glob(r"C:\Windows\System32\DriverStore\FileRepository\*\nvml.dll"):
            if os.path.isfile(path): return path
    except Exception:
        pass
    return None

try:
    if platform.system() == "Windows":
        dll = _hint_nvml_path_windows()
        if dll:
            try: os.add_dll_directory(os.path.dirname(dll))
            except Exception: pass
            os.environ["NVML_DLL"] = dll
    import pynvml  # pip install nvidia-ml-py3
    try:
        pynvml.nvmlInit()
        _NV_OK = True
        _HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception as e:
        _NV_ERR = f"NVML init failed: {e!r}"
except Exception as e:
    _NV_ERR = f"pynvml import failed: {e!r}"

# ========== nvidia-smi fallback ==========
def _find_nvidia_smi() -> Optional[str]:
    cand = os.environ.get("NVIDIA_SMI")
    if cand and os.path.isfile(cand): return cand
    cand = shutil.which("nvidia-smi")
    if cand and os.path.isfile(cand): return cand
    for pat in (r"C:\Windows\System32\DriverStore\FileRepository\nv_dispi.inf_amd64_*",
                r"C:\Program Files\NVIDIA Corporation\NVSMI"):
        for root in glob.glob(pat):
            exe = os.path.join(root, "nvidia-smi.exe")
            if os.path.isfile(exe): return exe
    return None

def _read_metrics_via_nvsmi() -> Optional[Dict[str, Any]]:
    exe = _find_nvidia_smi()
    if not exe: return None
    query = [
        "name","driver_version","temperature.gpu","utilization.gpu",
        "memory.used","memory.total","fan.speed","power.draw",
        "clocks.gr","clocks.mem"
    ]
    cmd = [exe, f"--query-gpu={','.join(query)}", "--format=csv,noheader,nounits"]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=3)
    except Exception:
        return None
    line = out.strip().splitlines()[0].strip()
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < len(query): return None

    def to_int(x, default=0):
        try: return int(float(re.sub(r"[^\d.\-]", "", x)))
        except Exception: return default

    return {
        "name": parts[0],
        "driver": parts[1],
        "temperature": to_int(parts[2]),
        "load": to_int(parts[3]),
        "mem_used_mb": to_int(parts[4]),
        "mem_total_mb": to_int(parts[5]),
        "fan_percent": to_int(parts[6]),
        "power_w": to_int(parts[7]),
        "core_clock_mhz": to_int(parts[8]),
        "mem_clock_mhz": to_int(parts[9]),
        "nvsmi_ok": True,
        "nvsmi_path": exe,
    }

# ========== Stress backends ==========
_CUPY_OK = False
_OPENCL_OK = False
try:
    import cupy as cp  # pip install cupy-cuda12x  (или подходящий под драйвер)
    _CUPY_OK = True
except Exception:
    pass
try:
    import numpy as _np
    import pyopencl as cl        # pip install pyopencl (если нужно)
    _OPENCL_OK = True
except Exception:
    pass

app = Flask(__name__)

# ========== State & history ==========
hist_temperature = deque(maxlen=600)
hist_load = deque(maxlen=600)
hist_labels = deque(maxlen=600)

@dataclass
class TestResult:
    id: int
    profile: str
    backend: str
    started_at: str
    finished_at: str
    duration_sec: int
    max_temp: int
    avg_temp: float
    max_load: int
    avg_load: float
    max_power: int
    avg_power: float
    status: str

class State:
    def __init__(self):
        self.running: bool = False
        self.started_at: Optional[float] = None
        self.profile: Optional[str] = None
        self.lock = threading.Lock()
        self.session_points: list[tuple[float, int, int, int]] = []

STATE = State()
RESULTS: list[TestResult] = []
_NEXT_ID = 1

# ========== Metrics ==========
def _safe_call(fn, *args, default=None):
    try: return fn(*args)
    except Exception: return default

def _mock_metrics(nvml_error: Optional[str]) -> Dict[str, Any]:
    temp = random.randint(45, 75)
    util = random.randint(5, 95)
    fan = random.randint(20, 70)
    power = random.randint(100, 350)
    return {
        "name": "GPU (mock)",
        "driver": "n/a",
        "temperature": temp,
        "load": util,
        "mem_used_mb": 6200,
        "mem_total_mb": 12288,
        "fan_percent": fan,
        "power_w": power,
        "core_clock_mhz": 0,
        "mem_clock_mhz": 0,
        "nvml_ok": False,
        "nvml_error": nvml_error,
        "nvsmi_ok": False,
        "nvsmi_path": None,
    }

def read_gpu_metrics() -> Dict[str, Any]:
    # 1) NVML
    if _NV_OK and _HANDLE is not None:
        try:
            name = _safe_call(pynvml.nvmlDeviceGetName, _HANDLE, default=b"").decode("utf-8") or "Unknown GPU"
            driver = _safe_call(pynvml.nvmlSystemGetDriverVersion, default=b"").decode("utf-8") or "unknown"
            temp = int(_safe_call(pynvml.nvmlDeviceGetTemperature, _HANDLE, pynvml.NVML_TEMPERATURE_GPU, default=0) or 0)
            util_obj = _safe_call(pynvml.nvmlDeviceGetUtilizationRates, _HANDLE, default=None)
            util = int(getattr(util_obj, "gpu", 0) or 0)
            mem = _safe_call(pynvml.nvmlDeviceGetMemoryInfo, _HANDLE, default=None)
            if mem:
                mem_used_mb = int(mem.used / 1024 / 1024)
                mem_total_mb = int(mem.total / 1024 / 1024)
            else:
                mem_used_mb = mem_total_mb = 0
            fan = int(_safe_call(pynvml.nvmlDeviceGetFanSpeed, _HANDLE, default=0) or 0)
            power_mw = int(_safe_call(pynvml.nvmlDeviceGetPowerUsage, _HANDLE, default=0) or 0)
            power_w = power_mw // 1000 if power_mw else 0
            core_clock = int(_safe_call(pynvml.nvmlDeviceGetClockInfo, _HANDLE, pynvml.NVML_CLOCK_GRAPHICS, default=0) or 0)
            mem_clock  = int(_safe_call(pynvml.nvmlDeviceGetClockInfo, _HANDLE, pynvml.NVML_CLOCK_MEM,      default=0) or 0)
            return {
                "name": name, "driver": driver,
                "temperature": temp, "load": util,
                "mem_used_mb": mem_used_mb, "mem_total_mb": mem_total_mb,
                "fan_percent": fan, "power_w": power_w,
                "core_clock_mhz": core_clock, "mem_clock_mhz": mem_clock,
                "nvml_ok": True, "nvml_error": None,
                "nvsmi_ok": False, "nvsmi_path": None,
            }
        except Exception:
            pass
    # 2) nvidia-smi
    via_smi = _read_metrics_via_nvsmi()
    if via_smi:
        return {**via_smi, "nvml_ok": False, "nvml_error": _NV_ERR or "NVML not available; using nvidia-smi"}
    # 3) mock
    return _mock_metrics(nvml_error=_NV_ERR or "NVML & nvidia-smi not available")

def _metrics_sampler():
    while True:
        m = read_gpu_metrics()
        hist_temperature.append(m["temperature"])
        hist_load.append(m["load"])
        hist_labels.append(time.strftime("%H:%M:%S"))
        if STATE.running:
            STATE.session_points.append((time.time(), int(m["temperature"]), int(m["load"]), int(m.get("power_w", 0) or 0)))
        time.sleep(1)

threading.Thread(target=_metrics_sampler, daemon=True).start()

# ========== Stress ==========
class StressWorker:
    def __init__(self):
        self._th: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self, profile: str):
        if self.is_running(): return
        self._stop.clear()
        self._th = threading.Thread(target=self._run, args=(profile,), daemon=True)
        self._th.start()

    def stop(self):
        self._stop.set()
        t = self._th
        if t and t.is_alive(): t.join(timeout=3.0)
        self._th = None
        self._stop.clear()

    def is_running(self) -> bool:
        return self._th is not None and self._th.is_alive() and not self._stop.is_set()

    def _profile_params(self, profile: str):
        if profile == "quick":   return (10 * 60, 0.6, False)
        if profile == "full":    return (30 * 60, 1.0, False)
        if profile == "thermal": return (60 * 60, 1.2, False)
        if profile == "memory":  return (45 * 60, 1.0, True)
        return (10 * 60, 0.6, False)

    def _run(self, profile: str):
        dur_sec, intensity, memory_mode = self._profile_params(profile)
        deadline = time.time() + dur_sec
        if _CUPY_OK:
            try:
                self._burn_cupy(intensity, deadline, memory_mode); return
            except Exception:
                pass
        if _OPENCL_OK:
            try:
                self._burn_opencl(intensity, deadline, memory_mode); return
            except Exception:
                pass
        self._burn_cpu(intensity, deadline, memory_mode)

    # --- CUDA/CuPy ---
    def _burn_cupy(self, intensity: float, deadline: float, memory_mode: bool):
        import cupy as cp
        try:
            cp.cuda.Device(0).use()
        except Exception:
            pass

        a_w = cp.random.random((2048, 2048), dtype=cp.float32)
        b_w = cp.random.random((2048, 2048), dtype=cp.float32)
        for _ in range(20):
            _ = a_w.dot(b_w)
        cp.cuda.Stream.null.synchronize()

        def _try_size(n):
            a = cp.random.random((n, n), dtype=cp.float32)
            b = cp.random.random((n, n), dtype=cp.float32)
            c = cp.zeros((n, n), dtype=cp.float32)
            cp.cuda.Stream.null.synchronize()
            return a, b, c

        n_candidates = [12288, 11008, 10240, 9216, 8192]
        mats = None
        for n in n_candidates:
            try:
                mats = _try_size(n)
                break
            except Exception:
                cp.get_default_memory_pool().free_all_blocks()
        if mats is None:
            mats = _try_size(6144)

        a, b, c = mats
        scratch = None
        if memory_mode:
            try:
                chunks = []
                for _ in range(20):  # ~3–4 ГБ
                    chunks.append(cp.empty((4096, 4096), dtype=cp.float32))
                scratch = chunks
            except Exception:
                pass

        last_scale = time.time()
        while not self._stop.is_set() and time.time() < deadline:
            cp.matmul(a, b, out=c)
            c += a
            if memory_mode and scratch:
                for buf in scratch[:5]:
                    buf *= 1.000001
            cp.cuda.Stream.null.synchronize()
            if time.time() - last_scale > 15:
                try:
                    new_n = a.shape[0] + 512
                    a = cp.random.random((new_n, new_n), dtype=cp.float32)
                    b = cp.random.random((new_n, new_n), dtype=cp.float32)
                    c = cp.zeros((new_n, new_n), dtype=cp.float32)
                except Exception:
                    pass
                last_scale = time.time()

    # --- OpenCL ---
    def _burn_opencl(self, intensity: float, deadline: float, memory_mode: bool):
        import numpy as np
        import pyopencl as cl
        platforms = cl.get_platforms()
        gpus = []
        for p in platforms:
            gpus.extend(p.get_devices(device_type=cl.device_type.GPU))
        if not gpus:
            raise RuntimeError("No OpenCL GPU devices")
        ctx = cl.Context(devices=gpus)
        queue = cl.CommandQueue(ctx)
        base = 1024 * 1024 * (64 if intensity >= 1.0 else 32)
        n = int(base * min(max(intensity, 0.5), 2.0))
        A = np.random.rand(n).astype("float32")
        B = np.random.rand(n).astype("float32")
        C = np.zeros(n, dtype="float32")
        mf = cl.mem_flags
        dA = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
        dB = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
        dC = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=C)
        prg = cl.Program(ctx, """
        __kernel void muladd(__global const float* a,
                             __global const float* b,
                             __global float* c,
                             const float alpha,
                             const float beta) {
            int gid = get_global_id(0);
            c[gid] = alpha * a[gid] * b[gid] + beta * c[gid];
        }""").build()
        alpha = _np.float32(1.000001 if memory_mode else 1.0)
        beta  = _np.float32(1.000001 if memory_mode else 1.0)
        while not self._stop.is_set() and time.time() < deadline:
            prg.muladd(queue, (n,), None, dA, dB, dC, alpha, beta)
            queue.finish()

    # --- CPU fallback ---
    def _burn_cpu(self, intensity: float, deadline: float, memory_mode: bool):
        import numpy as np
        base = 2048 if intensity < 1.0 else 4096
        n = int(base * min(max(intensity, 0.5), 2.0))
        a = np.random.rand(n, n).astype("float32")
        b = np.random.rand(n, n).astype("float32")
        c = np.zeros((n, n), dtype="float32")
        while not self._stop.is_set() and time.time() < deadline:
            c += a @ b
            if memory_mode: c *= 1.000001

STRESS = StressWorker()

# ========== Routes ==========
@app.route("/")
def index():
    # index.html и stress.html должны лежать в папке templates/
    return render_template("index.html", brand="gpubench")

@app.route("/stress")
def stress():
    return render_template("stress.html", brand="gpubench")

@app.route("/api/gpu")
def api_gpu():
    return jsonify(read_gpu_metrics())

@app.route("/api/history")
def api_history():
    labels = list(hist_labels)[-24:]
    temps  = list(hist_temperature)[-24:]
    loads  = list(hist_load)[-24:]
    if len(labels) < 24:
        need = 24 - len(labels)
        pad = [f"-{i}s" for i in range(need, 0, -1)]
        labels = pad + labels
        temps  = ([temps[0]] if temps else [50]) * need + temps
        loads  = ([loads[0]] if loads else [10]) * need + loads
    return jsonify({"labels": labels, "temperature": temps, "load": loads})

@app.route("/api/test/start", methods=["POST"])
def api_test_start():
    body = request.get_json(silent=True) or {}
    profile = body.get("profile", "quick")
    with STATE.lock:
        if STATE.running:
            return jsonify({"ok": False, "error": "already_running"}), 400
        STATE.session_points = []
        STATE.running = True
        STATE.started_at = time.time()
        STATE.profile = profile
        STRESS.start(profile)
    return jsonify({"ok": True, "running": True, "profile": profile})

def _finalize_result() -> TestResult:
    global _NEXT_ID
    points = STATE.session_points[:]
    started_ts = STATE.started_at or time.time()
    finished_ts = time.time()
    duration = int(finished_ts - started_ts)
    if not points: points = [(time.time(), 0, 0, 0)]
    temps  = [t for _, t, _, _ in points]
    loads  = [l for _, _, l, _ in points]
    powers = [p for _, _, _, p in points if isinstance(p, int)]
    max_temp = max(temps)
    avg_temp = sum(temps) / len(temps)
    max_load = max(loads)
    avg_load = sum(loads) / len(loads)
    max_power = max(powers) if powers else 0
    avg_power = int(sum(powers) / len(powers)) if powers else 0
    status = "Passed" if max_temp < 87 else "Warning"
    started_iso  = datetime.datetime.fromtimestamp(started_ts).isoformat(timespec="seconds")
    finished_iso = datetime.datetime.fromtimestamp(finished_ts).isoformat(timespec="seconds")
    res = TestResult(
        id=_NEXT_ID,
        profile=STATE.profile or "unknown",
        backend=("cupy" if _CUPY_OK else ("opencl" if _OPENCL_OK else "cpu-fallback")),
        started_at=started_iso,
        finished_at=finished_iso,
        duration_sec=duration,
        max_temp=int(max_temp),
        avg_temp=round(avg_temp, 1),
        max_load=int(max_load),
        avg_load=round(avg_load, 1),
        max_power=int(max_power),
        avg_power=int(avg_power),
        status=status,
    )
    _NEXT_ID += 1
    return res

@app.route("/api/test/stop", methods=["POST"])
def api_test_stop():
    with STATE.lock:
        STRESS.stop()
        result = _finalize_result()
        RESULTS.append(result)
        STATE.running = False
        STATE.profile = None
        STATE.started_at = None
        STATE.session_points = []
    return jsonify({"ok": True, "running": False, "result": asdict(result)})

@app.route("/api/test/status")
def api_test_status():
    with STATE.lock:
        running = STATE.running and STRESS.is_running()
        elapsed = int(time.time() - STATE.started_at) if (STATE.started_at and running) else 0
        if STATE.running and not running:
            STATE.running = False
            STATE.profile = None
            STATE.started_at = None
    data = read_gpu_metrics()
    data.update({
        "running": STATE.running,
        "elapsed": elapsed,
        "profile": STATE.profile,
        "stress_backend": "cupy" if _CUPY_OK else ("opencl" if _OPENCL_OK else "cpu-fallback"),
    })
    return jsonify(data)

@app.route("/api/test/results")
def api_test_results():
    data = [asdict(r) for r in reversed(RESULTS)]
    return jsonify({"results": data})

@app.route("/api/test/export/<int:rid>")
def api_test_export(rid: int):
    res = next((r for r in RESULTS if r.id == rid), None)
    if not res:
        return jsonify({"ok": False, "error": "not_found"}), 404
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id","profile","backend","started_at","finished_at","duration_sec",
                     "max_temp","avg_temp","max_load","avg_load","max_power","avg_power","status"])
    writer.writerow([
        res.id, res.profile, res.backend, res.started_at, res.finished_at, res.duration_sec,
        res.max_temp, res.avg_temp, res.max_load, res.avg_load, res.max_power, res.avg_power, res.status
    ])
    csv_bytes = io.BytesIO(output.getvalue().encode("utf-8"))
    filename = f"gpubench_result_{res.id}.csv"
    return send_file(csv_bytes, mimetype="text/csv", as_attachment=True, download_name=filename)

@app.route("/api/test/export/latest")
def api_test_export_latest():
    if not RESULTS:
        return jsonify({"ok": False, "error": "no_results"}), 404
    return api_test_export(RESULTS[-1].id)

@app.route("/api/debug")
def api_debug():
    cupy_info = None
    if _CUPY_OK:
        try:
            import cupy as cp
            dev = cp.cuda.Device(0)
            name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
            free, total = cp.cuda.runtime.memGetInfo()
            cupy_info = {
                "device_index": int(dev.id),
                "device_name": name,
                "mem_free_GB": round(free / (1024**3), 2),
                "mem_total_GB": round(total / (1024**3), 2),
            }
        except Exception as e:
            cupy_info = {"error": str(e)}
    return jsonify({
        "nvml_ok": _NV_OK,
        "nvml_error": _NV_ERR,
        "env_NVML_DLL": os.environ.get("NVML_DLL"),
        "nvidia_smi_path": _find_nvidia_smi(),
        "stress_backend_available": {"cupy": _CUPY_OK, "opencl": _OPENCL_OK},
        "cupy": cupy_info,
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
