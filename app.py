from __future__ import annotations

import threading
import time
import random
from collections import deque
from typing import Optional, Tuple

from flask import Flask, render_template, jsonify, request

# ====== NVML (метрики) ======
try:
    import pynvml
    pynvml.nvmlInit()
    _NV_OK = True
    _NV_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    _NV_OK = False
    _NV_HANDLE = None

# ====== Попытка подключить GPU-стресс через CuPy или PyOpenCL ======
_CUPY_OK = False
_OPENCL_OK = False
try:
    import cupy as cp  # type: ignore
    _CUPY_OK = True
except Exception:
    pass

try:
    import pyopencl as cl  # type: ignore
    import pyopencl.array as cl_array  # type: ignore
    _OPENCL_OK = True
except Exception:
    pass

app = Flask(__name__)

# ====== Глобальное состояние ======
class State:
    def __init__(self):
        self.running: bool = False
        self.started_at: Optional[float] = None
        self.profile: Optional[str] = None
        self.lock = threading.Lock()

STATE = State()

# Хранилище истории для графиков (последние 600 точек ~ 10 минут при 1с шаге)
hist_temperature = deque(maxlen=600)
hist_load = deque(maxlen=600)
hist_labels = deque(maxlen=600)

# ====== Сборщик метрик (фон) ======
def read_gpu_metrics() -> dict:
    """Читает метрики GPU через NVML. Если NVML недоступен — генерирует правдоподобные мок-значения."""
    if _NV_OK:
        try:
            name = pynvml.nvmlDeviceGetName(_NV_HANDLE)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            driver = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(driver, bytes):
                driver = driver.decode("utf-8")

            temp = int(pynvml.nvmlDeviceGetTemperature(_NV_HANDLE, pynvml.NVML_TEMPERATURE_GPU))
            util = int(pynvml.nvmlDeviceGetUtilizationRates(_NV_HANDLE).gpu)
            mem = pynvml.nvmlDeviceGetMemoryInfo(_NV_HANDLE)
            mem_used_mb = int(mem.used / 1024 / 1024)
            mem_total_mb = int(mem.total / 1024 / 1024)
            fan = int(getattr(pynvml, "nvmlDeviceGetFanSpeed", lambda *_: 0)(_NV_HANDLE))
            power = int(getattr(pynvml, "nvmlDeviceGetPowerUsage", lambda *_: 0)(_NV_HANDLE) / 1000)
            return {
                "name": name,
                "driver": driver,
                "temperature": temp,
                "load": util,
                "mem_used_mb": mem_used_mb,
                "mem_total_mb": mem_total_mb,
                "fan_percent": fan,
                "power_w": power,
            }
        except Exception:
            # провал NVML → мок ниже
            pass

    # Моки (для дев-окружений без NVIDIA)
    temp = random.randint(45, 75)
    util = random.randint(5, 95)
    mem_used_mb, mem_total_mb = 6200, 12288
    fan = random.randint(20, 70)
    power = random.randint(100, 350)
    return {
        "name": "GPU (mock)",
        "driver": "n/a",
        "temperature": temp,
        "load": util,
        "mem_used_mb": mem_used_mb,
        "mem_total_mb": mem_total_mb,
        "fan_percent": fan,
        "power_w": power,
    }

def metrics_sampler():
    """Каждую секунду читает метрики и добавляет в историю для графиков."""
    i = 0
    while True:
        m = read_gpu_metrics()
        hist_temperature.append(m["temperature"])
        hist_load.append(m["load"])
        hist_labels.append(time.strftime("%H:%M:%S"))
        i += 1
        time.sleep(1)

threading.Thread(target=metrics_sampler, daemon=True).start()

# ====== Стресс-воркер (фон) ======
class StressWorker:
    def __init__(self):
        self._th: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self, profile: str):
        """Запускает нагрузку в фоне. Профили: quick/full/thermal/memory."""
        if self.is_running():
            return
        self._stop.clear()
        self._th = threading.Thread(target=self._run, args=(profile,), daemon=True)
        self._th.start()

    def stop(self):
        self._stop.set()

    def is_running(self) -> bool:
        return self._th is not None and self._th.is_alive() and not self._stop.is_set()

    # ---- реализация нагрузки ----
    def _run(self, profile: str):
        # Настройка размеров под профиль
        if profile == "quick":
            dur_sec = 10 * 60
            size = 2048
        elif profile == "full":
            dur_sec = 30 * 60
            size = 4096
        elif profile == "thermal":
            dur_sec = 60 * 60
            size = 4096
        elif profile == "memory":
            dur_sec = 45 * 60
            size = 3072
        else:
            dur_sec = 10 * 60
            size = 2048

        deadline = time.time() + dur_sec

        # Пытаемся грузить GPU через CuPy → PyOpenCL → иначе CPU fallback
        if _CUPY_OK:
            self._burn_cupy(size, deadline)
        elif _OPENCL_OK:
            self._burn_opencl(size, deadline)
        else:
            self._burn_cpu(size, deadline)

    def _burn_cupy(self, size: int, deadline: float):
        # матричные перемножения на GPU
        try:
            a = cp.random.random((size, size), dtype=cp.float32)
            b = cp.random.random((size, size), dtype=cp.float32)
            while not self._stop.is_set() and time.time() < deadline:
                _ = a.dot(b)  # GPU GEMM
                cp.cuda.Stream.null.synchronize()
                # небольшая пауза, чтобы дать обновиться метрикам
                time.sleep(0.01)
        except Exception:
            # fallback если что-то пошло не так
            self._burn_cpu(size, deadline)

    def _burn_opencl(self, size: int, deadline: float):
        try:
            platforms = cl.get_platforms()
            if not platforms:
                self._burn_cpu(size, deadline); return
            devs = []
            for p in platforms:
                devs.extend(p.get_devices(device_type=cl.device_type.GPU))
            if not devs:
                self._burn_cpu(size, deadline); return

            ctx = cl.Context(devs)
            queue = cl.CommandQueue(ctx)

            # простой kernel: C[i] = A[i] * B[i] + C[i]
            n = size * size
            prg = cl.Program(ctx, """
            __kernel void saxpy(__global const float* a,
                                __global const float* b,
                                __global float* c,
                                const float alpha) {
                int gid = get_global_id(0);
                c[gid] = alpha * a[gid] * b[gid] + c[gid];
            }""").build()

            a = cl_array.to_device(queue, (random.random(),) * 0)  # just to init
            # Большие буферы
            import numpy as np
            A = np.random.rand(n).astype("float32")
            B = np.random.rand(n).astype("float32")
            C = np.zeros(n, dtype="float32")

            dA = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=A)
            dB = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=B)
            dC = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=C)

            while not self._stop.is_set() and time.time() < deadline:
                prg.saxpy(queue, (n,), None, dA, dB, dC, cl.array.vec.make_float1(1.0))
                queue.finish()
        except Exception:
            self._burn_cpu(size, deadline)

    def _burn_cpu(self, size: int, deadline: float):
        # CPU-нагрузка как честный fallback, чтобы кнопки работали везде
        import numpy as np
        a = np.random.rand(size, size).astype("float32")
        b = np.random.rand(size, size).astype("float32")
        while not self._stop.is_set() and time.time() < deadline:
            _ = a @ b  # запарим одно ядро/несколько ядер — зависит от BLAS
            time.sleep(0.005)

STRESS = StressWorker()

# ====== РОУТЫ ======
@app.route("/")
def index():
    return render_template("index.html", brand="gpubench")

@app.route("/stress")
def stress():
    return render_template("stress.html", brand="gpubench")

@app.route("/api/gpu")
def api_gpu():
    return jsonify(read_gpu_metrics())

@app.route("/api/history")
def api_history():
    # последние 24 точки (или меньше, если приложение только запущено)
    labels = list(hist_labels)[-24:]
    temps = list(hist_temperature)[-24:]
    loads = list(hist_load)[-24:]
    # Если ещё не накопили — сгенерируем placeholders
    if len(labels) < 24:
        need = 24 - len(labels)
        pad_labels = [f"-{i}s" for i in range(need, 0, -1)]
        labels = pad_labels + labels
        if temps:
            temps = [temps[0]] * need + temps
            loads = [loads[0]] * need + loads
        else:
            temps = [50] * 24
            loads = [10] * 24
    return jsonify({"labels": labels, "temperature": temps, "load": loads})

@app.route("/api/test/start", methods=["POST"])
def api_test_start():
    body = request.get_json(silent=True) or {}
    profile = body.get("profile", "quick")
    with STATE.lock:
        if STATE.running:
            return jsonify({"ok": False, "error": "already_running"}), 400
        STATE.running = True
        STATE.started_at = time.time()
        STATE.profile = profile
        STRESS.start(profile)
    return jsonify({"ok": True, "running": True, "profile": profile})

@app.route("/api/test/stop", methods=["POST"])
def api_test_stop():
    with STATE.lock:
        STRESS.stop()
        STATE.running = False
        STATE.profile = None
        STATE.started_at = None
    return jsonify({"ok": True, "running": False})

@app.route("/api/test/status")
def api_test_status():
    with STATE.lock:
        running = STATE.running and STRESS.is_running()
        elapsed = int(time.time() - STATE.started_at) if (STATE.started_at and running) else 0
        if STATE.running and not running:
            # воркер завершился естественно
            STATE.running = False
            STATE.profile = None
            STATE.started_at = None
    data = read_gpu_metrics()
    data.update({"running": STATE.running, "elapsed": elapsed, "profile": STATE.profile})
    # Информация о доступности нагрузчика
    data["stress_backend"] = "cupy" if _CUPY_OK else ("opencl" if _OPENCL_OK else "cpu-fallback")
    return jsonify(data)

if __name__ == "__main__":
    # Запуск локального dev-сервера
    app.run(host="127.0.0.1", port=5000, debug=True)
