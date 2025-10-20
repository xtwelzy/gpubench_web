#gpubench — GPU Stress Testing & Monitoring Suite

**gpubench** — это локальное Flask-веб-приложение для мониторинга и стресс-тестирования видеокарт (GPU) с визуализацией температуры, загрузки, потребления энергии и памяти.  
Поддерживает **NVML**, **nvidia-smi**, **CuPy (CUDA)**, а также резерв через **OpenCL** или CPU-эмуляцию.

---

##Возможности

- **Онлайн-мониторинг** температуры, загрузки и мощности GPU  
- **Стресс-тесты** (Quick, Full, Thermal, Memory) с выбором профиля
- **Живые графики** с обновлением каждую секунду (Chart.js)
- **Экспорт CSV-отчётов** после завершения теста
- **NVML / nvidia-smi / CuPy backend** для точных данных
- **Резервный CPU-режим**, если CUDA недоступна
- Полностью **локальная работа** — без интернета
- Современный дизайн на **Tailwind CSS**

---

##Технологии

| Компонент | Назначение |
|------------|------------|
| **Flask** | Веб-сервер и API |
| **Chart.js** | Графики и визуализация |
| **CuPy** | GPU-нагрузка (через CUDA) |
| **NVML / nvidia-ml-py3** | Метрики NVIDIA |
| **Tailwind CSS** | UI-оформление |
| **Python ≥ 3.9** | Основной язык |

---

##Установка
```
### 1️⃣ Клонируй проект
git clone https://github.com/xtwelzy/gpubench.git
cd gpubench
2️⃣ Создай виртуальное окружение
python -m venv .venv
.venv\Scripts\activate   # Windows
3️⃣ Установи зависимости
pip install -r requirements.txt
(Выбери нужную версию CuPy под свою CUDA — см. комментарии в requirements.txt.)

▶️ Запуск
python app.py
Открой браузер и перейди по адресу:
👉 http://127.0.0.1:5000

Структура проекта
gpubench/
│
├── app.py                  # Основной Flask-сервер
├── requirements.txt        # Зависимости
│
├── templates/
│   ├── index.html          # Панель мониторинга
│   └── stress.html         # Стресс-тестирование

Результаты тестов
После завершения стресс-теста результаты доступны в браузере:

/api/test/results
или напрямую скачиваются CSV-файлом:

/api/test/export/latest
CSV содержит поля:

id, profile, backend, started_at, finished_at,
duration_sec, max_temp, avg_temp, max_load,
avg_load, max_power, avg_power, status

Требования к системе
Windows 10/11 x64 (тестировалось)

NVIDIA GPU с установленным драйвером и nvml.dll

Python 3.9+

Установленная CUDA (для CuPy)

Поддержка
Если NVML не определяется:

проверь путь C:\Windows\System32\nvml.dll

убедись, что nvidia-smi.exe доступен в PATH

можно вручную задать путь:

os.environ["NVML_DLL"] = r"C:\Windows\System32\nvml.dll"

🧾 Лицензия
MIT License © 2025 [xtwelzy]

🟢 gpubench создан для локальных нагрузочных тестов и визуального анализа.
Используй с осторожностью — тесты могут сильно нагружать GPU.
