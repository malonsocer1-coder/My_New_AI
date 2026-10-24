"""Diagnóstico del entorno: ``titan doctor``.

Comprueba lo necesario para entrenar y ejecutar TITAN en este equipo y explica
qué falta. No envía datos a ningún servicio (solo comprueba la conectividad
con Hugging Face y PyPI mediante una petición HEAD).
"""

from __future__ import annotations

import importlib
import platform
import shutil
import sys
import urllib.request
from pathlib import Path


def _ok(flag: bool) -> str:
    return "OK " if flag else "-- "


def _reachable(url: str) -> bool:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=8):
            return True
    except Exception:
        return False


def run_doctor(path: str = ".") -> dict:
    import torch
    rep: dict = {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__}
    cuda = torch.cuda.is_available()
    rep["cuda"] = cuda
    if cuda:
        p = torch.cuda.get_device_properties(0)
        rep.update({"gpu": p.name, "vram_gb": round(p.total_memory / 1e9, 1), "bf16": torch.cuda.is_bf16_supported(),
                    "cuda_version": torch.version.cuda, "compute_capability": f"{p.major}.{p.minor}"})
    try:
        import psutil
        rep["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
        rep["cpu_cores"] = psutil.cpu_count(logical=False)
    except ImportError:
        pass
    rep["disk_free_gb"] = round(shutil.disk_usage(Path(path).resolve()).free / 1e9, 1)
    for mod in ("tokenizers", "safetensors", "datasets", "gguf", "llama_cpp", "transformers", "pypdf", "openpyxl"):
        try:
            m = importlib.import_module(mod)
            rep[f"pkg_{mod}"] = getattr(m, "__version__", "instalado")
        except Exception:
            rep[f"pkg_{mod}"] = None
    if rep.get("pkg_llama_cpp"):
        import llama_cpp
        rep["llama_cpp_gpu_offload"] = bool(llama_cpp.llama_supports_gpu_offload())
    rep["net_huggingface"] = _reachable("https://huggingface.co")
    rep["net_pypi"] = _reachable("https://pypi.org/simple/")
    return rep


def print_doctor(rep: dict) -> None:
    print(f"Python {rep['python']} · {rep['platform']} · PyTorch {rep['torch']}")
    if rep["cuda"]:
        print(f"{_ok(True)}GPU: {rep['gpu']} ({rep['vram_gb']} GB, CC {rep['compute_capability']}, CUDA {rep['cuda_version']})")
        print(f"{_ok(rep['bf16'])}bf16 {'disponible' if rep['bf16'] else 'no disponible: use dtype fp16'}")
    else:
        print(f"{_ok(False)}Sin GPU CUDA: el entrenamiento será muy lento (solo validación). En Windows reinstale "
              "PyTorch con CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu128")
    if "ram_gb" in rep:
        print(f"{_ok(rep['ram_gb'] >= 15)}RAM: {rep['ram_gb']} GB · núcleos físicos: {rep.get('cpu_cores')}")
    print(f"{_ok(rep['disk_free_gb'] >= 50)}Disco libre: {rep['disk_free_gb']} GB "
          "(datos del PC ~35 GB; checkpoints ~1,2 GB cada uno)")
    print(f"{_ok(bool(rep['pkg_datasets']))}datasets (necesario para descargar datos de Hugging Face)")
    print(f"{_ok(rep['net_huggingface'])}Acceso a huggingface.co")
    print(f"{_ok(bool(rep['pkg_gguf']))}gguf (exportar para llama.cpp)")
    if rep["pkg_llama_cpp"]:
        print(f"{_ok(rep.get('llama_cpp_gpu_offload', False))}llama-cpp-python "
              + ("con GPU" if rep.get("llama_cpp_gpu_offload") else "SIN GPU: recompile con CMAKE_ARGS=-DGGML_CUDA=on"))
    else:
        print(f"{_ok(False)}llama-cpp-python no instalado (inferencia GGUF rápida en la GPU)")
    if rep["cuda"] and rep.get("vram_gb", 0) < 10:
        print("   Nota: con < 10 GB de VRAM el máximo práctico para entrenar es ~350 M parámetros "
              "(ver docs/PC_ENTRENAMIENTO_POR_PARTES.md)")
