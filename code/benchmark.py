"""Indicative CPU throughput/latency on this laptop for YOLO11 (pretrained COCO weights) and the AudioCNN.

YOLO: 640x640 predict() on a fixed image, batch 1, 20 warm-up + 200 timed runs, PyTorch CPU.
AudioCNN: one 4-s window, batch 1: (a) model forward only, (b) log-Mel + forward.
"""
import json
import platform
import statistics as st
import time

import cpuinfo
import psutil
import torch

from common import OUT, N_SAMPLES, AudioCNN, logmel_transform


def timed(fn, warm=20, n=200):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    return {"mean_ms": st.mean(ts), "median_ms": st.median(ts), "p95_ms": ts[int(0.95 * len(ts)) - 1],
            "fps_from_mean": 1000 / st.mean(ts), "runs": n}


def main():
    hw = {"cpu": cpuinfo.get_cpu_info().get("brand_raw"), "physical_cores": psutil.cpu_count(False),
          "logical_cores": psutil.cpu_count(True), "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
          "gpu": "Intel integrated graphics only (no CUDA GPU); all timings on CPU",
          "os": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
          "torch_threads": torch.get_num_threads(),
          "on_ac_power": getattr(psutil.sensors_battery(), "power_plugged", None)}
    res = {"hardware": hw, "yolo11": {}, "audiocnn": {}}

    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils import ASSETS
    hw["ultralytics"] = ultralytics.__version__
    img = str(ASSETS / "bus.jpg")
    for v in ("yolo11n", "yolo11s", "yolo11m"):
        m = YOLO(f"{v}.pt")
        n_params = sum(p.numel() for p in m.model.parameters())
        r = timed(lambda: m.predict(img, imgsz=640, device="cpu", verbose=False), warm=10, n=100)
        res["yolo11"][v] = {"params_M": round(n_params / 1e6, 2), **r}
        print(v, res["yolo11"][v], flush=True)

    model = AudioCNN().eval()
    ck = OUT / "seed0" / "model.pt"
    if ck.exists():
        model.load_state_dict(torch.load(ck)["state"])
    to_mel = logmel_transform()
    wav = torch.randn(N_SAMPLES) * 0.1
    x = to_mel(wav).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        res["audiocnn"]["params_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 3)
        res["audiocnn"]["forward_only"] = timed(lambda: model(x))
        res["audiocnn"]["logmel_plus_forward"] = timed(lambda: model(to_mel(wav).unsqueeze(0).unsqueeze(0)))
    print(json.dumps(res, indent=2))
    json.dump(res, open(OUT / "benchmark_this_laptop.json", "w"), indent=2)


if __name__ == "__main__":
    main()
