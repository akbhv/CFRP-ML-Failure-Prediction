
import time
import statistics
import numpy as np

from api.main import ml_models, ml_scaler, ML_TARGETS

features = np.array(
    [[100000.0, 20000.0, 10000.0, 0.0, 0.0, 0.0]],
    dtype=np.float32,
)
scaled = ml_scaler.transform(features).astype(np.float32)

# Warm up both paths
for model in ml_models.values():
    model.predict(scaled, verbose=0)
    model(scaled, training=False).numpy()

def measure_predict(model, runs=50):
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        value = model.predict(scaled, verbose=0)
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times), float(value.reshape(-1)[0])

def measure_direct(model, runs=50):
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        value = model(scaled, training=False).numpy()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times), float(value.reshape(-1)[0])

print("Criterion             predict ms   direct ms   max abs difference")
print("-" * 70)

for target, name in ML_TARGETS.items():
    model = ml_models[target]
    predict_ms, predict_value = measure_predict(model)
    direct_ms, direct_value = measure_direct(model)

    print(
        f"{name:20s} "
        f"{predict_ms:10.3f} "
        f"{direct_ms:11.3f} "
        f"{abs(predict_value - direct_value):19.8g}"
    )
