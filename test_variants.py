"""
Focused wiring checks for bundled model variants.
"""
import json
import subprocess
import time
import urllib.request

import numpy as np

from e5 import E5 as RepoE5
from nanoe5 import E5 as PkgE5

HOST = "127.0.0.1"
PORT = 8141


def fetch_json(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def wait_health(timeout=15):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if fetch_json(f"http://{HOST}:{PORT}/health")["status"] == "ok":
                return True
        except Exception:
            time.sleep(0.1)
    return False


def main():
    base = RepoE5(variant="original")
    enpt = RepoE5(variant="enpt")
    text = "Como preparar arroz e feijão com alho?"
    unsupported = "机器学习是人工智能的一个分支。"

    vb = base.passage(text)
    ve = enpt.passage(text)
    assert vb.shape == (384,) and ve.shape == (384,)
    assert np.isfinite(vb).all() and np.isfinite(ve).all()
    assert abs(float(np.linalg.norm(vb)) - 1.0) < 2e-3
    assert abs(float(np.linalg.norm(ve)) - 1.0) < 2e-3
    ub = base.passage(unsupported)
    ue = enpt.passage(unsupported)
    assert not np.allclose(ub, ue, atol=1e-6), "variants should differ on pruned-language text"

    try:
        RepoE5(variant="bad")
    except ValueError:
        pass
    else:
        raise AssertionError("bad repo variant should raise ValueError")

    pkg = PkgE5(variant="enpt")
    vp = pkg.passage(text)
    assert float(vp @ ve) > 0.9999, "packaged variant should closely match repo wrapper"

    proc = subprocess.Popen(
        ["./e5", "--server", "--host", HOST, "--port", str(PORT), "--variant", "enpt"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        assert wait_health(), "embedded server failed to start"
        models = fetch_json(f"http://{HOST}:{PORT}/v1/models")
        ids = [m["id"] for m in models["data"]]
        assert ids == ["portuguese-multilingual-e5-small-q4"], ids
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    print("variant wiring OK")


if __name__ == "__main__":
    main()
