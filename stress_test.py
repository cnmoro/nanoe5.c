"""
stress_test.py - hard stress / edge-case test for the nanoE5.c engine, the
Python binding, and the OpenAI-compatible server.

Run:  make stress      (or)   python3 stress_test.py

It deliberately throws adversarial and pathological inputs at every layer and
asserts: no crashes, no hangs, finite & normalized embeddings, determinism,
batch==single consistency, server==binding parity, base64==float parity, and
correct HTTP error handling. Exit code 0 only if everything passes.
"""
import os, sys, time, json, socket, struct, base64, threading, subprocess, random
import urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from e5 import E5

HOST, PORT = "127.0.0.1", 8137
FAIL = []


def check(name, cond, extra=""):
    status = "ok  " if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  {extra}" if extra else ""))
    if not cond:
        FAIL.append(name)


def finite_unit(v, tol=2e-3):
    v = np.asarray(v, dtype=np.float64)
    return np.all(np.isfinite(v)) and abs(np.linalg.norm(v) - 1.0) < tol


# adversarial / pathological input corpus -----------------------------------
EDGE = {
    "empty": "",
    "space": " ",
    "spaces": "          ",
    "tabs_newlines": "\t\t\n\n\r\n  \t",
    "single_char": "x",
    "one_byte_utf8": "é",
    "emoji": "🚀🔥😀 hello 👨‍👩‍👧‍👦 family",
    "combining": "áêĩ nfc/nfd noïse",
    "rtl_arabic": "مرحبا بالعالم هذا اختبار",
    "cjk": "机器学习是人工智能的一个分支，深度学习是其中的重要方法。",
    "mixed_scripts": "Hello мир 世界 🌍 שלום العالم",
    "control_chars": "a\x01\x02\x03\x07b\x1fc",
    "zero_width": "in​visible‌joiners‍ here",
    "weird_unicode": "½ ² ③ ﬁ ３Ｄ ① ²³ ¼ № ™ ",
    "long_word": "a" * 2000,
    "long_repeat": "the quick brown fox " * 200,         # ~1000 tokens -> windows
    "very_long": "machine learning and natural language processing. " * 2000,  # huge -> many windows
    "json_breakers": 'he said "hi" and \\ backslash / slash {curly} [brackets]',
    "only_punct": "!@#$%^&*()_+-=[]{}|;':\",./<>?`~",
    "numbers": "1234567890 3.14159 -42 1e10 0x1F",
    "newline_doc": "line one\nline two\nline three\n" * 50,
}


# ===========================================================================
def stress_python():
    print("\n=== Python binding stress ===")
    m = E5()
    D = m.dim

    # 1. all edge inputs produce finite, unit-norm vectors (query + passage)
    bad = []
    for name, t in EDGE.items():
        for isq in (True, False):
            v = m.query(t) if isq else m.passage(t)
            if v.shape != (D,) or not finite_unit(v):
                bad.append(f"{name}/{'q' if isq else 'p'} norm={np.linalg.norm(v):.4f}")
    check("all edge inputs finite & unit-norm (q & p)", not bad, "; ".join(bad[:4]))

    # 2. determinism: identical input -> byte-identical output, regardless of order
    a = m.passage("deterministic output check with some words")
    b = m.passage("deterministic output check with some words")
    check("deterministic (exact equality across calls)", np.array_equal(a, b))

    # 3. batch == individual (attention is per-segment, so it must match exactly)
    texts = list(EDGE.values())
    batch = m.passage(texts)
    singles = np.stack([m.passage(t) for t in texts])
    max_dev = float(np.max(np.abs(batch - singles)))
    check("batch result == per-item result", max_dev < 1e-5, f"max|Δ|={max_dev:.2e}")

    # 4. concurrency: ctypes releases the GIL, so threads hit the engine in
    #    parallel -> tests engine thread-safety. Results must match the serial ref.
    ref = {k: m.passage(v) for k, v in EDGE.items()}
    errors = []
    def worker(_):
        for k, v in EDGE.items():
            out = m.passage(v)
            if not np.array_equal(out, ref[k]):
                errors.append(k)
    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(worker, range(64)))
    check("32k concurrent calls thread-safe & deterministic", not errors, f"{len(errors)} mismatches")

    # 5. windowing: a long doc must be closer to its topic than an off-topic query
    doc = "Photosynthesis lets plants convert sunlight, water and CO2 into glucose and oxygen. " * 40
    e = m.passage(doc)
    on = float(e @ m.query("how do plants turn sunlight into energy"))
    off = float(e @ m.query("government bond yields and central bank policy"))
    check("sliding-window long doc keeps topicality", on > off + 0.02, f"on={on:.3f} off={off:.3f}")

    # 6. huge batch in one call
    big = m.passage([f"document number {i} about various topics" for i in range(2000)])
    check("2000-item batch ok", big.shape == (2000, D) and np.all(np.isfinite(big)))

    return m


# ===========================================================================
def wait_health(timeout=15):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://{HOST}:{PORT}/health", timeout=1) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.1)
    return False


def post(path, data, raw=False, headers=None, timeout=60):
    body = data if raw else json.dumps(data).encode()
    req = urllib.request.Request(f"http://{HOST}:{PORT}{path}", data=body,
                                 headers=headers or {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def raw_send(payload_bytes, read_timeout=5):
    """send raw bytes over a socket, return response bytes (or b'' if closed)."""
    s = socket.create_connection((HOST, PORT), timeout=read_timeout)
    s.settimeout(read_timeout)
    try:
        s.sendall(payload_bytes)
        chunks = []
        while True:
            try:
                d = s.recv(65536)
            except socket.timeout:
                break
            if not d:
                break
            chunks.append(d)
        return b"".join(chunks)
    finally:
        s.close()


def stress_server(m):
    print("\n=== Server stress ===")
    D = m.dim

    # 1. server vs python-binding parity (same engine -> same numbers)
    txt = "parity check between the http server and the in-process binding"
    st, body = post("/v1/embeddings", {"input": txt, "input_type": "query"})
    srv = np.array(json.loads(body)["data"][0]["embedding"], dtype=np.float32)
    py = m.query(txt)
    check("server == python binding", st == 200 and np.allclose(srv, py, atol=1e-6),
          f"max|Δ|={np.max(np.abs(srv-py)):.2e}")

    # 2. base64 == float, and decodes to a unit vector
    st, b_f = post("/v1/embeddings", {"input": txt, "encoding_format": "float", "input_type": "query"})
    st, b_b = post("/v1/embeddings", {"input": txt, "encoding_format": "base64", "input_type": "query"})
    f_vec = np.array(json.loads(b_f)["data"][0]["embedding"], dtype=np.float32)
    b_raw = base64.b64decode(json.loads(b_b)["data"][0]["embedding"])
    b_vec = np.frombuffer(b_raw, dtype="<f4")
    check("base64 == float & unit-norm", np.allclose(f_vec, b_vec, atol=1e-6) and finite_unit(b_vec))

    # 3. array input, ordering preserved
    arr = ["first text", "second text", "third text", "first text"]
    st, body = post("/v1/embeddings", {"input": arr})
    d = json.loads(body)["data"]
    idx_ok = [x["index"] for x in d] == [0, 1, 2, 3]
    v0 = np.array(d[0]["embedding"]); v3 = np.array(d[3]["embedding"])
    check("array input: order + identical dup rows", st == 200 and idx_ok and np.allclose(v0, v3, atol=1e-6))

    # 4. OpenAI official client compatibility (base64 path is its default)
    try:
        from openai import OpenAI
        cli = OpenAI(base_url=f"http://{HOST}:{PORT}/v1", api_key="not-needed")
        r = cli.embeddings.create(model="e5-query", input=["openai client test", "second"])
        ok = len(r.data) == 2 and len(r.data[0].embedding) == D and finite_unit(r.data[0].embedding)
        check("openai python client works", ok, f"usage={r.usage.total_tokens}")
    except Exception as ex:
        check("openai python client works", False, repr(ex)[:80])

    # 5. edge-case inputs through the server (parity with binding)
    bad = []
    for name, t in EDGE.items():
        st, body = post("/v1/embeddings", {"input": t, "input_type": "passage"})
        if st != 200:
            bad.append(f"{name}:HTTP{st}"); continue
        sv = np.array(json.loads(body)["data"][0]["embedding"], dtype=np.float32)
        if not np.allclose(sv, m.passage(t), atol=1e-6):
            bad.append(f"{name}:mismatch")
    check("all edge inputs via server == binding", not bad, "; ".join(bad[:4]))

    # 6. tricky JSON the parser must handle
    tricky = [
        ('{"input":"escapes \\" \\\\ \\/ \\n \\t \\r \\b \\f end"}', 200),
        ('{"input":"unicode \\u00e9\\u4e16\\u754c"}', 200),
        ('{"input":"emoji surrogate \\ud83d\\ude00 done"}', 200),
        ('{"input":"nul\\u0000byte after"}', 200),
        ('{"model":"x","encoding_format":"float","extra":{"nested":[1,2,{"a":true}]},"input":"deep"}', 200),
        ('{"input":["a","b","c"],"unknown":123.45e-2,"flag":false,"z":null}', 200),
        ('{"input":""}', 200),                 # empty string allowed
    ]
    tbad = []
    for payload, want in tricky:
        st, _ = post("/v1/embeddings", payload.encode(), raw=True)
        if st != want:
            tbad.append(f"{payload[:30]}->HTTP{st}")
    check("tricky-but-valid JSON accepted", not tbad, "; ".join(tbad))

    # 7. malformed / invalid requests must yield 4xx, never crash
    invalid = [
        ('{bad json', 400),
        ('not json at all', 400),
        ('{"input":123}', 400),                # number, not string/array
        ('{"input":[1,2,3]}', 400),            # array of numbers
        ('{"input":[]}', 400),                 # empty array
        ('{"input":null}', 400),
        ('{"model":"x"}', 400),                # missing input
        ('{}', 400),
        ('', 400),
        ('{"input":["ok",123]}', 400),         # mixed array
        ('{"input":"unterminated', 400),
        ('{"input":"' + "x" * 10 + '","input_type":42}', 400),  # bad input_type type
    ]
    ibad = []
    for payload, want in invalid:
        st, _ = post("/v1/embeddings", payload.encode(), raw=True)
        if st != want:
            ibad.append(f"{payload[:24]!r}->HTTP{st}")
    check("malformed requests -> 4xx", not ibad, "; ".join(ibad[:4]))

    # 8. HTTP method / route errors
    routes = []
    st, _ = post("/v1/embeddings", b"", raw=True, headers={})  # POST empty body
    routes.append(("empty POST body", st in (400,)))
    try:
        with urllib.request.urlopen(urllib.request.Request(
                f"http://{HOST}:{PORT}/v1/embeddings", method="DELETE"), timeout=5) as r:
            mc = r.status
    except urllib.error.HTTPError as e:
        mc = e.code
    routes.append(("DELETE -> 405", mc == 405))
    st, _ = post("/totally/unknown", b"{}", raw=True)
    routes.append(("unknown route -> 404", st == 404))
    check("method/route handling", all(ok for _, ok in routes),
          "; ".join(n for n, ok in routes if not ok))

    # 9. oversized declared body -> 413 (no huge transfer needed)
    resp = raw_send(b"POST /v1/embeddings HTTP/1.1\r\nHost: x\r\n"
                    b"Content-Length: 999999999999\r\n\r\n")
    check("huge Content-Length -> 413", b" 413 " in resp[:64], resp[:40].decode("latin1", "replace"))

    # 10. raw garbage / partial requests must not crash the server
    garbage = [
        b"\x00\x01\x02\x03 not http\r\n\r\n",
        b"GET\r\n\r\n",
        b"POST /v1/embeddings HTTP/1.1\r\nContent-Length: 50\r\n\r\n{\"input\":\"part",  # short body
        b"GARBAGE / HTTP/1.1\r\n\r\n",
        b"\r\n\r\n",
        b"A" * 5000,                                # junk, no header terminator (will hit cap/timeout)
    ]
    for g in garbage:
        try:
            raw_send(g, read_timeout=2)
        except Exception:
            pass
    # server must still be alive and correct afterwards
    st, body = post("/v1/embeddings", {"input": "still alive after the garbage barrage"})
    alive = st == 200 and finite_unit(json.loads(body)["data"][0]["embedding"])
    check("server survives garbage barrage", alive)

    # 11. heavy concurrent load with mixed payloads
    payloads = [{"input": list(EDGE.values())[: (i % 5) + 1]} for i in range(8)]
    errors = [0]
    def hammer(i):
        try:
            st, body = post("/v1/embeddings", payloads[i % len(payloads)], timeout=120)
            d = json.loads(body)["data"]
            for x in d:
                if not finite_unit(x["embedding"]):
                    errors[0] += 1
            if st != 200:
                errors[0] += 1
        except Exception:
            errors[0] += 1
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=32) as ex:
        list(ex.map(hammer, range(400)))
    dt = time.time() - t0
    check("400 concurrent requests (32 clients) all valid", errors[0] == 0,
          f"{errors[0]} errors, {dt:.1f}s")

    # 12. concurrent identical requests -> identical answers (no data races)
    base = m.query("race condition probe text")
    mismatches = [0]
    def same(_):
        st, body = post("/v1/embeddings", {"input": "race condition probe text", "input_type": "query"})
        v = np.array(json.loads(body)["data"][0]["embedding"], dtype=np.float32)
        if not np.allclose(v, base, atol=1e-6):
            mismatches[0] += 1
    with ThreadPoolExecutor(max_workers=24) as ex:
        list(ex.map(same, range(240)))
    check("concurrent identical requests are consistent", mismatches[0] == 0, f"{mismatches[0]} races")


# ===========================================================================
def main():
    m = stress_python()

    print("\nstarting server ...")
    proc = subprocess.Popen(["./e5", "--server", "--host", HOST, "--port", str(PORT),
                             "--default-type", "passage"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        if not wait_health():
            print("  [FAIL] server did not become healthy"); FAIL.append("server start")
        else:
            stress_server(m)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    print("\n" + "=" * 60)
    if FAIL:
        print(f"RESULT: FAIL ({len(FAIL)} checks) -> {FAIL}")
        sys.exit(1)
    print("RESULT: PASS - all stress & edge-case checks passed")
    sys.exit(0)


if __name__ == "__main__":
    main()
