"""
Build the nanoe5 C engine as a bundled shared object (loaded via ctypes).

The engine source (``e5.c``/``e5.h``) and the 4-bit model (``e5-small-q4.bin``)
live inside the ``nanoe5/`` package, so the sdist/wheel is fully self-contained.

ISA selection (via the ``NANOE5_ARCH`` env var):
  * unset / ``native``  -> ``-march=native`` (ideal for source installs)
  * ``avx2``            -> ``-mavx2 -mfma`` (portable, fast x86-64 wheels)
  * ``portable``        -> baseline scalar (auto-vectorized; never SIGILLs)
CI builds distributable wheels with ``avx2`` (x86-64) or ``portable`` (arm64).

Because the engine uses no CPython API (it is loaded via ctypes), the wheel is
tagged ``py3-none-<platform>`` — one wheel per platform works on every Python 3.
"""
import os
import shutil
import subprocess
import tempfile

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

try:                                            # setuptools >= 70 vendors this
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:                             # older toolchains
    from wheel.bdist_wheel import bdist_wheel

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "nanoe5")

# Keep the package self-contained: refresh the engine sources / model from the
# repo root when present (dev builds); on install-from-sdist they already exist.
for fn in ("e5.c", "e5.h", "e5-small-q4.bin", "sae.bin"):   # sae.bin is optional
    src, dst = os.path.join(HERE, fn), os.path.join(PKG, fn)
    if os.path.exists(src) and (not os.path.exists(dst) or
                                os.path.getmtime(src) > os.path.getmtime(dst)):
        shutil.copy2(src, dst)


def _compiles(cc, flags):
    """Return True if `cc` accepts `flags` on a trivial program."""
    d = tempfile.mkdtemp()
    try:
        c = os.path.join(d, "t.c")
        with open(c, "w") as f:
            f.write("int main(void){return 0;}\n")
        return subprocess.call([cc, *flags, c, "-o", os.path.join(d, "t")],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
    except Exception:
        return False
    finally:
        shutil.rmtree(d, ignore_errors=True)


class BuildEngine(build_ext):
    def build_extensions(self):
        cc = os.environ.get("CC") or self.compiler.compiler[0]
        cflags = ["-O3", "-funroll-loops", "-ffast-math"]
        lflags = ["-lm"]                       # libm / libmvec (vectorized erff)
        if _compiles(cc, ["-fopenmp"]):        # OpenMP optional (pragmas ignored without it)
            cflags.append("-fopenmp"); lflags.append("-fopenmp")

        arch = os.environ.get("NANOE5_ARCH", "native").lower()
        if arch == "avx2" and _compiles(cc, ["-mavx2", "-mfma"]):
            cflags += ["-mavx2", "-mfma"]
        elif arch == "portable":
            pass                               # baseline scalar (auto-vectorized)
        else:                                  # "native" / default
            if _compiles(cc, ["-march=native"]):
                cflags.append("-march=native")
            elif _compiles(cc, ["-mavx2", "-mfma"]):
                cflags += ["-mavx2", "-mfma"]

        for ext in self.extensions:
            ext.extra_compile_args = cflags + ext.extra_compile_args
            ext.extra_link_args = lflags + ext.extra_link_args
        super().build_extensions()


class GenericWheel(bdist_wheel):
    """Emit a python-agnostic platform wheel (py3-none-<platform>): the engine
    is a ctypes-loaded shared lib with no CPython ABI, so one wheel serves every
    Python 3 on a given platform."""
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False              # impure -> keep the platform tag

    def get_tag(self):
        _python, _abi, plat = super().get_tag()
        return "py3", "none", plat


engine = Extension(
    "nanoe5._engine",
    sources=["nanoe5/e5.c"],
    include_dirs=["nanoe5"],
    depends=["nanoe5/e5.h"],
)

setup(ext_modules=[engine], cmdclass={"build_ext": BuildEngine, "bdist_wheel": GenericWheel})
