"""Compile the C simulator core into a shared library (core.so) with gcc."""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CORE_C = os.path.join(HERE, "core.c")
OUT = os.path.join(HERE, "core.so")

def _default_cc():
    for cand in (os.environ.get("CC"), "cc", "gcc", "clang"):
        if cand and shutil.which(cand):
            return cand
    raise RuntimeError("no C compiler found (need gcc/clang)")

def build(cc=None, extra=None, force=True):
    cc = cc or _default_cc()
    if os.path.exists(OUT) and not force:
        return OUT
    cmd = [
        cc, "-O3", "-fPIC", "-shared", "-std=c99",
        "-o", OUT, CORE_C, "-lm",
    ]
    if extra:
        cmd = cmd[:2] + extra + cmd[2:]  # keep flags order sane
    subprocess.run(cmd, check=True)
    if not os.path.exists(OUT):
        raise RuntimeError("build failed, no core.so produced")
    return OUT

if __name__ == "__main__":
    print("built ->", build())
