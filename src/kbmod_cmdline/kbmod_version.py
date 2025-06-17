"""Display basic version and build information to standard out."""

import kbmod
from kbmod.search import kb_has_gpu, HAS_OMP


def main():
    print(f"KBMOD\n-----")
    print(f"Version: {kbmod.__version__}")
    print(f"GPU Code Enabled: {kb_has_gpu()}")
    print(f"OpenMP Enabled: {HAS_OMP}")
