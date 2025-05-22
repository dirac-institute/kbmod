"""Display basic version and build information to standard out."""

import kbmod
from kbmod.search import HAS_GPU, HAS_OMP


def main():
    print(f"KBMOD\m-----n")
    print(f"Version: {kbmod.__version__}")
    print(f"GPU Code Enabled: {HAS_GPU}")
    print(f"OpenMP Enabled: {HAS_OMP}")
