import os
import re
import subprocess
import sys
from pathlib import Path

from packaging.version import Version
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    # helper function to execute cmd --version in terminal
    # and check against minimal allowed version
    def try_cmake(self, cmakecmd, min_version="3.5.0"):
        try:
            out = subprocess.check_output([cmakecmd, '--version'])
        except OSError:
            return False

        cmake_version = Version(re.search(r'version\s*([\d.]+)',
                                          out.decode()).group(1))
        if cmake_version < Version(min_version):
            return False

        return True

    # execute cmake and then cmake3 commands in the terminal
    # to find which one provides the minimum required version
    def set_cmake(self):
        if self.try_cmake("cmake"):
            self.cmake = "cmake"
        elif self.try_cmake("cmake3"):
            self.cmake = "cmake3"
        else:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

    def build_extension(self, ext: CMakeExtension) -> None:
        # find out if cmake or cmake3 provides min required version
        self.set_cmake()

        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)  # type: ignore[no-untyped-call]
        extdir = ext_fullpath.parent.resolve()

        ###
        # Set compilation flags
        ###
        # Check for compilation mode flags in the environment
        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # Set various CMAKE compilation flags, e.g. here we add Python_EXECUTABLE
        # required for PYBIND11_FINDPYTHON discovry
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DPYBIND11_FINDPYTHON=ON",
        ]

        # Add the setuptools-scm version to CMAKE flags
        cmake_args += [f"-DSDISTVERSION={self.distribution.get_version()}"]  # type: ignore[attr-defined]

        # Add additional CMake arguments found in environment, if any
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        # Cross-compile support for macOS - respect ARCHFLAGS if set
        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        # Check if we have GPU support.
        try:
            subprocess.check_output('nvidia-smi')
            cmake_args += ["-DCPU_ONLY=OFF"]
        except Exception:
            cmake_args += ["-DCPU_ONLY=ON"]
            print("WARNING: No GPU Found. Building with CPU only mode.")

        ###
        # Set build execution flags
        ###
        # Set CMAKE_BUILD_PARALLEL_LEVEL to control N threads used to build
        build_args = []
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]

        # build in temporary directories, so create them
        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        ###
        # Run the build
        ###
        subprocess.run(
            [self.cmake, ext.sourcedir] + cmake_args, cwd=build_temp, check=True
        )
        subprocess.run(
            [self.cmake, "--build", "."] + build_args, cwd=build_temp, check=True
        )


setup(
    ext_modules=[CMakeExtension("kbmod.search")],
    cmdclass={"build_ext": CMakeBuild},
    use_scm_version=True,
)
