import glob
from pathlib import Path

patterns = [
    "pyccel/**/*.py",
    "pyccel/**/*.pyi",
    "pyccel/parser/grammar/*.tx",
    "pyccel/stdlib/**/*.h",
    "pyccel/stdlib/**/*.c",
    "pyccel/stdlib/**/*.f90",
    "pyccel/stdlib/**/*.inc",
    "pyccel/stdlib/**/CMakeLists.txt",
    "pyccel/stdlib/**/meson.build",
    "pyccel/extensions/STC/docs/**/*",
    "pyccel/extensions/STC/include/**/*",
    "pyccel/extensions/STC/src/**/*",
    "pyccel/extensions/STC/tests/**/*",
    "pyccel/extensions/STC/examples/**/*",
    "pyccel/extensions/STC/Makefile",
    "pyccel/extensions/STC/meson.build",
    "pyccel/extensions/STC/meson_options.txt",
]

files = []
for pattern in patterns:
    files.extend(Path('.').rglob(pattern))

inv_patterns = [
    "pyccel/extensions/**/*.py",
    "**/__pyccel__/**/*",
]

for pattern in inv_patterns:
    for p in Path('.').rglob(pattern):
        try:
            files.remove(p)
        except ValueError:
            pass

print("\n".join(str(f) for f in files if f.is_file()))

