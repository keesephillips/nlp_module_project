import subprocess
import sys

script = 'make_dataset.py'
command = f'{sys.executable} scripts/{script}'
subprocess.run(command, shell=True)

script = 'build_features.py'
command = f'{sys.executable} python scripts/{script}'
subprocess.run(command, shell=True)

script = 'model.py'
command = f'{sys.executable} python scripts/{script}'
subprocess.run(command, shell=True)