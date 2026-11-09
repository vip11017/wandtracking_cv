import subprocess

camera_args = ['/usr/bin/python', 'camera.py']
camera_process = subprocess.run(camera_args)

model_args = ['./.venv/bin/python', 'test.py']
model_process = subprocess.run(model_args)