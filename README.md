### Plate reader

#### A tiny detection net trained with the reference trainer, then ported to torch so inference stays cheap
 
### Where it runs

Written on Arch, expected to behave on Windows and macOS too

<img src="/screens/env.jpg" style="float: center; margin-right: 10px;" width="1000"/>

### In action

<img src="/screens/sc1.png" style="float: center; margin-right: 10px;" width="1000"/><img src="/screens/sc2.png" style="float: center; margin-right: 10px;" width="1000"/>

### Getting it running

#### Torch

#### Packages first

Shared bits
```bash
conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses  # paket dasar
```

Linux adds
```bash
# LAPACK on the GPU
conda install -c pytorch magma-cuda102  # swap for magma-cuda101 / 100 / 92 to match your toolkit
```

Once that finishes, move into the clone from the previous step and install the python bindings. The runtime_data folder in this repo is what the next command expects.

```bash
cd plate/src/bindings/python/
sudo python3 pasang.py install
```

#### Driving it

```bash
python jalankan.py -berkas-konfigurasi <path to cfg file>\
               -berkasbobot <lokasi berkas bobot>\
               -berkaslabel <lokasi berkas label kelas>\
               -keluaran <nama video hasil>\
               -sumber <lokasi video, atau indeks kamera bila memakai kamera>
```

#### A star helps

If this saved you an afternoon, leave a star on the repository.

#### Trimming the cost

The recognition stage can be tuned for speed.

Override the system-wide settings in the local config file:
```python
; local overrides land here, the shipped defaults stay untouched
akselerasi_perangkat = 1
gpu_id = 0
gpu_batch_size = 8
```
Then relax the detector defaults:
```python
max_detection_input_width = 768
max_detection_input_height = 576

pendeteksi = lbpgpu

skip_detection = 1

sudut_maks_pelat = 30
```
Resolution lives in the network config:

```bash

[net]
# evaluation
#batch=1
#subdivisions=1
# fitting
batch=48
subdivisions=12
width=576      # smaller width and height trade
height=576     # accuracy for throughput
channels=3     # larger values trade the other way
momentum=0.85
decay=0.0007
angle=0
saturation = 1.35
exposure = 1.35
hue=.15

```

#### Throughput

- On a Tesla T4, recognition stage excluded

| input side | frames/s |
| ---------- | -------- |
| 832 | 51 |
| 704 | 61 |
| 576 | 79 |
| 448 | 95 |
| 384 | 104 |
| 288 | 118 |

Building the recognition library on a hosted notebook did not work out; switching it on costs about eleven frames per second.

A numba pass shaves a little off the inference loop. On a low-power Pentium it cut the merge step from roughly 265 down to 168 on average. Anyone comfortable with numba is welcome to push it further — there is no GPU on hand to test with.

#### Still open

1. Writing the cropped plate images to disk

#### Built on

[**the detector trainer**](https://github.com/example/trainer-repo)
A well known C implementation of the detector, for Windows and Linux

[**torch port of the detector**](https://github.com/example/port-repo)
A torch, ONNX and TensorRT implementation of the same network

[**the tensor library**](https://pytorch.org/)
Tensors and dynamic graphs in Python with GPU acceleration

[**the vision toolkit**](https://opencv.org/)
An open source computer vision library

[**the recognition library**](https://github.com/example/plate-repo)
A plate recognition library