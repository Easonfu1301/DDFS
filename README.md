# Detector Design & Fast Simulation(DDFS)

## Introduction

This project aim at developing a fast simulation tool for the detector design in high energy physics.

## Installation

Simply install from pip **(NEED TO BE UPLOADED TO PIP YET)**

```bash
pip install DDFS
```

You can also download the source code from the github repository and install it manually.

```bash 
pip install dist\DDFS-1.0.0.tar.gz
```

## Usage

One can import the package by

```python
import DDFS
```

and use the functions provided in the package.

## Documentation

### 1. Detector Design

#### 1.1 Detector

This project use the `Detector` class to represent the detector. The `Detector` class has the following attributes:

- `Silayer`: The Detector is composed by a number of `Silayer` objects. Each `Silayer` object represents a layer of the
  detector. The `Silayer` class has the following attributes:
    - `material_budget`: 散射强度（基本正比于一个层的厚度）
    - `radius`: Layer 所在半径位置
    - `half_z`: Layer 的半长度
    - `efficiency`: 探测效率
    - `loc0`: r-phi方向分辨能力
    - `loc1`: z 方向分辨能力

#### 1.2 Detector Design

The Detector can be designed by both .csv file

```python
import matplotlib.pyplot as plt
from DDFS.element import Detector

file_path = 'detector.csv'
dec = Detector()

dec.load_designed(file_path)

dec.visualize_detector()
plt.show()
```

or by simply setting the parameters for each layer

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DDFS.element import Detector, SiLayer

d = Detector()
d.add_layer(SiLayer(0.0015, 10, 4000, 0, 9.9, 9.9))
for i in np.linspace(20, 1020, 10):
    d.add_layer(SiLayer(0.002, i, 4000, 1, 0.004, 0.004))

d.visualize_detector()
plt.show()
```

Easily update the parameters of the detector and store the new design to a .csv file

```python
from DDFS.element import Detector

dec = Detector()
dec.load_designed('detector.csv')
dec.update_layer(0, "Radius", 15)  # Note that the order will refresh after each update
dec.store_designed('detector_new.csv')
```

### 2. Environment and Particle setup

#### 2.1 Environment

The `Environment` class is used to represent the environment of the detector, Including

- `B`: The magnetic field of the detector
- `multiple_scattering`: 是否开启多重散射
- `position_resolution`: 是否开启位置精度?(Only work in Analytic estimate)

One can build the environment by

```python
from DDFS.element import Environment

envir = Environment()

envir.update_environment("B", 3)
envir.update_environment("multiple_scattering", True)
envir.update_environment("position_resolution", True)

print(envir)
```

### 3. Particle and Emitter setup

#### 3.1 Particle

The `Particle` class is used to represent the particle that will pass through the detector. Including

- `emit_mode`: 发射参数
- `charge`: 电荷
- `mass`: 质量

The emit_mode MUST be like this:

```python
import numpy as np

para_p = {
    "type": "steps",
    "maxvalue": 120,
    "minvalue": 2,
    "steps": 10,
    "count": 0
}

para_t = {
    "type": "fixed",
    "value": 0.3
}

para_phi = {
    "type": "even",
    "minvalue": -np.pi,
    "maxvalue": np.pi,
}

emit_mode = {
    "p": para_p,
    "theta": para_t,
    "phi": para_phi
}
```

There are three types of emit_mode: `steps`, `fixed`, `even`. The `steps` type will generate a random number
between `minvalue` and `maxvalue` with `steps` steps. The `fixed` type will generate a fixed value. The `even` type will
generate a random number between `minvalue` and `maxvalue`.

#### 3.2 Emitter

Once you setup the `Particle` class, you can use the `Emitter` class to emit the particle. it is recommended to use
the `deepcopy` function to avoid the reference problem.

```python
from copy import deepcopy
from DDFS.element import Particle, Emitter

m = Emitter()
m.add_particle(p, 1, deepcopy(emit_mode))  # p, possibility of particles, emit_mode
```

### 3. Fast Simulation
#### 3.1 Analytical estimate

You can use the Analytic_method to estimate the Resolution.

```python
from DDFS.element import *
import DDFS.analytic_method as am
from copy import deepcopy

d = Detector()
e = Environment()
e.update_environment("B", 3)
e.update_environment("position_resolution", True)
e.update_environment("multiple_scattering", True)
m = Emitter()

p = Particle()
p.update_particle("Charge", -1)
p.update_particle("Mass", 0.106)

m.add_particle(p, 1, deepcopy(emit_mode))
d.add_layer(SiLayer(0.0015, 10, 4000, 0, 9.9, 9.9))
for i in np.linspace(20, 1020, 10):
    d.add_layer(SiLayer(0.002, i, 4000, 1, 0.004, 0.004))

# d.visualize_detector()
# plt.show()

dec_info = d.get_param()
envir_info = e.get_param()




res_a1 = am.Resolution(dec_info, envir_info, m)
N = 20 # test num correspond to the steps in para
re_a1 = Result(N)
for i in tqdm(range(N)):
    ini, ret = res_a1.analytic_estimate()
    re_a1.append(ini, ret)

re_a1.analytic_plot('p', 'dp')
```


#### 3.2 Kalman estimate
By simply changing the method to `Kalman_method`, you can use the Kalman method to estimate the resolution.

```python
from DDFS.element import *
import DDFS.kalman_method as km


res_k = km.Resolution(dec_info, envir_info, m)


N = 10*1000 # test num correspond to the steps in para
result = Result(N)
for i in tqdm(range(N)):
    res_k.generate_path()
    res_k.generate_ref_path()
    res_k.backward_kalman_estimate()
    ini, res = res_k.result_analysis()
    result.append(ini, res)
    
    
result.kalman_post_process_all(emit_mode, len(dec))
result.kalman_plot('p', 'dp', emit_mode=emit_mode, layer_idx=2, filter="backward")
```




























