Quantum Cartpole
=========

Installation
------------

To get started with the Quantum Cartpole environment, you can install it by simply cloning this repository:

```
git clone https://github.com/condensedAI/quantum-cartpole.git
```
Basic Usage
------------

The Quantum Cartpole environment adheres to the OpenAI Gym standard and can be used as follows. It offers a choice of three different potentials: quadratic, cosine, and quartic, and can be switched between classical and quantum environment.

```
import gym
import gym_qcart
env = gym.make('qcart-v0', potential = 'quadratic', system = 'quantum')
```
To run examples for training and testing the agents, you can use the following commands:

For training with a quadratic potential:
```
python -m example.train
```

For testing:
```
python -m example.test quadratic
```

Reference
---------

When you cite this repository, please use the following:

.. code-block:: bibtex

  @software{meinerz_2023_10060570,
    author       = {Meinerz, Kai},
    title        = {Quantum cartpole environment},
    month        = nov,
    year         = 2023,
    publisher    = {Zenodo},
    version      = {v0.1.0},
    doi          = {10.5281/zenodo.10060570},
    url          = {https://doi.org/10.5281/zenodo.10060570}
  }
