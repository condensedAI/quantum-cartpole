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

.. code-block:: bash

	import gym
	import gym_qcart
	env = gym.make('qcart-v0', potential = 'quadratic', system = 'quantum')

To run examples for training and testing the agents, you can use the following commands:

For training with a quadratic potential:
.. code-block:: bash 

	python -m example.train


For testing:
.. code-block:: bash 

	python -m example.test quadratic
