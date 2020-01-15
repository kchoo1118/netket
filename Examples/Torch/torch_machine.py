# Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import netket as nk
import torch
import numpy as np

# 1D Lattice
g = nk.graph.Hypercube(length=20, n_dim=1, pbc=True)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=0.5, graph=g)

# Ising spin hamiltonian
ha = nk.operator.Ising(h=1.0, hilbert=hi)


input_size = hi.size
alpha = 1

model = torch.nn.Sequential(
    torch.nn.Linear(input_size, alpha * input_size),
    torch.nn.ReLU(),
    torch.nn.Linear(alpha * input_size, 2),
    torch.nn.ReLU(),
)

ma = nk.machine.Torch(model, hilbert=hi)

ma.parameters = 0.1 * (np.random.randn(ma.n_par))

# Metropolis Local Sampling
sa = nk.sampler.MetropolisLocal(machine=ma, n_chains=8)

# Optimizer
op = nk.optimizer.AdaDelta()

# Stochastic reconfiguration
gs = nk.variational.Vmc(
    hamiltonian=ha, sampler=sa, optimizer=op, n_samples=500, method="Gd"
)

gs.run(output_prefix="test", n_iter=30000)
