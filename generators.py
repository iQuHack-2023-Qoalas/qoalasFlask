# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A circuit that encodes a discretized normal probability distribution in qubit amplitudes."""

from typing import Tuple, Union, List, Optional
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit import Aer
from qiskit.providers.fake_provider import FakeWashington, FakeMontreal, FakeNairobi
washington, montreal, fake_nairobi = FakeWashington(), FakeMontreal(), FakeNairobi()
from numpy.random import choice
ibm_sim_local = Aer.get_backend('qasm_simulator')
IBMQ.save_account('001725cf4ad0eafc1d267990a28fafc71dc62783be9ba638874f69dd2139964f2fe7209c23732257dc5f69ab26421772e46aae5c6dea61afac980b1632912116', overwrite=True)
IBMQ.load_account() # Load account from disk
IBM_provider = IBMQ.get_provider(hub='ibm-q-community')
nairobi = IBM_provider.backends('ibm_nairobi')[0]


import random 
from math import pi

class NormalDistribution(QuantumCircuit):
    r"""A circuit to encode a discretized normal distribution in qubit amplitudes.

    The probability density function of the normal distribution is defined as

    .. math::

        \mathbb{P}(X = x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x - \mu)^2}{\sigma^2}}

    .. note::

        The parameter ``sigma`` in this class equals the **variance**, :math:`\sigma^2` and not the
        standard deviation. This is for consistency with multivariate distributions, where the
        uppercase sigma, :math:`\Sigma`, is associated with the covariance.

    This circuit considers the discretized version of the normal distribution on
    ``2 ** num_qubits`` equidistant points, :math:`x_i`, truncated to ``bounds``.
    For a one-dimensional random variable, meaning `num_qubits` is a single integer, it applies
    the operation

    .. math::

        \mathcal{P}_X |0\rangle^n = \sum_{i=0}^{2^n - 1} \sqrt{\mathbb{P}(x_i)} |i\rangle

    where :math:`n` is `num_qubits`.

    .. note::

        The circuit loads the **square root** of the probabilities into the qubit amplitudes such
        that the sampling probability, which is the square of the amplitude, equals the
        probability of the distribution.

    In the multi-dimensional case, the distribution is defined as

    .. math::

        \mathbb{P}(X = x) = \frac{\Sigma^{-1}}{\sqrt{2\pi}} e^{-\frac{(x - \mu)^2}{\Sigma}}

    where :math:`\Sigma` is the covariance. To specify a multivariate normal distribution,
    ``num_qubits`` is a list of integers, each specifying how many
    qubits are used to discretize the respective dimension. The arguments ``mu`` and ``sigma``
    in this case are a vector and square matrix.
    If for instance, ``num_qubits = [2, 3]`` then ``mu`` is a 2d vector and ``sigma`` is the
    :math:`2 \times 2` covariance matrix. The first dimension is discretized using 2 qubits, hence
    on 4 points, and the second dimension on 3 qubits, hence 8 points. Therefore the random variable
    is discretized on :math:`4 \times 8 = 32` points.

    Since, in general, it is not yet known how to efficiently prepare the qubit amplitudes to
    represent a normal distribution, this class computes the expected amplitudes and then uses
    the ``QuantumCircuit.initialize`` method to construct the corresponding circuit.

    This circuit is for example used in amplitude estimation applications, such as finance [1, 2],
    where customer demand or the return of a portfolio could be modelled using a normal
    distribution.

    References:
        [1]: Gacon, J., Zoufal, C., & Woerner, S. (2020).
             Quantum-Enhanced Simulation-Based Optimization.
             `arXiv:2005.10780 <http://arxiv.org/abs/2005.10780>`_

        [2]: Woerner, S., & Egger, D. J. (2018).
             Quantum Risk Analysis.
             `arXiv:1806.06893 <http://arxiv.org/abs/1806.06893>`_

    """

    def __init__(
        self,
        num_qubits: Union[int, List[int]],
        mu: Optional[Union[float, List[float]]] = None,
        sigma: Optional[Union[float, List[float]]] = None,
        bounds: Optional[Union[Tuple[float, float], List[Tuple[float, float]]]] = None,
        upto_diag: bool = False,
        name: str = "N(X)",
    ) -> None:
        r"""
        Args:
            num_qubits: The number of qubits used to discretize the random variable. For a 1d
                random variable, ``num_qubits`` is an integer, for multiple dimensions a list
                of integers indicating the number of qubits to use in each dimension.
            mu: The parameter :math:`\mu`, which is the expected value of the distribution.
                Can be either a float for a 1d random variable or a list of floats for a higher
                dimensional random variable. Defaults to 0.
            sigma: The parameter :math:`\sigma^2` or :math:`\Sigma`, which is the variance or
                covariance matrix. Default to the identity matrix of appropriate size.
            bounds: The truncation bounds of the distribution as tuples. For multiple dimensions,
                ``bounds`` is a list of tuples ``[(low0, high0), (low1, high1), ...]``.
                If ``None``, the bounds are set to ``(-1, 1)`` for each dimension.
            upto_diag: If True, load the square root of the probabilities up to multiplication
                with a diagonal for a more efficient circuit.
            name: The name of the circuit.
        """

        _check_dimensions_match(num_qubits, mu, sigma, bounds)

        # set default arguments
        dim = 1 if isinstance(num_qubits, int) else len(num_qubits)
        if mu is None:
            mu = 0 if dim == 1 else [0] * dim

        if sigma is None:
            sigma = 1 if dim == 1 else np.eye(dim)

        if bounds is None:
            bounds = (-1, 1) if dim == 1 else [(-1, 1)] * dim

        if not isinstance(num_qubits, list):  # univariate case
            circuit = QuantumCircuit(num_qubits, name=name)

            x = np.linspace(bounds[0], bounds[1], num=2 ** num_qubits)
        else:  # multivariate case
            circuit = QuantumCircuit(sum(num_qubits), name=name)

            # compute the evaluation points using numpy's meshgrid
            # indexing 'ij' yields the "column-based" indexing
            meshgrid = np.meshgrid(
                *(
                    np.linspace(bound[0], bound[1], num=2 ** num_qubits[i])
                    for i, bound in enumerate(bounds)
                ),
                indexing="ij",
            )
            # flatten into a list of points
            x = list(zip(*(grid.flatten() for grid in meshgrid)))

        from scipy.stats import multivariate_normal

        # compute the normalized, truncated probabilities
        probabilities = multivariate_normal.pdf(x, mu, sigma)
        normalized_probabilities = probabilities / np.sum(probabilities)

        # store the values, probabilities and bounds to make them user accessible
        self._values = x
        self._probabilities = normalized_probabilities
        self._bounds = bounds

        # use default the isometry (or initialize w/o resets) algorithm to construct the circuit
        # pylint: disable=no-member
        if upto_diag:
            circuit.isometry(np.sqrt(normalized_probabilities), circuit.qubits, None)
        else:
            from qiskit.extensions import Initialize  # pylint: disable=cyclic-import

            initialize = Initialize(np.sqrt(normalized_probabilities))
            distribution = initialize.gates_to_uncompute().inverse()
            circuit.compose(distribution, inplace=True)

        super().__init__(*circuit.qregs, name=name)

        try:
            instr = circuit.to_gate()
        except QiskitError:
            instr = circuit.to_instruction()

        self.compose(instr, qubits=self.qubits, inplace=True)

    @property
    def values(self) -> np.ndarray:
        """Return the discretized points of the random variable."""
        return self._values

    @property
    def probabilities(self) -> np.ndarray:
        """Return the sampling probabilities for the values."""
        return self._probabilities

    @property
    def bounds(self) -> Union[Tuple[float, float], List[Tuple[float, float]]]:
        """Return the bounds of the probability distribution."""
        return self._bounds

def _check_dimensions_match(num_qubits, mu, sigma, bounds):
    num_qubits = [num_qubits] if not isinstance(num_qubits, (list, np.ndarray)) else num_qubits
    dim = len(num_qubits)

    if mu is not None:
        mu = [mu] if not isinstance(mu, (list, np.ndarray)) else mu
        if len(mu) != dim:
            raise ValueError(
                "Dimension of mu ({}) does not match the dimension of the "
                "random variable specified by the number of qubits ({})"
                "".format(len(mu), dim)
            )

    if sigma is not None:
        sigma = [[sigma]] if not isinstance(sigma, (list, np.ndarray)) else sigma
        if len(sigma) != dim or len(sigma[0]) != dim:
            raise ValueError(
                "Dimension of sigma ({} x {}) does not match the dimension of "
                "the random variable specified by the number of qubits ({})"
                "".format(len(sigma), len(sigma[0]), dim)
            )

    if bounds is not None:
        # bit differently to cover the case the users might pass `bounds` as a single list,
        # e.g. [0, 1], instead of a tuple
        bounds = [bounds] if not isinstance(bounds[0], tuple) else bounds
        if len(bounds) != dim:
            raise ValueError(
                "Dimension of bounds ({}) does not match the dimension of the "
                "random variable specified by the number of qubits ({})"
                "".format(len(bounds), dim)
            )


######################################################################################################

class UniformDistribution(QuantumCircuit):
    r"""A circuit to encode a discretized uniform distribution in qubit amplitudes.

    This simply corresponds to applying Hadamard gates on all qubits.

    The probability density function of the discretized uniform distribution on
    :math:`N` values is

    .. math::

        \mathbb{P}(X = x) = \frac{1}{N}.

    This circuit considers :math:`N = 2^n`, where :math:`n =` ``num_qubits`` and prepares the state

    .. math::

        \mathcal{P}_X |0\rangle^{\otimes n} = \frac{1}{\sqrt{2^n}} \sum_{x=0}^{2^n - 1} |x\rangle
    """

    def __init__(
        self, 
        num_qubits: int,
        name: str = "U(X)"
        ) -> None:
        """
        Args:
            num_qubits: The number of qubits in the circuit, the distribution is uniform over
                ``2 ** num_qubits`` values.
            name: The name of the circuit.
        """

        circuit = QuantumCircuit(num_qubits, name=name)
        circuit.h(circuit.qubits)

        super().__init__(*circuit.qregs, name=name)
        self.compose(circuit.to_gate(), qubits=self.qubits, inplace=True)

######################################################################################################

class PorterThomasDistribution(QuantumCircuit):
    r"""A circuit to encode a discretized Porter-Thomas distribution in qubit amplitudes.
    """

    def __init__(
        self,
        num_qubits: Union[int, List[int]],
        depth: int = 25,
        name: str = "PT(X)",
        mapping: bool = True,
    ) -> None:
        circuit = QuantumCircuit(num_qubits)
        circuit.name = name
        for layer in range(depth):
            if layer == 0:
                paulis = [random.randint(0,2) for i in range(num_qubits)]
            else:
                paulis = _choose_paulis(num_qubits, paulis)

            if mapping:
                _add_rotation(circuit, pos, paulis[pos])
            else:
                for pos in range(num_qubits):
                    _add_rotation(circuit, pos, paulis[pos])
                    if pos != 0:
                        if (pos % 2 and layer % 2) or not (pos % 2 or layer % 2):
                            circuit.cz(pos-1, pos)
                    if not (num_qubits%2 or layer%2): # connecting the last to the first
                        circuit.cz(-1, 0)
        
        super().__init__(*circuit.qregs, name=name)

        try:
            instr = circuit.to_gate()
        except QiskitError:
            instr = circuit.to_instruction()

        self.compose(instr, qubits=self.qubits, inplace=True)

def _add_rotation(circuit, pos, pauli):
    r"""Add a single-qubit gate to 'qc' in 'pos' position. 
    
    0 == sqrtX, 1 == sqrtY, 2 == T
    """
    if pauli == 0:
        circuit.rx(pi/2, pos)
        return
    if pauli == 1:
        circuit.ry(pi/2, pos)
        return
    if pauli == 2:
        circuit.t(pos)
        return

def _choose_paulis(num_qubits, last_paulis):
    paulis = []
    for i in range(num_qubits):
        s = [i for i in range(3)]
        s.remove(last_paulis[i])
        index = random.randint(0,1)
        paulis.append(s[index])
    return paulis
        
################################################################################################
from qiskit import execute
resolution_backends = {
    'classical_low' : fake_nairobi,
    'low' : nairobi,
    'medium' : montreal,
    'high' : washington,
}

class DeepThermalRandom(QuantumCircuit):
    def __init__(
            self,
            num_qubits: Union[int, List[int]],
            depth: int = 25,
            name: str = "DT(X)",
            mapping: bool = True,
            resolution: str = 'low',
        ) -> None:

        backend = resolution_backends[self.resolution]

        if backend.name() in ['ibm_nairobi', 'fake_nairobi']:
            pattern1 = [[0, 1], [5, 6]]
            pattern2 = [[1, 2], [3, 5]]
            pattern3 = [[1, 3], [4, 5]]
                
            circuit = QuantumCircuit(7)
            gate = entangling_gate().to_gate()
            
            for layer in range(depth):
                for pair in pattern1:
                    circuit.append(gate, pair)
                for pair in pattern2:
                    circuit.append(gate, pair)
                for pair in pattern3:
                    circuit.append(gate, pair)

            circuit.measure_all()
            shots = 2**5
            initial_layout = [0, 1, 2, 3, 4, 5, 6]
            
            result = execute(circuit, backend, shots=shots, initial_layout=initial_layout).result()
            results_dict = result.get_counts()
            conditional_dict = {i: 0 for i in generate_binary_strings(3)}

            for result in results_dict:
                if result[3:] == '0' * 4:
                    conditional_dict[result[:3]] += results_dict[result]

            answer = choice(list(conditional_dict.keys()), p = np.array(list(conditional_dict.values())) / sum(
            list(conditional_dict.values())))
            return answer
        else: #generic mid-size IBM Quantum Device
            config = backend.configuration()
            N = config.num_qubits
            cmap = config.coupling_map
            graph = {j: {k for k in find_neighbours(cmap, j)} for j in range(N)} 
            try:
                use_this_cycle = best_cycle(backend, graph, num_qubits)
            except:
                raise Exception('No cycles of this length found on your backend')

            gate = entangling_gate().to_gate()

            circuit = QuantumCircuit(num_qubits)
            for i in range(depth):
                for j in range(num_qubits):
                    if j%2 == 0:
                        try:
                            circuit.append(gate, [j, j+1])
                        except:
                            None
                if num_qubits%2 == 1:
                    circuit.append(gate, [num_qubits-1, 0])
                for j in range(num_qubits):
                    if j%2 == 1:
                        try:
                            circuit.append(gate, [j, j+1])
                        except:
                            None
                if num_qubits%2 == 0:
                    circuit.append(gate, [num_qubits-1, 0])
            shots = 2**(int(num_qubits/2) + 1)     
            A_size = int(num_qubits/2)
            B_size = num_qubits - int(num_qubits/2)
            circuit.measure_all()            
            initial_layout = use_this_cycle
            result = execute(circuit, backend, shots=shots, initial_layout=initial_layout).result()
            results_dict = result.get_counts()
            conditional_dict = {i: 0 for i in generate_binary_strings(A_size)}

            for result in results_dict:
                if result[A_size:] == '0' * B_size:
                    conditional_dict[result[:A_size]] += results_dict[result]

            answer = choice(list(conditional_dict.keys()), p = np.array(list(conditional_dict.values())) / sum(
            list(conditional_dict.values())))
        
        return int(answer, 2)

def find_neighbours(coupling_map, j):
    neighbours = []
    for bond in coupling_map:
        if bond[0] == j:
            neighbours.append(bond[1])
    return neighbours

def find_cycles_recursive(graph, L, cycle):
    successors = graph[cycle[-1]]
    if len(cycle) == L:
        if cycle[0] in successors:
            yield cycle
    elif len(cycle) < L:
        for v in successors:
            if v in cycle:
                continue
            yield from find_cycles_recursive(graph, L, cycle + [v])

def find_cycles(graph, L):
    for v in graph:
        yield from find_cycles_recursive(graph, L, [v])
        
def unique_cycles(graph, L):
    list_cycles = list(find_cycles(graph, L))
    if list_cycles == []:
        return []
    unique_cycles = [list_cycles[0]]
    for cycle in list_cycles:
        if set(cycle) not in [set(cycle) for cycle in unique_cycles]:
            unique_cycles.append(cycle)
    return unique_cycles

def best_cycle(backend, graph, L):
    cycles_to_look_at = [list(cycle) for cycle in unique_cycles(graph, L)]
    errors = []
    for cycle in cycles_to_look_at:
        total_error_cycle = 1
        qpairs = [[cycle[i], cycle[i+1]] for i in range(L-1)]
        for pair in qpairs:
            total_error_cycle *= backend.properties().gate_error('cx', pair)
        errors.append(total_error_cycle)
    min_error = min(errors)
    return cycles_to_look_at[errors.index(min_error)]

def entangling_gate():  
    circ = QuantumCircuit(2)
    circ.rzz(np.pi/2, 0, 1)
    circ.rx(np.pi/2, 0)
    circ.rx(np.pi/2, 1)
    circ.rzz(np.pi/2, 0, 1)
    return circ

def generate_binary_strings(bit_count):
    binary_strings = []
    def genbin(n, bs=''):
        if len(bs) == n:
            binary_strings.append(bs)
        else:
            genbin(n, bs + '0')
            genbin(n, bs + '1')
    genbin(bit_count)
    return binary_strings