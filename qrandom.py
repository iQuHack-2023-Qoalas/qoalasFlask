"""
This module utilizes quantum computers 
to produce truly random numbers
"""
from qiskit import IBMQ, Aer, transpile, execute
from qiskit.tools.monitor import job_monitor
from generators import NormalDistribution,UniformDistribution,PorterThomasDistribution,DeepThermalRandom
import covalent as ct

DISTRIBUTION_TYPES = ['normal', 'uniform', 'porterthomas', 'deepthermo']

@ct.electron
def get_random_seed(distribution_type, backend, ibm_sim_local, SHOTS, NUM_QUBITS):
    if distribution_type == 'normal':
        circuit = NormalDistribution(num_qubits = NUM_QUBITS)
    elif distribution_type == 'uniform':
        circuit = UniformDistribution(num_qubits = NUM_QUBITS)
    elif distribution_type == 'porterthomas':
        circuit = PorterThomasDistribution(num_qubits = NUM_QUBITS)
    elif distribution_type == 'deepthermo':
        answer = DeepThermalRandom(num_qubits = NUM_QUBITS, backend=backend)
        return int(answer, 2)

    circuit.measure_all()
    circuit = transpile(circuit, backend=backend)
    job = execute(circuit, ibm_sim_local, shots=SHOTS)
    job_monitor(job)
    counts = job.result().get_counts()
    rnd_number = 0
    num_shot = 1
    for k in counts.keys():
        while counts[k] > 0:
            rnd_number += int(k, 2) / (2**(NUM_QUBITS*num_shot))
            num_shot += 1
            counts[k] -= 1
    return rnd_number

@ct.lattice
def workflow(number_of_seeds, distribution_type_index):
    IBMQ.save_account('001725cf4ad0eafc1d267990a28fafc71dc62783be9ba638874f69dd2139964f2fe7209c23732257dc5f69ab26421772e46aae5c6dea61afac980b1632912116', overwrite=True)
    IBMQ.load_account() # Load account from disk
    IBM_provider = IBMQ.get_provider(hub='ibm-q-community')
    ibm_nairobi = IBM_provider.backends('ibm_nairobi')[0]
    ibm_oslo = IBM_provider.backends('ibm_oslo')[0]
    # ibm_sim_online = IBM_provider.backends('ibmq_qasm_simulator')[0]
    ibm_sim_local = Aer.get_backend('qasm_simulator')
    SHOTS = 10
    NUM_QUBITS = 7
    distribution_type = DISTRIBUTION_TYPES[distribution_type_index]
    numb_of_random = 10
    backend = [ibm_nairobi, ibm_oslo, ibm_sim_local][2]
    filename = 'output_'+distribution_type+'.txt'
    random_seeds = []
    random_seeds = [get_random_seed(distribution_type, backend, ibm_sim_local, SHOTS, NUM_QUBITS) for _ in range(number_of_seeds)]
    return random_seeds


def send_covalent_request(number_of_seeds, distribution_type_index=1):
    dispatch_id = ct.dispatch(workflow)(number_of_seeds, distribution_type_index)
    result = ct.get_result(dispatch_id)
    # filename = 'output_'+DISTRIBUTION_TYPES[distribution_type_index]+'.yml'
    # with open(filename, "a") as fo:
    #     for seed in result:
    #         fo.write("%f\n"%seed)

    for i in result:
        print(i)

send_covalent_request(1)
