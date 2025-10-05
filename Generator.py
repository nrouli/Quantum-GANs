from qiskit_algorithms.utils import algorithm_globals
from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import SamplerQNN
from torch import nn, float32



class QuantumGenerator(nn.Module):
    def __init__(self, num_qubits, circuit_reps=5, seed=42):
        super().__init__()
        self.num_qubits = num_qubits
        self.circuit_reps = circuit_reps
        self.statevector_size = 2 ** num_qubits
        
        # Set random seed
        algorithm_globals.random_seed = seed
        
        # Create the quantum components
        self.torch_connector = self._create_quantum_circuit()

    def _create_quantum_circuit(self):
        """Create quantum circuit with rotation angles from statevector"""
        qc = QuantumCircuit(self.num_qubits)
        
        qc.h(qc.qubits)
        
        # Add your ansatz
        ansatz = EfficientSU2(
            num_qubits=self.num_qubits,
            reps=self.circuit_reps
        )
        qc.compose(ansatz, inplace=True)
        
        from qiskit_algorithms.gradients import ParamShiftSamplerGradient
        sampler = StatevectorSampler(default_shots=1024, seed=algorithm_globals.random_seed)
        gradient = ParamShiftSamplerGradient(sampler=sampler)
        
        qnn = SamplerQNN(
            circuit=qc,
            sampler=sampler,
            input_params=[],
            weight_params=ansatz.parameters,
            input_gradients=True,
            sparse=False,
            gradient=gradient
        )
        
        initial_weights = algorithm_globals.random.random(len(ansatz.parameters))
        return TorchConnector(qnn, initial_weights)
    
    
    
    def forward(self, noise):
        """Forward pass"""
        if noise.dim() == 1:
            noise = noise.unsqueeze(1)
        
        out = self.torch_connector(noise)
        # out = out.to(float32)
        return out.view(noise.size(0), 1, 8, 8)