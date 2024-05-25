import numpy as np

class StateVectorSimulator:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.state_vector = np.zeros(2**num_qubits, dtype=complex)
        self.state_vector[0] = 1.0  # Initialize with |0...0> state.

    def apply_gate(self, gate_matrix, target_qubits):
        # Check if the gate_matrix and target_qubits are compatible.
        num_target_qubits = len(target_qubits)
        if gate_matrix.shape != (2**num_target_qubits, 2**num_target_qubits):
            raise ValueError("Invalid gate_matrix dimensions for the specified target qubits.")

        # Calculate the new state vector.
        state_vector_copy = np.copy(self.state_vector)
        for i in range(2**self.num_qubits):
            bits = [int(bit) for bit in format(i, f'0{self.num_qubits}b')]
            if all(bits[qubit] == 0 for qubit in target_qubits):
                new_bits = bits[:]
                for j, qubit in enumerate(target_qubits):
                    new_bits[qubit] = 0 if bits[qubit] == 1 else 1
                new_index = sum([new_bits[qubit] * 2**(self.num_qubits - qubit - 1) for qubit in range(self.num_qubits)])
                self.state_vector[new_index] = sum([gate_matrix[i, j] * state_vector_copy[j] for j in range(2**num_target_qubits)])

    def measure(self):
        # Normalize the probabilities
        probabilities = np.abs(self.state_vector)**2
        probabilities /= np.sum(probabilities)

        outcome = np.random.choice(2**self.num_qubits, p=probabilities)
        return outcome, probabilities

    def get_state_vector(self):
        return self.state_vector

# Example usage for 4 qubits:
if __name__ == "__main__":
    num_qubits = 4
    simulator = StateVectorSimulator(num_qubits)

    # Apply a custom 16x16 unitary gate as an example
    unitary_matrix_4qubits = np.kron(np.kron(np.kron(
        np.array([[1, 0], [0, 1]], dtype=complex),
        np.array([[1/np.sqrt(2), 1/np.sqrt(2)], [1/np.sqrt(2), -1/np.sqrt(2)]], dtype=complex)),
        np.array([[1/np.sqrt(2), 1/np.sqrt(2)], [1/np.sqrt(2), -1/np.sqrt(2)]], dtype=complex)),
        np.array([[1, 0], [0, 1]], dtype=complex))
    
    simulator.apply_gate(unitary_matrix_4qubits, target_qubits=[0, 1, 2, 3])

    state_vector = simulator.get_state_vector()
    print("Final State Vector:")
    print(state_vector)

    outcome, probabilities = simulator.measure()
    print("Measurement Outcome:", outcome)
    print("Measurement Probabilities:", probabilities)
# This code should work for 4 qubits and demonstrates how to apply a 16x16 unitary matrix as an example.
