import numpy as np
from typing import Tuple, List, Dict

class LDPCDecoder:
    """
    LDPC Decoder using Loopy Belief Propagation
    """

    def __init__(self, H: List[List[int]], y: List[int], noise_ratio: float, max_iter: int = 20):
        """
        Initialise LDPC decoder.
        """
        self.H = np.array(H)
        self.y = np.array(y)
        self.noise_ratio = noise_ratio
        self.max_iter = max_iter

        self.diagnostic: Dict[str, int] = {
            'RET_CODE': -1,
            'ITERATIONS': 0
        }

        ## Neighbourhoods ##
        self.check_nbrs = {i: [j for j in range(len(H[0])) if H[i][j] != 0] for i in range(len(H))}
        self.bit_nbrs = {j: [i for i in range(len(H)) if H[i][j] != 0] for j in range(len(H[0]))}

        ## Init messages ##
        self.messages = self._initialise_messages()
        self.init_messages = self.messages.copy()

    def _initialise_messages(self) -> List[List[float]]:
        """
        Initialise the messages
        """
        M = [[0.0 for _ in range(self.H.shape[1])] for _ in range(self.H.shape[0])]

        for i, bit in enumerate(self.y):
            for check in self.bit_nbrs[i]:
                if bit == 0:
                    M[check][i] = np.log((1 - self.noise_ratio) / self.noise_ratio)
                else:
                    M[check][i] = np.log(self.noise_ratio / (1 - self.noise_ratio))
        return M

    def _vtc(self, col: List[float], nbrs: List[int]) -> List[float]:
        """
        Variable-to-check update
        """
        total_sum = sum(col[i] for i in nbrs)
        return [total_sum - col[i] for i in nbrs]

    def _ctv(self, row: List[float], nbrs: List[int]) -> List[float]:
        """
        Check-to-variable update
        """
        tanh_product = np.prod([np.tanh(0.5 * row[i]) for i in nbrs])
        return [2 * np.arctanh(tanh_product / np.tanh(0.5 * row[i])) for i in nbrs]

    def decode(self) -> Tuple[Dict[str, int], List[int]]:
        """
        Perform decoding.
        """
        M = [row.copy() for row in self.messages]
        z = self.y.copy()
        step = 0

        while step <= self.max_iter:
            ## Check to var updates ##
            for i in range(len(self.H)):
                updated_row = self._ctv(M[i], self.check_nbrs[i])
                for idx, j in enumerate(self.check_nbrs[i]):
                    M[i][j] = updated_row[idx]

            step += 1
            self.diagnostic['ITERATIONS'] = step

            ## Posterior log-likelihood ##
            posterior = [sum(row[i] for row in M) for i in range(len(self.y))]
            for i in range(len(self.y)):
                if self.y[i] == 0:
                    posterior[i] += np.log((1 - self.noise_ratio) / self.noise_ratio)
                else:
                    posterior[i] += np.log(self.noise_ratio / (1 - self.noise_ratio))

            z = [0 if p > 0 else 1 for p in posterior]

            ## Terminate? ##
            test = np.mod(np.dot(self.H, z), 2)
            if np.all(test == 0):
                self.diagnostic['RET_CODE'] = 0
                break

            ## Var to check Updates ##
            for i in range(len(self.y)):
                updated_col = self._vtc([M[j][i] for j in range(len(self.H))], self.bit_nbrs[i])
                for idx, j in enumerate(self.bit_nbrs[i]):
                    M[j][i] = updated_col[idx] + self.init_messages[j][i]

        return self.diagnostic, z


def recover(decoded_vector: List[int]) -> str:
    """
    Recover the original msg
    """
    binary = decoded_vector[:248]
    message = ''.join(
        chr(int(''.join(map(str, binary[i:i + 8])), 2))
        for i in range(0, 248, 8)
    )
    return message


if __name__ == "__main__":
    with open('H1.txt', 'r') as f:
        H = [[int(x) for x in line.split()] for line in f if line.strip()]

    with open('y1.txt', 'r') as f:
        y = [int(x) for x in f.read().split()]

    noise = 0.1

    decoder = LDPCDecoder(H, y, noise)
    diagnostic, decoded = decoder.decode()
    recovered = recover(decoded)

    print("Decoded Vector:", decoded)
    print(len(decoded))
    print("Diagnostic:", diagnostic)
    print("Recovered message:", recovered)
