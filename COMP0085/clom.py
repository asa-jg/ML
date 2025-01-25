import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# STUB / PLACEHOLDER CLASSES & FUNCTIONS (Replace or adapt as needed)
###############################################################################

class BinaryLatentFactorModel:
    """
    A placeholder/stub for the base binary latent factor model class.
    Replace this with the actual implementation if you have it.
    """
    def __init__(self, mu: np.ndarray, sigma: float, pi: np.ndarray):
        self.mu = mu      # shape (D, K)
        self.sigma = sigma
        self.pi = pi      # shape (1, K) or (K,)

    @property
    def precision(self) -> float:
        """Precision = 1 / sigma^2"""
        return 1.0 / (self.sigma**2)

    @property
    def log_pi(self) -> np.ndarray:
        """Log of pi (clamped)."""
        return np.log(np.clip(self.pi, 1e-15, 1.0))

    @property
    def log_oneminus_pi(self) -> np.ndarray:
        """Log of (1 - pi) (clamped)."""
        return np.log(np.clip(1 - self.pi, 1e-15, 1.0))

    @staticmethod
    def calculate_maximisation_parameters(x: np.ndarray, approximation) -> tuple:
        """
        Stub for computing mu, sigma, pi from data x and an approximation.

        Replace this with a real 'maximisation' or M-step calculation.
        For demonstration, returns random values matching shapes.
        """
        n, d = x.shape
        k = approximation.bernoulli_parameter_matrix.shape[1]

        # Dummy parameters for demonstration
        mu = np.random.randn(d, k)
        sigma = np.abs(np.random.rand()) + 0.1
        pi = np.clip(np.random.rand(1, k), 1e-2, 1 - 1e-2)
        return mu, sigma, pi


class AbstractBinaryLatentFactorApproximation:
    """
    A placeholder/stub for the abstract approximation class.
    Replace with the actual interface if you have it.
    """
    def compute_free_energy(self, x: np.ndarray, model: BinaryLatentFactorModel) -> float:
        """
        Stub method. The actual method should compute the free energy
        or an analogous objective function for the approximation.
        """
        return 0.0

def learnbinaryfactors(
    x: np.ndarray,
    emiterations: int,
    binary_latent_factor_model: BinaryLatentFactorModel,
    binary_latent_factor_approximation: AbstractBinaryLatentFactorApproximation
):
    """
    Stub for an EM-style loop. Replace with your real 'learnbinaryfactors' method.

    Returns:
      - the (possibly updated) approximation,
      - the (possibly updated) model,
      - a list of free-energy values over iterations.
    """
    free_energy_values = []

    # For demonstration: run a few dummy EM steps
    for step in range(emiterations):
        # E-step: update approximation via 'variational_expectation_step' or similar
        # (Here we call a stub method that doesn't do anything real)
        step_free_energy = binary_latent_factor_approximation.compute_free_energy(
            x, binary_latent_factor_model
        )
        free_energy_values.append(step_free_energy)

        # M-step: update model
        new_mu, new_sigma, new_pi = BinaryLatentFactorModel.calculate_maximisation_parameters(
            x, binary_latent_factor_approximation
        )
        binary_latent_factor_model.mu = new_mu
        binary_latent_factor_model.sigma = new_sigma
        binary_latent_factor_model.pi = new_pi

    return binary_latent_factor_approximation, binary_latent_factor_model, free_energy_values


###############################################################################
# BOLTZMANN MACHINE CODE
###############################################################################

class BoltzmannMachine(BinaryLatentFactorModel):
    """
    Binary latent factor model with Boltzmann Machine terms.
    Inherits from BinaryLatentFactorModel.
    """
    def __init__(
        self,
        mu: np.ndarray,
        sigma: float,
        pi: np.ndarray,
    ):
        super().__init__(mu, sigma, pi)

    @property
    def wmatrix(self) -> np.ndarray:
        """
        Weight matrix of the Boltzmann machine.

        :return: (K, K) matrix of weights
        """
        return -self.precision * (self.mu.T @ self.mu)

    def wmatrix_index(self, i: int, j: int) -> float:
        """
        Weight matrix at a specific index.

        :param i: row index
        :param j: column index
        :return: weight value
        """
        return -self.precision * (self.mu[:, i] @ self.mu[:, j])

    def b(self, x: np.ndarray) -> np.ndarray:
        """
        b term in the Boltzmann machine for all data points.

        :param x: design matrix (num_points, num_dimensions)
        :return: matrix of shape (num_points, num_latent_variables)
        """
        return -(
            self.precision * x @ self.mu
            + self.log_piratio
            - 0.5 * self.precision * np.multiply(self.mu, self.mu).sum(axis=0)
        )

    def bindex(self, x: np.ndarray, node_index: int) -> np.ndarray:
        """
        b term for a specific node in the Boltzmann machine for all data points.

        :param x: design matrix (num_points, num_dimensions)
        :param node_index: node index
        :return: vector of shape (num_points,)
        """
        return -(
            self.precision * x @ self.mu[:, node_index]
            + (self.log_pi[0, node_index] - self.log_oneminus_pi[0, node_index])
            - 0.5 * self.precision * (self.mu[:, node_index] @ self.mu[:, node_index])
        ).reshape(-1)

    @property
    def log_piratio(self) -> np.ndarray:
        """Helper to return log_pi - log_oneminus_pi."""
        return self.log_pi - self.log_oneminus_pi


def init_boltzmann_machine(
    x: np.ndarray,
    binary_latent_factor_approximation: AbstractBinaryLatentFactorApproximation
) -> BinaryLatentFactorModel:
    """
    Initialise by running a maximisation step with the parameters of the
    binary latent factor approximation.

    :param x: data matrix (num_points, num_dimensions)
    :param binary_latent_factor_approximation: an approximation instance
    :return: an initialised BoltzmannMachine model
    """
    mu, sigma, pi = BinaryLatentFactorModel.calculate_maximisation_parameters(
        x, binary_latent_factor_approximation
    )
    return BoltzmannMachine(mu=mu, sigma=sigma, pi=pi)


###############################################################################
# MESSAGE PASSING APPROXIMATION CODE
###############################################################################

class MessagePassingApproximation(AbstractBinaryLatentFactorApproximation):
    """
    bernoulli_parameter_matrix (theta): shape (n, k, k)
      - Off-diagonals correspond to g_{ij, ¬s_i}(s_j) for data point n
      - Diagonals correspond to f_i(s_i)
    """

    def __init__(self, bernoulli_parameter_matrix: np.ndarray):
        self.bernoulli_parameter_matrix = bernoulli_parameter_matrix

    @property
    def k(self) -> int:
        """
        Number of latent variables (extracted from second dimension).
        """
        return self.bernoulli_parameter_matrix.shape[1]

    @property
    def lambda_matrix(self) -> np.ndarray:
        """
        Aggregate messages and compute parameter for Bernoulli distribution.

        :return: (num_points, num_latent_variables) array of responsibilities
        """
        val = 1.0 / (1.0 + np.exp(-self.natural_parameter_matrix.sum(axis=1)))
        val[val == 0] = 1e-10
        val[val == 1] = 1 - 1e-10
        return val

    @property
    def natural_parameter_matrix(self) -> np.ndarray:
        """
        Matrix of natural parameters (eta).

        Shape: (num_points, k, k)
        Off-diagonals: g_{ij, ¬s_i}(s_j) for data point n
        Diagonals:     f_i(s_i)
        """
        return np.log(
            np.divide(
                self.bernoulli_parameter_matrix,
                1 - self.bernoulli_parameter_matrix
            )
        )

    def aggregate_incoming_binary_factor_messages(
        self,
        node_index: int,
        excluded_node_index: int
    ) -> np.ndarray:
        """
        Sum over all incoming messages except from excluded_node_index -> node_index.

        :param node_index: the node whose messages we are aggregating
        :param excluded_node_index: the node to exclude
        :return: shape (num_points,)
        """
        first_part = np.sum(
            self.natural_parameter_matrix[:, :excluded_node_index, node_index],
            axis=1
        )
        second_part = np.sum(
            self.natural_parameter_matrix[:, excluded_node_index + 1:, node_index],
            axis=1
        )
        return (first_part + second_part).reshape(-1)

    @staticmethod
    def calculate_bernoulli_parameter(natural_parameter_matrix: np.ndarray) -> np.ndarray:
        """
        Convert natural parameters to Bernoulli parameters via logistic function.
        """
        val = 1.0 / (1.0 + np.exp(-natural_parameter_matrix))
        val[val == 0] = 1e-10
        val[val == 1] = 1 - 1e-10
        return val

    def variational_expectation_step(
        self,
        x: np.ndarray,
        binary_latent_factor_model: BoltzmannMachine
    ) -> list:
        """
        Iteratively updates singleton and binary factors.
        Returns free energies after each update.
        """
        free_energy = [self.compute_free_energy(x, binary_latent_factor_model)]

        for i in range(self.k):
            # Singleton factor update
            natural_parameter_ii = self.calculate_singleton_message_update(
                x=x,
                boltzmann_machine=binary_latent_factor_model,
                i=i,
            )
            self.bernoulli_parameter_matrix[:, i, i] = self.calculate_bernoulli_parameter(
                natural_parameter_ii
            )
            free_energy.append(self.compute_free_energy(x, binary_latent_factor_model))

            for j in range(i):
                # Binary factor update i->j and j->i
                natural_parameter_ij = self.calculate_binary_message_update(
                    x=x,
                    boltzmann_machine=binary_latent_factor_model,
                    i=i,
                    j=j,
                )
                self.bernoulli_parameter_matrix[:, i, j] = self.calculate_bernoulli_parameter(
                    natural_parameter_ij
                )

                natural_parameter_ji = self.calculate_binary_message_update(
                    x=x,
                    boltzmann_machine=binary_latent_factor_model,
                    i=j,
                    j=i,
                )
                self.bernoulli_parameter_matrix[:, j, i] = self.calculate_bernoulli_parameter(
                    natural_parameter_ji
                )
                free_energy.append(self.compute_free_energy(x, binary_latent_factor_model))

        return free_energy

    def calculate_binary_message_update(
        self,
        x: np.ndarray,
        boltzmann_machine: BoltzmannMachine,
        i: int,
        j: int
    ) -> np.ndarray:
        """
        Calculate new parameters for a binary-factored message.

        :param x: data matrix (n, d)
        :param boltzmann_machine: BoltzmannMachine model
        :param i, j: node indices
        :return: shape (n,)
        """
        natural_parameter_i_not_j = (
            boltzmann_machine.bindex(x=x, node_index=i)
            + self.aggregate_incoming_binary_factor_messages(
                node_index=i, excluded_node_index=j
            )
        )
        wij = boltzmann_machine.wmatrix_index(i, j)

        # log(1 + exp(wij + term)) - log(1 + exp(term))
        return np.log(1 + np.exp(wij + natural_parameter_i_not_j)) - np.log(
            1 + np.exp(natural_parameter_i_not_j)
        )

    @staticmethod
    def calculate_singleton_message_update(
        x: np.ndarray,
        boltzmann_machine: BoltzmannMachine,
        i: int
    ) -> np.ndarray:
        """
        Calculate the parameter update for the singleton message.
        """
        return boltzmann_machine.bindex(x=x, node_index=i)

    def compute_free_energy(self, x: np.ndarray, model: BinaryLatentFactorModel) -> float:
        """
        Example placeholder method to 'compute free energy' of the system.
        Replace with a real implementation as desired.
        """
        # For demonstration, just sum up the negative log-likelihood or something similar.
        # This is a placeholder:
        return float(np.random.rand(1))


def init_message_passing(k: int, n: int) -> MessagePassingApproximation:
    """
    Message passing initialisation.

    :param k: number of latent variables
    :param n: number of data points
    :return: a MessagePassingApproximation instance
    """
    bernoulli_parameter_matrix = np.random.random(size=(n, k, k))
    return MessagePassingApproximation(bernoulli_parameter_matrix)


###############################################################################
# "RUN" FUNCTION (PUTTING IT ALL TOGETHER)
###############################################################################

def run(x: np.ndarray, k: int, em_iterations: int, savepath: str) -> None:
    """
    Example top-level function for running Loopy-BP-based EM on data 'x'.
    Plots initial and final features, plus free energy over iterations.
    """
    n = x.shape[0]

    # 1) Init message passing approximation
    message_passing = init_message_passing(k, n)

    # 2) Init Boltzmann machine
    boltzmann_machine = init_boltzmann_machine(x, message_passing)

    # Plot initial features
    fig, ax = plt.subplots(1, k, figsize=(k * 2, 2))
    for i in range(k):
        ax[i].imshow(boltzmann_machine.mu[:, i].reshape(4, 4))
        ax[i].set_title(f"Latent Feature mu{i}")
    fig.suptitle("Initial Features (Loopy BP)")
    plt.tight_layout()
    plt.savefig(savepath + "-init-latent-factors", bbox_inches="tight")
    plt.close()

    # 3) Run EM
    message_passing, boltzmann_machine, free_energy = learnbinaryfactors(
        x=x,
        emiterations=em_iterations,
        binary_latent_factor_model=boltzmann_machine,
        binary_latent_factor_approximation=message_passing,
    )

    # Plot learned features
    fig, ax = plt.subplots(1, k, figsize=(k * 2, 2))
    for i in range(k):
        ax[i].imshow(boltzmann_machine.mu[:, i].reshape(4, 4))
        ax[i].set_title(f"Latent Feature mu{i}")
    fig.suptitle("Learned Features (Loopy BP)")
    plt.tight_layout()
    plt.savefig(savepath + "-latent-factors", bbox_inches="tight")
    plt.close()

    # Plot free energy
    plt.title("Free Energy (Loopy BP)")
    plt.xlabel("t (EM steps)")
    plt.ylabel("Free Energy")
    plt.plot(free_energy, marker='o')
    plt.savefig(savepath + "-free-energy", bbox_inches="tight")
    plt.close()
    print("helo")


###############################################################################
# EXAMPLE MAIN (Optional)
###############################################################################

if __name__ == "__main__":
    # Example usage

    # Synthetic data
    num_factors = 8
    num_samples = 200
    dim = 16

    # Generate random data for demonstration
    X = np.random.randn(num_samples, dim)

    # Run
    run(X, k=num_factors, em_iterations=10, savepath="example_output")
