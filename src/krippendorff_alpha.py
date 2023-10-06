
import numpy as np
import pandas as pd
import random
from tqdm import tqdm

class Krippendorff():
    """
    Calculate Krippendorff's Alpha reliability estimate for inter-rater agreement.

    This class computes Krippendorff's Alpha, a measure of inter-rater agreement, based on a given dataset.

    Parameters:
    - x (numpy.ndarray): The input dataset where rows represent observers, and columns represent units or items being rated.
    - method (str, optional): The measurement level of the data, one of {"nominal", "ordinal", "interval", "ratio"}.
      Defaults to "nominal".
    - run_bootstrap (bool, optional): Whether to perform a bootstrap procedure to estimate confidence intervals.
      Defaults to False.

    Attributes:
    - x (numpy.ndarray): The input dataset.
    - method (str): The measurement level of the data.
    - if_run_bootstrap (bool): Indicates if the bootstrap procedure is enabled.
    - number_of_observers (int): The number of observers in the dataset.
    - number_of_units (int): The number of units or items being rated.
    - unique_values (numpy.ndarray): Unique values in the dataset.
    - unique_values_without_nan (int): Number of unique values excluding NaN.
    - pairable_values (numpy.ndarray): Number of non-NaN ratings for each unit.
    - coincidence_matrix (numpy.ndarray): Observed coincidence matrix.
    - expected_matrix (numpy.ndarray): Expected coincidence matrix.
    - delta_matrix (numpy.ndarray): Delta matrix.
    - alpha (float): Krippendorff's Alpha reliability estimate.
    - boot_alphas (numpy.ndarray): Bootstrap estimates of Alpha (if run_bootstrap is True).
    - boot_iterations (int): Number of bootstrap iterations (if run_bootstrap is True).
    - lower_confidence_level (float): Lower bound of confidence interval (if run_bootstrap is True).
    - upper_confidence_level (float): Upper bound of confidence interval (if run_bootstrap is True).
    - alpha_min_probability (float): Probability of achieving Alpha greater than alpha_min (if run_bootstrap is True).
    - alpha_min_list (list): List of alpha_min values (if run_bootstrap is True).
    - probabilities_for_alphas (list): List of probabilities for achieving specified alpha_min values (if run_bootstrap is True).

    Methods:
    - get_coincidence_matrix(): Calculate the observed coincidence matrix.
    - get_expected_matrix(): Calculate the expected coincidence matrix.
    - get_delta_matrix(): Calculate the delta matrix.
    - get_alpha(): Calculate Krippendorff's Alpha reliability estimate.
    - run_bootstrap(iterations): Perform bootstrap procedure to estimate Alpha (if enabled).
    - get_bootstrap_values(alpha_min, probabilities): Calculate confidence intervals and probabilities for alpha_min (if run_bootstrap is True).
    - print_data(): Display the results, including Alpha, confidence intervals, and matrices.
    """
    def __init__(self, x, method="nominal", run_bootstrap=False):
        self.x = x
        self.method = method
        self.if_run_bootstrap = run_bootstrap
        dimensions = np.shape(self.x)
        self.number_of_observers = dimensions[0]
        self.number_of_units = dimensions[1]

        self.unique_values = np.unique(self.x)
        self.unique_values_without_nan = sum(~np.isnan(self.unique_values))

        self.pairable_values = np.array([np.sum(~np.isnan(self.x)[:, column]) for column in range(self.number_of_units)])

        self.get_coincidence_matrix()
        self.get_expected_matrix()
        self.get_delta_matrix()
        self.get_alpha()

        if self.if_run_bootstrap:
            self.run_bootstrap()


    def get_coincidence_matrix(self):
        """
        Calculate the observed coincidence matrix.

        This method computes the observed coincidence matrix for the input data.
        """
        coincidence_matrix = np.zeros((self.unique_values_without_nan, self.unique_values_without_nan))
        for column in range(self.number_of_units):
            for index_1 in range(self.number_of_observers - 1):
                for index_2 in range(index_1 + 1, self.number_of_observers):
                    if not np.isnan(self.x[index_1, column]) and not np.isnan(self.x[index_2, column]):
                        judgement1 = np.where(self.unique_values == self.x[index_1, column])[0][0]
                        judgement2 = np.where(self.unique_values == self.x[index_2, column])[0][0]
                        coincidence_matrix[judgement1, judgement2] += (1 + (judgement1 == judgement2)) / (self.pairable_values[column] - 1)
                        if judgement1 != judgement2:
                            coincidence_matrix[judgement2, judgement1] = coincidence_matrix[judgement1, judgement2]
        self.coincidence_matrix = coincidence_matrix

    def get_expected_matrix(self):
        """
        Calculate the expected coincidence matrix.

        This method computes the expected coincidence matrix for the input data.
        """
        self.jugdments = np.sum(self.coincidence_matrix, axis=1)
        self.amount_of_jugdments = np.sum(np.sum(self.coincidence_matrix, axis=0))

        expected_matrix = np.zeros((self.unique_values_without_nan, self.unique_values_without_nan))
        for k in range(self.unique_values_without_nan):
            for c in range(self.unique_values_without_nan):
                expected_matrix[c, k] = self.jugdments[c] * (self.jugdments[k] - (c == k)) / (self.amount_of_jugdments - 1)
        self.expected_matrix = expected_matrix

    def get_delta_matrix(self):
        """
        Calculate the delta matrix.

        This method computes the delta matrix based on the measurement level of the data.
        """
        delta_matrix = np.zeros((self.unique_values_without_nan, self.unique_values_without_nan))
        for k in range(self.unique_values_without_nan):
            for c in range(self.unique_values_without_nan):
                if self.method == "nominal":
                    delta_matrix[c, k] = float(c != k)
                if self.method == "ordinal":
                    delta_matrix[c, k] = 0
                    if c <= k:
                        for g in range(c, k+1):
                            delta_matrix[c, k] += self.jugdments[g]
                    else:
                        for g in range(k, c+1):
                            delta_matrix[c, k] += self.jugdments[g]
                    delta_matrix[c, k] -= (self.jugdments[k] + self.jugdments[c]) / 2
                    delta_matrix[c, k] = delta_matrix[c, k] ** 2
                if self.method == "interval":
                    delta_matrix[c, k] = (self.unique_values[c] - self.unique_values[k]) ** 2
                if self.method == "ratio":
                    delta_matrix[c, k] = (self.unique_values[c] - self.unique_values[k]) ** 2 / (self.unique_values[c] + self.unique_values[k]) ** 2
        self.delta_matrix = delta_matrix

    def get_alpha(self):
        """
        Calculate Krippendorff's Alpha reliability estimate.

        This method computes Krippendorff's Alpha based on the observed and expected coincidence matrices.
        """
        self.observed_disagreement = np.sum(self.coincidence_matrix * self.delta_matrix)
        self.expected_disagreement = np.sum(self.expected_matrix * self.delta_matrix)
        self.alpha = 1 - self.observed_disagreement / self.expected_disagreement

    def run_bootstrap(self, iterations: int = 10000):
        """
        Perform bootstrap procedure to estimate Alpha.

        This method performs a bootstrap procedure to estimate Krippendorff's Alpha and store the results.
        """
        dimension_coincidence_matrix = self.coincidence_matrix.shape
        triangle_coincidence_matrix = self.coincidence_matrix[np.triu_indices(dimension_coincidence_matrix[0], k=0)]
        probability_coincidence_matrix = 2 * (self.coincidence_matrix / self.amount_of_jugdments)
        np.fill_diagonal(probability_coincidence_matrix, np.diag(self.coincidence_matrix) / self.amount_of_jugdments)
        probability_sum = 0
        ck = 0
        probability_matrix = np.zeros((2, len(triangle_coincidence_matrix)))
        for k in range(0, self.unique_values_without_nan):
            for c in range(k, self.unique_values_without_nan):
                probability_sum += probability_coincidence_matrix[c, k]
                probability_matrix[0, ck] = probability_sum
                probability_matrix[1, ck] = self.delta_matrix[c, k]
                ck += 1


        total_unique_pairs = sum((self.pairable_values - 1)*self.pairable_values)/2
        boot_alphas = np.empty(iterations)
        number_one = 0

        for iters in tqdm(range(1, iterations+1)):
            randoms = [random.uniform(0, 1) for _ in range(int(np.round(total_unique_pairs)))]
            number_sum = 0
            for i in range(len(randoms)):
                for j in range(1, probability_matrix.shape[1]):
                    if randoms[i] <= probability_matrix[0][j]:
                        if randoms[i] >= probability_matrix[0][j-1]:
                            number_sum += probability_matrix[1][j]

            alpha = 1 - (number_sum/((self.expected_disagreement/self.amount_of_jugdments)*total_unique_pairs))

            if alpha < -1:
                alpha = -1
            if alpha == 1 and sum(np.diagonal(self.coincidence_matrix) != 0) == 1:
                alpha = 0
            if alpha == 1 and sum(np.diagonal(self.coincidence_matrix) != 0) > 1:
                number_one += 1
            boot_alphas[iters-1] = alpha

        self.boot_alphas = boot_alphas
        self.boot_iterations = iterations

        self.get_bootstrap_values()

    def get_bootstrap_values(self, alpha_min: float = 0.80, probabilities=[0.025, 0.975]):
        """
        Calculate bootstrap values and confidence intervals for a statistic.

        This method calculates various statistics related to a bootstrap sampling distribution,
        including confidence intervals and probabilities.
        """
        assert alpha_min <= 1.0 and alpha_min > 0.0, 'alpha_min must be between 1.0 and 0.0'
        self.alpha_min = alpha_min
        quantiles = np.quantile(self.boot_alphas, probabilities)
        self.lower_confidence_level = quantiles[0]
        self.upper_confidence_level = quantiles[1]
        self.alpha_min_probability = sum(self.boot_alphas > self.alpha_min) / self.boot_iterations

        alpha_min_list = [0.9, 0.8, 0.7, 0.67, 0.6, 0.5]
        probabilities_for_alphas = []
        for alpha in alpha_min_list:
            probabilities_for_alphas.append(1 - (sum(self.boot_alphas > alpha) / self.boot_iterations))

        self.alpha_min_list = alpha_min_list
        self.probailities_for_alphas = probabilities_for_alphas
        
    def print_data(self):
        """
        Display the results, including Alpha, confidence intervals, and matrices.

        This method prints the computed values, observed and expected matrices, and additional statistics.
        """
        print_dataframe = pd.DataFrame()
        print_dataframe['Alpha'] = [round(self.alpha, 4)]
        print_dataframe['LL95%CI'] = [round(self.lower_confidence_level, 4)]
        print_dataframe['UL95%CI'] = [round(self.upper_confidence_level, 4)]
        print_dataframe['Units'] = [self.number_of_units]
        print_dataframe['Observers'] = [self.number_of_observers]
        print_dataframe['Pairs'] = [sum(sum(range(1, value)) for value in self.pairable_values)]
        print_dataframe.index = [self.method.capitalize()]
        probabilities_dataframe = pd.DataFrame()
        probabilities_dataframe['alphamin'] = self.alpha_min_list
        probabilities_dataframe['q'] = self.probailities_for_alphas
        print("Krippendorff's Alpha Reliability Estimate\n")
        print(print_dataframe)
        print("\nProbability (q) of failure to achieve an alpha of at least alphamin:")
        print(probabilities_dataframe.round(4).to_string(index=False))
        print("\nNumber of bootstrap samples:\n {}".format(self.boot_iterations))
        print("\nJudges used in these computations:")
        print("obsa obsb obsc obsd")
        print("=" * 50)
        print("\nObserved Coincidence Matrix")
        print(pd.DataFrame(self.coincidence_matrix.round(2)).to_string(index=False, header=False))
        print("\nExpected Coincidence Matrix")
        print(pd.DataFrame(self.expected_matrix.round(2)).to_string(index=False, header=False))
        print("\nDelta Matrix")
        print(pd.DataFrame(self.delta_matrix.round(2)).to_string(index=False, header=False))
        print("\nRows and columns correspond to following unit values")
        print(*self.unique_values[~np.isnan(self.unique_values)])

