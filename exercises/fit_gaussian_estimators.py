from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    univariate_gaussian = UnivariateGaussian()
    expectation, standard_deviation = 10, 1
    normal_samples = np.random.normal(expectation, standard_deviation, 1000)
    univariate_gaussian.fit(normal_samples)
    print(f"({univariate_gaussian.mu_}, {univariate_gaussian.var_})")

    # Question 2 - Empirically showing sample mean is consistent
    sample_sizes = np.arange(10, 1001, 10)
    expectations_diffs = [np.abs(univariate_gaussian.fit(
        np.random.normal(expectation, standard_deviation,
                         samples_size)).mu_ - expectation) for
                          samples_size in sample_sizes]
    go.Figure(go.Scatter(x=sample_sizes, y=expectations_diffs,
                         mode='markers+lines'), layout=go.Layout(
        title=r"$\text{Distance between Estimated and True Value "
              r"of Expectation As Function Of Sample Size}$",
        xaxis_title=r"$\text{Sample Size}$",
        yaxis_title=r"$\text{Distance}$", height=300)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    normal_samples.sort()
    pdf = univariate_gaussian.pdf(normal_samples)
    go.Figure(go.Scatter(x=normal_samples, y=pdf,
                         mode='markers+lines'), layout=go.Layout(
        title=r"$\text{Empirical PDF under the fitted model}$",
        xaxis_title=r"$\text{Sample}$",
        yaxis_title=r"$\text{Density}$", height=300)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1, 0.2, 0, 0.5],
                      [0.2, 2, 0, 0],
                      [0, 0, 1, 0],
                      [0.5, 0, 0, 1]])
    multivariate_gaussian = MultivariateGaussian()
    X = np.random.multivariate_normal(mu, sigma, 1000)
    multivariate_gaussian.fit(X)
    print(multivariate_gaussian.mu_)
    print(multivariate_gaussian.cov_)

    # Question 5 - Likelihood evaluation
    f1 = f3 = np.linspace(-10, 10, 200)
    log_likelihoods = np.array([])
    for val1 in f1:
        row = np.array([])
        for val3 in f3:
            row = np.append(row, multivariate_gaussian.log_likelihood(
                np.array([val1, 0, val3, 0]), multivariate_gaussian.cov_, X))
        log_likelihoods = np.append(log_likelihoods, row)
    log_likelihoods = log_likelihoods.reshape(200, 200)
    go.Figure(go.Heatmap(x=f1, y=f3, z=log_likelihoods), layout=go.Layout(
        title=r"$\text{Log-Likelihood As Function of f1 Values and f3 "
              r"Values}$",
        xaxis_title=r"$\text{f1 Value}$",
        yaxis_title=r"$\text{f3 Value}$", height=300)).show()

    # Question 6 - Maximum likelihood
    arg_max = np.argmax(log_likelihoods)
    idxs_arg_max = np.unravel_index(arg_max, (200, 200))
    f1_val_arg_max_likelihood = f1[idxs_arg_max[0]]
    f3_val_arg_max_likelihood = f3[idxs_arg_max[1]]
    print(np.round(f1_val_arg_max_likelihood, 3),
          np.round(f3_val_arg_max_likelihood), 3)


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
