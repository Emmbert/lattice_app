import matplotlib.pyplot as plt

import numpy as np

from GSW.algorithms import chi, key_gen_GSW, enc_GSW, psi, dec_GSW, XOR, AND
from GSW.parameters import Parameters


def plot_chi(parameters: Parameters, size=1000):
    q = parameters.q
    n = parameters.n
    m = parameters.m
    alpha = parameters.alpha

    samples = [chi(q, alpha, mod=False) for x in range(size)]

    # Plotten
    plt.figure(figsize=(10, 6))
    # Balkendiagramme
    #bins = np.concatenate([np.arange(0, tau, 1), np.arange(q - tau, q + 1, 1)])
    bins = np.arange(-q/2, q/2, 1)
    counts, edges = np.histogram(samples, bins=bins)
    bin_centers = 0.5 * (edges[1:] + edges[:-1]) - 0.5
    bars = plt.bar(bin_centers, counts / size, width=0.8, edgecolor='black', color='skyblue')

    # Beschriftungen
    plt.xlabel("x")
    plt.ylabel("Wahrscheinlichkeitsdichte")
    # TODO caution hardcoded! from GSW.algorithms.chi
    sigma = (q*alpha) / 4
    plt.title("Discrete Gaussian Distribution over $\mathbb{{Z}}_{{{}}}$ with $\\alpha = {}, \\sigma = {}$ ".format(q, alpha, sigma))
    #plt.legend()
    plt.grid(True)
    plt.show()


# Plot noise distribution during decryption
def plot_decryption_noise(parameters: Parameters, num_tests=1000):
    q = parameters.q
    n = parameters.n
    m = parameters.m
    bits = parameters.bits
    alpha = parameters.alpha
    samples = []
    for i in range(num_tests):
        s, A = key_gen_GSW(parameters, psi)
        b = np.random.randint(0, 2**bits)
        C = enc_GSW(b, A, parameters)
        b_, noise = dec_GSW(C, s, parameters, noise=True)
        samples.append(noise)

    plt.figure(figsize=(10, 6))
    plt.hist(samples, bins=int(q / 1), color='skyblue', edgecolor='darkblue')
    # plot decision borders q/4
    plt.axvline(x=3 * q / 4, color='violet', linestyle='--', linewidth=1.5,
                label="rounding border of $b$ at $q/4 = {}$".format(int(q / 4)))
    plt.axvline(x=q / 4, color='violet', linestyle='--', linewidth=1.5)

    # Add labels and title
    plt.xlabel("Values of $s*C*G^{-1}(w)$")
    plt.ylabel("Frequency")
    plt.title("Noise Distribution after Decryption of fresh Ciphertext ({} Samples)\n over $\mathbb{{Z}}_{{{}}}$ with $n = {}, m = {} \\alpha = {}$".format(num_tests, q, n, m, alpha))
    # plt.legend()
    plt.grid(True)

    plt.show()


def plot_XOR_noise(parameters: Parameters, num_tests=1000):
    # Plot noise distribution after XOR during decryption
    # Test GSW decryption correctness
    q = parameters.q
    n = parameters.n
    m = parameters.m
    bits = parameters.bits
    alpha = parameters.alpha
    samples = []
    for i in range(num_tests):
        s, A = key_gen_GSW(parameters, psi)
        b1 = np.random.randint(0, 2 ** bits)
        b2 = np.random.randint(0, 2 ** bits)
        C1 = enc_GSW(b1, A, parameters, log=False)
        C2 = enc_GSW(b2, A, parameters, log=False)

        C_XOR = XOR(C1, C2, parameters)

        b_XOR, noise = dec_GSW(C_XOR, s, parameters, log=False, noise=True)
        samples.append(noise)

    plt.figure(figsize=(10, 6))
    plt.hist(samples, bins=int(q / 1), color='skyblue', edgecolor='darkblue')
    # plot decision borders q/4
    plt.axvline(x=3 * q / 4, color='violet', linestyle='--', linewidth=1.5,
                label="rounding border of $b$ at $q/4 = {}$".format(int(q / 4)))
    plt.axvline(x=q / 4, color='violet', linestyle='--', linewidth=1.5)

    # Add labels and title
    plt.xlabel("Values of $s*C*G^{-1}(w)$")
    plt.ylabel("Frequency")
    plt.title(
        "Noise Distribution after Decryption of XOR-evaluated Ciphertext ({} Samples)\n over $\mathbb{{Z}}_{{{}}}$ with $n = {}, m = {}, \\alpha = {}$ and {} bit message".format(
            num_tests, q, n, m, alpha, bits))
    # plt.legend()
    plt.grid(True)
    plt.show()


def plot_AND_noise(parameters: Parameters, num_tests=1000):
    # Plot noise distribution after XOR during decryption
    # Test GSW decryption correctness
    q = parameters.q
    n = parameters.n
    m = parameters.m
    bits = parameters.bits
    alpha = parameters.alpha
    samples = []
    for i in range(num_tests):
        s, A = key_gen_GSW(parameters, psi)
        b1 = np.random.randint(0, 2 ** bits)
        b2 = np.random.randint(0, 2 ** bits)
        C1 = enc_GSW(b1, A, parameters, log=False)
        C2 = enc_GSW(b2, A, parameters, log=False)

        C_AND = AND(C1, C2, parameters)

        b_AND, noise = dec_GSW(C_AND, s, parameters, log=False, noise=True)
        samples.append(noise)

    plt.figure(figsize=(10, 6))
    plt.hist(samples, bins=int(q / 1), color='skyblue', edgecolor='darkblue')
    # plot decision borders q/4
    plt.axvline(x=3 * q / 4, color='violet', linestyle='--', linewidth=1.5,
                label="rounding border of $b$ at $q/4 = {}$".format(int(q / 4)))
    plt.axvline(x=q / 4, color='violet', linestyle='--', linewidth=1.5)

    # Add labels and title
    plt.xlabel("Values of $s*C*G^{-1}(w)$")
    plt.ylabel("Frequency")
    plt.title(
        "Noise Distribution after Decryption of AND-evaluated Ciphertext ({} Samples)\n over $\mathbb{{Z}}_{{{}}}$ with $ n = {}, m = {}, \\alpha = {}$ and {} bit message".format(
            num_tests, q, n, m, alpha, bits))
    # plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    params = Parameters(16384, 24, 50, 0.0004, 1)

    #plot_chi(params)
    #plot_decryption_noise(params)
    #plot_XOR_noise(params)
    plot_AND_noise(params)
