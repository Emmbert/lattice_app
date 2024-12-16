import numpy as np
import matplotlib.pyplot as plt

# Funktion zur Berechnung der Parameter
def calculate_parameters(lambda_GSW, epsilon_GSW, tau_GSW):
    n_GSW = int(max(lambda_GSW, np.ceil(pow((4/epsilon_GSW) * tau_GSW * np.log2(tau_GSW), 1/epsilon_GSW))))
    q_GSW = int(np.ceil(pow(2, pow(n_GSW, epsilon_GSW))))
    #print(q_GSW)
    #print(type(q_GSW))
    #print(f"n_GSW: {n_GSW}, Typ: {type(n_GSW)}")
    #print(np.log2(float(q_GSW)))
    m_GSW = int(1 + np.ceil(2 * n_GSW * (2 + np.log2(float(q_GSW)))))
    alpha_GSW = 1 #n_GSW / q_GSW
    l_GSW = int(np.ceil(np.log2(float(q_GSW))))
    N_GSW = n_GSW * l_GSW
    return n_GSW, q_GSW, m_GSW, alpha_GSW, l_GSW, N_GSW

# Funktion zur Prüfung der Bedingungen
def check_conditions(n_GSW, q_GSW, m_GSW, alpha_GSW, l_GSW, N_GSW, epsilon_GSW, tau_GSW):
    results = []
    results.append(q_GSW >= 2 ** ((n_GSW - 1) ** epsilon_GSW))
    results.append(m_GSW >= (n_GSW - 1) * np.log2(float(q_GSW)))
    results.append(pow(n_GSW, epsilon_GSW) > 2 * tau_GSW * np.log2(n_GSW))
    results.append(m_GSW >= 1 + 2 * n_GSW * (2 + np.log2(float(q_GSW))))
    results.append(n_GSW * m_GSW * pow(N_GSW + 1, tau_GSW + 1) < q_GSW / 4)
    return all(results)


def plot_parameters():
    lambda_values = np.arange(20, 220, 5)
    epsilon_GSW = 0.8
    tau_GSW = 1

    # Parameter berechnen und Bedingungen prüfen
    conditions_met = []
    for lambda_GSW in lambda_values:
        n_GSW, q_GSW, m_GSW, alpha_GSW, l_GSW, N_GSW = calculate_parameters(lambda_GSW, epsilon_GSW, tau_GSW)
        conditions_met.append(check_conditions(n_GSW, q_GSW, m_GSW, alpha_GSW, l_GSW, N_GSW, epsilon_GSW, tau_GSW))

    # Plots
    plt.figure(figsize=(12, 8))

    # Plot der Parameter
    n_values, q_values, m_values, alpha_values, l_values, N_values = zip(
        *[calculate_parameters(l, epsilon_GSW, tau_GSW) for l in lambda_values])
    plt.plot(lambda_values, n_values, label=r"$n_{\text{GSW}}$", marker="o")
    plt.plot(lambda_values, q_values, label=r"$q_{\text{GSW}}$", marker="s")
    plt.plot(lambda_values, m_values, label=r"$m_{\text{GSW}}$", marker="^")
    plt.plot(lambda_values, alpha_values, label=r"$\alpha_{\text{GSW}}$", marker="v")
    plt.plot(lambda_values, l_values, label=r"$l_{\text{GSW}}$", marker="*")
    plt.plot(lambda_values, N_values, label=r"$N_{\text{GSW}}$", marker="x")

    # Markiere Bereiche, in denen die Bedingungen erfüllt sind
    for i, (lambda_GSW, condition) in enumerate(zip(lambda_values, conditions_met)):
        if condition:
            plt.axvspan(lambda_GSW - 2.5, lambda_GSW + 2.5, color='lightgreen', alpha=0.3,
                        label="Conditions Met" if i == 0 else "")

    # Labels und Titel
    plt.xlabel(r"$\lambda_{\text{GSW}}$")
    plt.ylabel("Parameterwerte")
    plt.title("Parameter in Abhängigkeit von $\lambda_{\text{GSW}}$ mit markierten Bereichen")
    plt.legend()
    plt.yscale("log")  # Optional: Logarithmic scale for y-axis to better see differences in growth
    plt.grid(True, linestyle="--", alpha=0.6)

    # Plot anzeigen
    plt.show()


class Parameters:
    def __init__(self, q, n, m, alpha, bits):
        self.q = q
        self.n = n
        self.m = m
        self.alpha = alpha
        self.bits = bits
        l = int(np.ceil(np.log2(float(q))))
        self.N = n * l
