import numpy as np

from GSW.distributions import Dgi
from GSW.parameters import Parameters


def chi(q, alpha, m=1, mod=True):
    sigma = (q*alpha) / 5
    #print("sigma = ", sigma)
    tau = 4 * sigma
    assert (sigma < q * alpha / 4) # My conversion from the bound |x| < alpha * q to the sigma value of the discrete gaussian.

    dgi = Dgi(q, sigma, tau=tau, mod=mod)
    if m == 1:
        return dgi.D()
    return np.array([dgi.D() for i in range(0, m)])


def psi(parameters):
    q = parameters.q
    n = parameters.n
    m = parameters.m
    alpha = parameters.alpha

    A = np.random.randint(0, q, size=(n-1, m))
    s = np.random.randint(0, q, n-1).transpose()
    noise = chi(q, alpha, m, mod=False).transpose()
    b = s.dot(A) + noise

    A_ = np.vstack([A, b])
    s_ = np.append(s, -1)
    return A_, s_


# Falttening Gadget Matrix
def get_g(q):
    l = int(np.ceil(np.log2(float(q))))
    #l = int(np.ceil(np.log2(q)))
    g = 2 ** np.arange(l)[::-1]
    return g


def g_inv(x, q):
    l = int(np.ceil(np.log2(float(q))))
    return np.array([(x >> i) & 1 for i in range(l - 1, -1, -1)])


def get_G(q, n):
    l = int(np.ceil(np.log2(float(q))))
    return np.kron(np.eye(n, dtype="int"), 2 ** np.arange(l)[::-1])


def G_inv(x, q):
    if len(x.shape) == 1:
        return np.transpose(np.concatenate([g_inv(val, q) for val in x]))
    else:
        A = x
        transformed_columns = []
        for col in range(A.shape[1]):
            transformed_col = G_inv(A[:, col], q)
            transformed_columns.append(transformed_col)

        # Stack the transformed columns horizontally to form the new matrix
        A_new = np.column_stack(transformed_columns)
        return A_new

def l_inf_norm(x, q=None):
    if q:
        x = [min(x_i, q-x_i) for x_i in x]
        return max(x)
    return max(x)


def my_round(x, q, bits):
    r1 = (round(x / (q / 2**bits)))
    r2 = round(r1 % 2**bits)
    return r2


def key_gen_GSW(parameters: Parameters, psi):
    q = parameters.q
    n = parameters.n
    while True:
        A,s = psi(parameters)
        if (s != 0).any():
            if (l_inf_norm((s.dot(A)%q), q) < n).all():
                break
    return s, A


def enc_GSW(b, A, parameters: Parameters, log=False):
    q = parameters.q
    n = parameters.n
    m = parameters.m
    bits = parameters.bits
    N = parameters.N

    R = np.random.randint(0, 2, size=(m, N))
    G = get_G(q, n)
    G_encoded = (np.rint(b*(2/2**bits) * G)).astype(int)
    C = (G_encoded + A.dot(R)) % q
    if log:
        print("A =  {} ({})".format(A, A.shape))
        print("R =  {} ({})".format(R, R.shape))
        print("b*G =  {} ({})".format(b * G, G.shape))
        print("C =  {} ({})".format(C, C.shape))

    return C


def dec_GSW(C, s, parameters: Parameters, log=False, noise=False):
    q = parameters.q
    n = parameters.n
    bits = parameters.bits
    s_x_C = s.dot(C) % q
    w = np.zeros(n, dtype=int)  # Ein Vektor mit N Nullen
    w[-1] = -1*(q // 2)
    # w[-1] = 2**(l-1) -1
    G_w = G_inv(w, q)
    b_x_mu = s_x_C.dot(G_w) % q
    rounded_value = my_round(b_x_mu, q, bits)
    b = (2**bits - rounded_value) % 2**bits
    if log:
        print("C =  {} ({})".format(C, C.shape))
        print("s =  {} ({})".format(s, s.shape))
        print("s*C", s.dot(C) % q)
        print("w =  {} ({})".format(w, w.shape))
        print("G^-1(w) =  {} ({})".format(G_w, G_w.shape))
        print("s*C*G^-1(w) = ", b_x_mu)
    if noise:
        return b, b_x_mu
    return b


def XOR(C1, C2, parameters: Parameters):
    q = parameters.q
    return (C1 + C2) % q


def AND(C1, C2, parameters: Parameters, log=False):
    q = parameters.q
    if log:
        G_inv_C2 = G_inv(C2, q)
        print("G_inv(C2) =  \n{} ({})".format(G_inv_C2, G_inv_C2.shape))
    return C1.dot(G_inv(C2, q)) % q


if __name__ == '__main__':
    params = Parameters(1024, 10, 20, 0.004, 1)
    q = params.q
    n = params.n
    m = params.m
    alpha = params.alpha

    # Generate Keys
    s, A = key_gen_GSW(params, psi)
    print("s = ", s)
    print("A = ", A)
    print("noise = {}".format(s.dot(A)%q))
    print("|noise|∞ = {} < n = {}".format(l_inf_norm((s.dot(A)%q), q), n))

    b = 0  # np.random.randint(0,2)
    print("b = ", b)

    C = enc_GSW(b, A, params, True)
    b_ = dec_GSW(C, s, params, True)

    print("{} =?= {}".format(b, b_))

    # XOR

    b1 = 1
    b2 = 0
    C1 = enc_GSW(b1, A, params, True)
    print("C1 =  \n{} ({})".format(C1, C1.shape))
    C2 = enc_GSW(b2, A, params, True)
    print("C2 =  \n{} ({})".format(C2, C2.shape))

    C_XOR = XOR(C1, C2, params)
    print("C_XOR =  \n{} ({})".format(C_XOR, C_XOR.shape))

    b_XOR = dec_GSW(C_XOR, s, params, True)
    print("b1^b2 = {} ?= {}".format(b1 ^ b2, b_XOR))
    assert (b1 ^ b2 == b_XOR)

    # AND
    b1 = 1
    b2 = 1
    C1 = enc_GSW(b1, A, params, True)
    print("C1 =  \n{} ({})".format(C1, C1.shape))
    C2 = enc_GSW(b2, A, params, True)
    print("C2 =  \n{} ({})".format(C2, C2.shape))

    C_AND = AND(C1, C2, params, log=True)
    print("C_AND =  \n{} ({})".format(C_AND, C_AND.shape))

    b_AND = dec_GSW(C_AND, s, params, True)
    print("b1&b2 = {} ?= {}".format(b1 & b2, b_AND))
    assert (b1 & b2 == b_AND)
