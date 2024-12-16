import numpy as np

from GSW.algorithms import key_gen_GSW, enc_GSW, dec_GSW, g_inv, G_inv, get_g, get_G, psi, l_inf_norm, XOR, AND
from GSW.parameters import Parameters


def test_gadget_matrix(parameters: Parameters, num_tests=1000):

    q = parameters.q
    n = parameters.n
    m = parameters.m

    for i in range(num_tests):
        x = np.random.randint(0, q)
        g = get_g(q)
        assert (g.dot(g_inv(x, q)) == x)
    print("All {} tests were successful.".format(num_tests))

    for i in range(num_tests):
        x = np.random.randint(0, q, n)
        G = get_G(q, n)
        assert ((G.dot(G_inv(x, q)) == x).all())
    print("All {} tests were successful.".format(num_tests))

    for i in range(num_tests):
        x = np.random.randint(0, q, (n, m))
        assert ((G.dot(G_inv(x, q)) == x).all())
    print("All {} tests were successful.".format(num_tests))


def test_psi(parameters: Parameters, num_tests=1000):
    q = parameters.q
    alpha = parameters.alpha

    for i in range(num_tests):
        A, s = psi(parameters)
        noise = s.dot(A)
        #print(l_inf_norm(noise))
        assert (l_inf_norm(noise) < q*alpha)

        # print(s.dot(u) % q)
    print("All {} tests successfull.".format(num_tests))


def test_decryption_correctness(parameters, num_test=1000):
    # Test GSW decryption correctness
    bits = parameters.bits
    for i in range(num_test):
        if i % 100 == 0:
            print(i, end=" ")
        b = np.random.randint(0, 2**bits)
        s, A = key_gen_GSW(parameters, psi)
        C = enc_GSW(b, A, parameters, log=False)
        b_ = dec_GSW(C, s, parameters, log=False)
        assert (b == b_)
    print("\nAll {} tests successfull.".format(num_test))


def test_addition_correctness(parameters, num_test=1000):
    bits = parameters.bits
    for i in range(num_test):
        if i % 100 == 0:
            print(i, end=" ")
        s, A = key_gen_GSW(parameters, psi)
        b1 = np.random.randint(0, 2**bits)
        b2 = np.random.randint(0, 2**bits)
        C1 = enc_GSW(b1, A, parameters, log=False)
        C2 = enc_GSW(b2, A, parameters, log=False)
        C_XOR = XOR(C1, C2, parameters)

        b_XOR = dec_GSW(C_XOR, s, parameters, log=False)
        assert ((b1 + b2) % 2**bits == b_XOR)
    print("All {} tests successfull.".format(num_test))


def test_multiplication_correctness(parameters, num_test=1000):
    bits = parameters.bits
    for i in range(num_test):
        # print(i)
        if i % 100 == 0:
            print(i, end=" ")
        s, A = key_gen_GSW(parameters, psi)
        b1 = np.random.randint(0, 2 ** bits)
        b2 = np.random.randint(0, 2 ** bits)
        C1 = enc_GSW(b1, A, parameters, log=False)
        C2 = enc_GSW(b2, A, parameters, log=False)

        C_AND = AND(C1, C2, parameters)

        b_AND = dec_GSW(C_AND, s, parameters, log=False)

        # print("b1&b2 = {} & {} = {} ?= {}".format(b1, b2, b1&b2, b_AND))
        assert ((b1 * b2) % 2**bits == b_AND)
    print("All {} tests successfull.".format(num_test))


if __name__ == '__main__':
    params = Parameters(1024, 10, 20, 0.003, 1)

    #test_gadget_matrix(params)
    #test_psi(params)
    test_decryption_correctness(params)
    test_addition_correctness(params)
    test_multiplication_correctness(params)
