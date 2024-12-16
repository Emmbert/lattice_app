import math
import random


class Dgi():
    """Discrete Gaussian sampler over the integers mod q."""

    def __init__(self, q, sigma=1.0, tau=6.0, mod=True):
        """Construct a new sampler for a discrete Gaussian distribution.

        Args:
            q (int): modulus, an integer greater than 1
            sigma (float): standard deviation
            tau (float): samples outside the range [-tau * sigma, tau * sigma]
            are considered to have probability zero.
        """
        # Centre c is always 0.
        self.q = q
        self.sigma = sigma
        self.mod = mod
        # Compute scale and integer bound
        self.scale = 1 / (sigma * math.sqrt(2 * math.pi))
        self.bound = math.floor(tau * sigma)
        self.fmax = self.f(0)
        # Set up table in integer range [-bound,bound)
        # NB bound+1 for range() upper limit
        self.tab = [self.f(x)
                    for x in range(-self.bound, self.bound + 1)]

    def f(self, x):
        """Gaussian probability density function, ``f(x)``."""
        return self.scale * math.exp(-x * x / (2 * self.sigma * self.sigma))

    def D(self):
        """Return a sample in the range [0,q-1]."""
        # Use rejection sampling
        '''
        do {
           select integer x in range [-bound, bound)
           select y in range [0.0, fmax)
        } while y > f(x)
        return x mod q
        '''
        while True:
            # NB randint(a,b) has range(a, b+1)
            x = random.randint(-self.bound, self.bound)
            # print("x",x)
            y = random.random() * self.fmax
            # print(self.fmax)
            # print("y",y)
            # if y > self.f(x):
            # print(self.tab)
            # print(x + self.bound)
            # print(self.tab[x + self.bound])
            if y > self.tab[x + self.bound]:
                continue
            else:
                break
        # Return x mod q
        if self.mod == False:
            return x
        # print(x + self.q if x < 0 else x)
        return x + self.q if x < 0 else x
