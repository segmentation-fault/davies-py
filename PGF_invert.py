#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

from mpmath import *
import matplotlib.pyplot as plt
from typing import Callable

mp.dps = 15


def invert_pgf(G: Callable[[mpf], mpf], k: int) -> mpf:
    """
    Numerically inverts the PGF G, returning the value of the CDF in k,
    according to Davies, Robert B. "Numerical inversion of a characteristic function." Biometrika 60.2 (1973): 415-417.
    :param G: PGF
    :param k: where to calculate the CDF of G
    :return: the value of the CDF at the point specified by k
    """
    x = k + 1
    my_integrand = lambda t: re(G(exp(1j * t)) * exp(-1j * t * x) / (2.0 * pi * (1.0 - exp(-1j * t))))
    my_integral = quad(my_integrand, [-pi, pi])
    return 0.5 - my_integral


def binomial_PMF(p, n, k):
    if p < 0 or p > 1 or n < 0 or k < 0 or k > n:
        raise (ValueError)

    q = 1 - p

    return exp(loggamma(n + 1) - loggamma(k + 1) - loggamma(n - k + 1)) * p ** k * q ** (n - k)


def binomial_CDF(p, n, k):
    if p < 0 or p > 1 or n < 0 or k < 0 or k > n:
        raise (ValueError)

    return nsum(lambda k: binomial_PMF(p, n, k), [0, k])


def binomial_PGF(p, n, z):
    if p < 0 or p > 1 or n < 0:
        raise ValueError

    q = 1 - p

    return (q + p * z) ** n


def neg_bin_PMF(r, m, k):
    if r < 0 or k < 0 or m < 0:
        raise ValueError

    p = m / (r + m)

    return exp(loggamma(k + r) - loggamma(r) - loggamma(k + 1.0)) * p ** k * (1.0 - p) ** r


def neg_bin_CDF(r, m, k):
    if r < 0 or k < 0 or m < 0:
        raise ValueError

    return nsum(lambda k: neg_bin_PMF(r, m, k), [0, k])


def neg_bin_PGF(r, m, z):
    if r < 0 or m < 0:
        raise ValueError

    p = m / (r + m)

    return ((1.0 - p) / (1.0 - p * z)) ** r


def main():
    # Example comparing the real CDF of the binomial and negative binomial distribution vs the values found by inversion

    # Binomial
    n = 20
    p = 0.5
    KB = arange(0, n + 1)

    PB0 = []
    PB = []
    for k in KB:
        PB0.append(binomial_CDF(p, n, k))
        PB.append(invert_pgf(lambda z: binomial_PGF(p, n, z), k))

    # Neg Exp
    r = 40
    m = 10
    KNB = arange(0, 25)

    PNB0 = []
    PNB = []

    for k in KNB:
        PNB0.append(neg_bin_CDF(r, m, k))
        PNB.append(invert_pgf(lambda z: neg_bin_PGF(r, m, z), k))

    plt.figure()
    plt.step(KB, PB0, label="Binomial - Original")
    plt.step(KB, PB, label="Binomial - Inverted")
    plt.legend(fontsize=22)

    plt.figure()
    plt.step(KNB, PNB0, label="Negative Binomial - Original")
    plt.step(KNB, PNB, label="Negative Binomial - Inverted")
    plt.legend(fontsize=22)

    plt.show()


if __name__ == "__main__":
    main()
