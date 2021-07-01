import numpy as np
import math
import matplotlib.pyplot as plt


def nll(pdf, measurements, *theta):
    """
    The negative logarithmic likelihood for measurements given
    pdf(x, theta).

    It is used as the cost function for numerical minimisation to estimate the
    optimal values for theta

    Args:
        pdf (callable): The probability density function that describes the
            measurements
        measurements (list of ints/list of lists): the measurements that
            where taken from either monte carlo generators or from actual
            experiments, can be multi-dimensional
        theta (list): parameters of the probability density function,
            they are normally estimated
        using the minimisation of the nll

    Returns:
        nll (float): the value of the negative logarithmic likelihood of
            the measurements using pdf(x, *theta) as hypothesized
            probability density function
    """
    probabilities = [pdf(m, *theta) for m in measurements]
    return np.sum(-np.log(np.array(probabilities)))

# an example using the nll to fit a pdf
# first of all a pdf is needed to generate random numbers with
# x is a single random vector drawn from X with the pdf 'true_pdf'
def pdf(x, A, theta1, theta2):
    return A * math.sin(x-theta1)**2 * np.exp(-(x-theta2)**2)

# the pdf needs to be normalized, for this an integration is needed, that
# is performed analytically here
def integrate(func, domain, *params):
    """
    uses random numbers to estimate the integral of an arbitrary pdf
    """
    rnd_cnt = 100000
    mind = min(domain)
    maxd = max(domain)
    dmn = np.linspace(mind, maxd, 10000)
    maxy = max([func(e, *params) for e in dmn])
    rndx = np.random.rand(rnd_cnt)*(maxd-mind)+mind
    rndy = np.random.rand(rnd_cnt)*maxy
    accepted = sum([1 if func(x, *params) <= y else 0
                   for x, y in zip(rndx, rndy)])
    return (maxd-mind)*maxy*(accepted/rnd_cnt)

def generate_arbitrarily_distributed_rngs(pdf, count, domain):
    mind = min(domain)
    maxd = max(domain)
    dmn = np.linspace(mind, maxd, 10000)
    maxy = max([pdf(e) for e in dmn])

    round_size = int(count*0.01)
    output = []
    while len(output) <= count:
        xs = np.random.rand(round_size) * (maxd-mind) + mind
        ys = np.random.rand(round_size) * maxy
        for x, y in zip(xs, ys):
            if y <= pdf(x):
                output.append(x)
        print(f'generating random numbers: {np.round(len(output)/count*100, 1)}% complete ...', end='\r')
    return output

# generate random numbers distributed according to ture_pdf
domain = (-10, 10)
area = integrate(pdf, domain, *(1, 1, 1))
true_pdf = lambda x: pdf(x, 1, 2, 3)
measurements = generate_arbitrarily_distributed_rngs(true_pdf, 2000, domain)

# now we use the nll to show the parameter estimation capability
theta1 = np.linspace(1, 3, 40)
theta2 = np.linspace(2, 4, 40)
theta1, theta2 = np.meshgrid(theta1, theta2)
likelyhood_scan = np.array([nll(pdf, measurements, *(1/area, t1, t2))
                            for t1, t2 in zip(theta1.flatten(),
                                              theta2.flatten())]).reshape(
                                                      theta1.shape)
plt.pcolormesh(theta1, theta2, likelyhood_scan, shading='nearest')
plt.xlabel(r'\theta_1')
plt.ylabel(r'\theta_2')
plt.show()
opt_t1 = theta1.flatten()[np.argmin(likelyhood_scan)]
opt_t2 = theta2.flatten()[np.argmin(likelyhood_scan)]

pltx = np.linspace(min(domain), max(domain), 1000)
plt.hist(measurements, 100, color='blue', alpha=0.1, edgecolor='black', density=True)
plt.plot(pltx, [pdf(x, 1, opt_t1, opt_t2) for x in pltx])
plt.show()
