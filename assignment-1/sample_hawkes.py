#!/usr/bin/env python
import click
import numpy as np


@click.command()
@click.argument('mu', type=float)
@click.argument('a', type=float)
@click.argument('w', type=float)
@click.argument('T', type=float)
@click.option('--max-events', 'max_events', default=1000000, help='Maximum number of events to generate before sampling termination.')
@click.option('--N', 'N', default=1, help='Number of time sequences to generate.', show_default=True)
@click.option('--seed', 'seed', help='Initial seed. A random sequence is generated if seed is -1.', default=-1, show_default=True)
def run(mu, a, w, t, N, seed, max_events):
    """Generates a sample from Hawkes process with the given MU, A, and W parameters.

    A sequence upto time T is generated, unless max-events are reached, in which case the
    program stops with an error."""

    T = t

    if seed < 0:
        seed = np.random.randint(10000) + 1

    for i in range(N):
        print(','.join([str(x)
                        for x in sample_hawkes(mu=mu, a=a, w=w, T=T,
                                               seed=seed + i,
                                               max_events=max_events)]))


def sample_hawkes(mu, a, w, T, seed, max_events):
    """Generate a list of sample drawn from Hawkes process."""
    # Assert that alpha is less than w otherwise will explode.
    assert a/w < 1
    # Initialization
    t_ev = np.array([0])
    np.random.seed(seed)
    # Sampling first point from a Poisson.
    lmb_max = mu
    n = 1
    u = np.random.uniform()
    s_new = -(1/lmb_max) * np.log(u)
    if s_new <= T:
        t_ev = np.append(t_ev, s_new) # t_1
    else:
        return np.array([])
    
    # General routine
    while n < max_events:
        n += 1
        # Update the maximum intensity
        lmb_1 = mu + a * np.sum(np.exp(-w * (t_ev[n-1] - t_ev[1:-1])))
        lmb_max = a + lmb_1
        while True:
            # New Event
            u1 = np.random.uniform()
            s_new = s_new - (1/lmb_max) * np.log(u1)
            if s_new >= T:
                break
            else:
                # Rejection Test
                u2 = np.random.uniform()
                lmb_2 = mu + a * np.sum(np.exp(-w * (s_new - t_ev[1:])))
                if u2 <= (lmb_2 / lmb_max):
                    # Successful sample. Append.
                    t_ev = np.append(t_ev, s_new)
                    break
                else:
                    # Not a proper sample. Update lambda_max and try again.
                    lmb_max = lmb_2
        if s_new >= T:
            break
    # Return the samples.
    return t_ev[1:]


if __name__ == '__main__':
    run()
