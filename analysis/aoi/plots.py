import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .parameters import *
from ..util import convert_rate

RATES = [1, 10, 100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]

def plot_time_evolution(title, aoi_x, aoi_y, mean, median, **kwargs) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(20, 5))
    ax.set_title(title)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('AoI [ms]')
    ax.set_yscale(kwargs.get('xscale', 'linear'))
    ax.set_yscale(kwargs.get('yscale', 'linear'))
    ax.plot(aoi_x, aoi_y, label='AoI')

    if mean is not None:
        ax.plot([0, aoi_x[-1]], [mean, mean], linestyle='--', color='orange', label='Mean AoI')

    if median is not None:
        ax.plot([0, aoi_x[-1]], [median, median], linestyle='--', color='green', label='Median AoI')

    plt.show()


def plot_rates(rates, **kwargs) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(20, 5))
    ax.set_title('True vs Nominal Rate')
    ax.set_xlabel('Nominal Rate [msg/s]')
    ax.set_ylabel('Real Rate [msg/s]')

    for config, rate in rates.items():
        ax.plot(RATES[:len(rate)], rate, marker='o', label=config)

    ax.legend()
    plt.show()


def plot_series(mean_aoi: dict, median_aoi: dict, total_energy: dict, true_rate: dict, **kwargs) -> None:
    # AoI vs Rate (mean)
    fig, ax = plt.subplots(1, 1, figsize=(20, 5))
    ax.set_xlabel('Rate [msg/s])')
    ax.set_ylabel('AoI [ms]')
    ax.set_yscale(kwargs.get('aoi_yscale', 'linear'))
    ax.set_title(f'Mean AoI over Rate')

    for config in mean_aoi.keys():
        ax.plot(true_rate[config], mean_aoi[config], marker='o', label=config)
    fig.legend()

    # AoI vs Rate (median)
    fig, ax = plt.subplots(1, 1, figsize=(20, 5))
    ax.set_xlabel('Rate [msg/s])')
    ax.set_ylabel('AoI [ms]')
    ax.set_yscale(kwargs.get('aoi_yscale', 'linear'))
    ax.set_title(f'Median AoI over Rate')
    for config in median_aoi.keys():
        ax.plot(true_rate[config], median_aoi[config], marker='o', label=config)
    fig.legend()

    # Energy vs Rate
    fig, ax = plt.subplots(1, 1, figsize=(20, 5))
    ax.set_title(f'Energy over Rate')
    ax.set_xlabel('Rate [msg/s])')
    ax.set_ylabel('Energy [J]')

    for config in total_energy.keys():
        ax.plot(true_rate[config], total_energy[config], marker='o', label=config)

    fig.legend()


def plot_pareto(mean_aoi: dict, median_aoi: dict, total_energy: dict, time: dict, **kwargs) -> dict:
    fig, ax = plt.subplots(1, 1, figsize=(20, 5))
    ax.set_title(f'Pareto Efficiency ({kwargs.get("metric", "mean").capitalize()})')
    ax.set_xlabel('Mean AoI [ms])')
    ax.set_ylabel('Power [W]')
    ax.set_xscale(kwargs.get('aoi_yscale', 'linear'))

    aoi_values = np.ravel([aoi for aoi in median_aoi.values()])
    power_values = np.ravel([np.divide(energy, time) for energy, time in zip(total_energy.values(), time.values())])
    queue_color = {
        'Q0': 'red',
        'Q1': 'orange',
        'Q16': 'green',
        'Q1024': 'blue'
    }
    markers = {'quic': 'o', 'tls': 'x'}
    front = []
    for config in mean_aoi.keys():
        queue = config.split('-')[-1]
        aoi = np.ravel(mean_aoi[config] if kwargs.get('metric', 'mean') == 'mean' else median_aoi[config])
        power = np.ravel(np.divide(total_energy[config], time[config]))

        transport = 'quic' if 'quic' in config else 'tls'
        config_front = []
        for i, (x, y) in enumerate(zip(aoi, power)):
            dominated = False
            for x1, y1 in zip(aoi_values, power_values):
                if (x1 < x) and (y1 < y):
                    dominated = True
                    break
            if not dominated:
                config_front.append((x, y))
                front.append((config, RATE[i], (x, y)))
        if len(config_front) == 0:
            continue

        ax.scatter(*[list(i) for i in zip(*config_front)],
                   label=config,
                   marker=markers[transport],
                   color=queue_color[queue])

    for point in front:
        ax.annotate(convert_rate(point[1]), point[2])

    front.sort(key=lambda p: p[2][0])
    pareto_front = {'aoi': [], 'power': [], 'config': [], 'rate': []}
    for point in front:
        pareto_front['aoi'].append(point[2][0])
        pareto_front['power'].append(point[2][1])
        pareto_front['config'].append(point[0])
        pareto_front['rate'].append(point[1])

    ax.plot(pareto_front['aoi'], pareto_front['power'], label='Pareto front', color='gray', linestyle='dotted')
    if kwargs.get('x_lim', None) is not None:
        ax.set_xlim(0, kwargs.get('x_lim'))
    ax.set_ylim(0, max(pareto_front['power']) * 1.1)

    ax.grid(linestyle='--', linewidth=0.5)
    fig.legend()

    plt.show()

    return pareto_front
