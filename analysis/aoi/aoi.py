import os
import json
import itertools
import numpy as np
import matplotlib.pyplot as plt

RATE = [1, 10, 100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
TRANSPORT = ['quic', 'tls']
DELAY = [0, 80]
PAYLOAD = ['128B']
QUEUE_LENGTH = [0, 1, 16, 1024]
N_ITERATION = 1

AOI_PATHS = {
    '128B_Q1024': '../results/aoi/128B_Q1024',
    '128B_Q16': '../results/aoi/128B_Q16',
    '128B_Q1': '../results/aoi/128B_Q1',
    '128B_Q0': '../results/aoi/128B_Q0',
}


def load_data(exp_path: str):
    with open(os.path.join(exp_path, 'summary.json'), 'r') as fin:
        summary = json.load(fin)

    aoi_data = {}
    for config in summary:
        [i, d, _b, _ng, p, t, r, _, q] = config['trace_name'].split('\\')[-1].split('_')[0:9]
        with open(os.path.join(exp_path, f'{config["trace_name"]}_observer.json'), 'r') as fin:
            aoi_data[((int(i) - 1) // 22, int(d), t, p, int(r), int(q))] = {
                'messages': json.load(fin),
                'energy': config['energy']['diff_ej'],
                'time': config['energy']['diff_t'],
            }

    return aoi_data


def compute_aoi(timestamps: list[dict]) -> tuple[list[float], list[float]]:
    base_rx_time = timestamps[0]['rx_ts']

    # Build AoI serie
    aoi_x = []
    aoi_y = []

    # Compute AoI at reception of messages
    for i in range(1, len(timestamps)):
        start_aoi = (timestamps[i - 1]['rx_ts'] - timestamps[i - 1]['gen_ts'])
        period = (timestamps[i]['rx_ts'] - timestamps[i - 1]['rx_ts'])

        aoi_y.append(start_aoi / 10 ** 6)
        aoi_y.append((start_aoi + period) / 10 ** 6)

        aoi_x.append((timestamps[i - 1]['rx_ts'] - base_rx_time) / 10 ** 9)
        aoi_x.append((timestamps[i - 1]['rx_ts'] - base_rx_time + period) / 10 ** 9)

    aoi_y.append((timestamps[-1]['rx_ts'] - timestamps[-1]['gen_ts']) / 10 ** 6)
    aoi_x.append((timestamps[-1]['rx_ts'] - base_rx_time) / 10 ** 9)

    return aoi_x, aoi_y


def compute_discrete_aoi(timestamps: list) -> list:
    base_rx_time = timestamps[0]['rx_ts']

    # Build AoI serie
    aoi_serie = np.zeros(int((timestamps[-1]['rx_ts'] - base_rx_time) / 10 ** 6) + 1)

    # Compute AoI at reception of messages
    for ts in timestamps:
        index_time = int((ts['rx_ts'] - base_rx_time) / 10 ** 6)
        aoi_serie[index_time] = (ts['rx_ts'] - ts['gen_ts']) / 10 ** 6

    # Fill AoI serie
    for i in range(1, len(aoi_serie)):
        if aoi_serie[i] == 0:
            aoi_serie[i] = aoi_serie[i - 1] + 1

    return aoi_serie.tolist()


def compute_integral_mean_aoi(timestamps: list[dict]) -> float:
    aoi = []
    window_length = timestamps[-1]['rx_ts'] - timestamps[0]['rx_ts']

    for ts in timestamps:
        aoi.append(ts['rx_ts'] - ts['gen_ts'])

    area = 0
    for i in range(1, len(aoi)):
        interarrival = timestamps[i]['rx_ts'] - timestamps[i - 1]['rx_ts']
        if interarrival == 0:
            continue
        rect = interarrival * aoi[i - 1]
        triangle = (interarrival * interarrival) / 2

        area += (rect + triangle)
        # area += (interarrival * ((aoi[i-1] + aoi[i-1] + interarrival) / 2))

    return round((area / window_length) / 10 ** 6, 2)


def check_median(timestamps: list[dict], median: float) -> tuple[float, float]:
    t_above = 0
    t_below = 0

    med = median * (10 ** 6)
    T = timestamps[-1]['rx_ts'] - timestamps[0]['rx_ts']
    for i in range(1, len(timestamps)):
        aoi = timestamps[i - 1]['rx_ts'] - timestamps[i - 1]['gen_ts']
        interarrival = timestamps[i]['rx_ts'] - timestamps[i - 1]['rx_ts']
        t_above += min(max(aoi + interarrival - med, 0), interarrival)
        # t_below += min(max(interarrival - (aoi[i] + interarrival - med), 0), interarrival)
        t_below += min(max(med - aoi, 0), interarrival)

    return t_above / T, t_below / T


def compute_rates(timestamps: list[dict]) -> tuple[float, float]:
    gen_rate = []
    rx_rate = []
    same_ts = 0
    for i in range(1, len(timestamps)):
        gen_rate.append((timestamps[i]['gen_ts'] - timestamps[i - 1]['gen_ts']) / 10 ** 9)
        rx_rate.append((timestamps[i]['rx_ts'] - timestamps[i - 1]['rx_ts']) / 10 ** 9)
        if timestamps[i]['rx_ts'] == timestamps[i - 1]['rx_ts']:
            same_ts += 1

    if same_ts > 0:
        raise "Messages with same rx timestamps"

    return (1 / np.mean(gen_rate)), (1 / np.mean(rx_rate))


def convert_rate(rate: int) -> str:
    mod = rate // 1000
    if mod > 0:
        return f'{mod}K'
    mod = rate // 100
    if mod > 0:
        return f'{mod}C'
    mod = rate // 10
    if mod > 0:
        return f'{mod}D'

    return f'{rate}'


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


def plot_series(mean_aoi: dict, median_aoi: dict, total_energy: dict, true_rate: dict, **kwargs):
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


def plot_pareto(mean_aoi, median_aoi, total_energy, time, **kwargs):
    expected_rate = kwargs.get('rate', RATE)
    fig, ax = plt.subplots(1, 1, figsize=(20, 5))
    ax.set_title(f'Pareto Efficiency ({kwargs.get("metric", "mean").capitalize()})')
    ax.set_xlabel('Mean AoI [ms])')
    ax.set_ylabel('Power [W]')
    ax.set_xscale(kwargs.get('aoi_yscale', 'linear'))
    max_power = 0
    max_aoi = 0
    for config in mean_aoi.keys():
        aoi = mean_aoi[config] if kwargs.get('metric', 'mean') == 'mean' else median_aoi[config]
        power = np.divide(total_energy[config], time[config])
        max_aoi = max(max_aoi, max(aoi))
        max_power = max(max_power, max(power))
        ax.scatter(aoi, power, label=config, marker='o' if 'quic' in config else 'x')
        for i, r in enumerate(expected_rate):
            ax.annotate(convert_rate(r), (aoi[i], power[i]))

    ax.set_xlim(0,max_aoi * 1.1)
    ax.set_ylim(0, max_power * 1.1)

    fig.legend()
    plt.show()


def analyse_experiment(exp_data, **kwargs):
    iteration_series = {
        'mean_aoi': {},
        'median_aoi': {},
        'time': {},
        'energy': {},
        'true_rate': {}
    }
    for it in kwargs.get('iteration', range(0, 1)):
        energy_serie = {}
        mean_aoi_serie = {}
        time_serie = {}
        true_rate = {}
        median_aoi_serie = {}

        if N_ITERATION > 1:
            print(f'\n*** Iteration {it} ***\n')

        # Iterate over all possible configurations
        for (d, t, p, r, q) in itertools.product(DELAY, TRANSPORT, PAYLOAD, kwargs.get('rate', RATE), QUEUE_LENGTH):
            if (it, d, t, p, r, q) not in exp_data:
                continue

            # Retrieve config data
            timestamps = exp_data[(it, d, t, p, r, q)]['messages']

            # Compute AoI
            aoi_x, aoi_y = compute_aoi(timestamps)

            # Compute mean rates
            gen_rate, rx_rate = compute_rates(timestamps)

            # Compute mean AoI
            mean_aoi = compute_integral_mean_aoi(timestamps)

            # Compute median AoI
            median_aoi = np.median(aoi_y[:-1])
            above, below = check_median(timestamps, median_aoi)

            curr_config = f'D{d}-{t}-P{p}-Q{q}'

            # Mean AoI vs rate
            if curr_config not in mean_aoi_serie:
                mean_aoi_serie[curr_config] = [mean_aoi]
            else:
                mean_aoi_serie[curr_config].append(mean_aoi)

            # Energy vs rate
            energy = exp_data[(it, d, t, p, r, q)]['energy']
            if curr_config not in energy_serie:
                energy_serie[curr_config] = [energy]
            else:
                energy_serie[curr_config].append(energy)

            # Median AoI
            if curr_config not in median_aoi_serie:
                median_aoi_serie[curr_config] = [median_aoi]
            else:
                median_aoi_serie[curr_config].append(median_aoi)

            # Execution time
            time = exp_data[(it, d, t, p, r, q)]['time']
            if curr_config not in time_serie:
                time_serie[curr_config] = [time]
            else:
                time_serie[curr_config].append(time)

            # True rate (generation rate)
            if curr_config not in true_rate:
                true_rate[curr_config] = [gen_rate]
            else:
                true_rate[curr_config].append(gen_rate)

            if kwargs.get('draw_plots', True):
                print(curr_config)
                print(f'{"# Messages":30}{len(timestamps)}')
                print(
                    f'{"Observer Time":20}{"[ms]":10}{round((timestamps[-1]["rx_ts"] - timestamps[0]["rx_ts"]) / 10 ** 6, 2)}')

                print(f'{"RPI Time":20}{"[ms]":10}{round(exp_data[(it, d, t, p, r, q)]["time"] * 1000, 2)}')
                print(f'{"RPI Energy":20}{"[mJ]":10}{round(exp_data[(it, d, t, p, r, q)]["energy"], 2)}')
                print(f'{"Mean gen rate":20}{"[msg/s]":10}{round(gen_rate, 2)} (expected {r})')

                print(f'{"Mean receive rate":20}{"[msg/s]":10}{round(rx_rate, 2)}')
                print(f'{"Peak AoI":20}{"[ms]":10}{round(max(aoi_y), 2)}')
                print(f'{"Mean AoI":20}{"[ms]":10}{round(mean_aoi, 2)}')
                print(
                    f'{"Median AoI":20}{"[ms]":10}{round(median_aoi, 2)} (Above {round(above, 3)}, Below {round(below, 3)})')
                print('\n')

            # AoI time evolution
            if kwargs.get('draw_plots', True):
                plot_time_evolution(
                    f'AoI time evolution - {t} Q{q} R{r}',
                    aoi_x,
                    aoi_y,
                    mean_aoi,
                    median_aoi,
                    **kwargs
                )

        for config in mean_aoi_serie.keys():
            iteration_series['mean_aoi'][config] = iteration_series['mean_aoi'].get(config, []) + [
                mean_aoi_serie[config]]
            iteration_series['median_aoi'][config] = iteration_series['median_aoi'].get(config, []) + [
                median_aoi_serie[config]]
            iteration_series['time'][config] = iteration_series['time'].get(config, []) + [time_serie[config]]
            iteration_series['energy'][config] = iteration_series['energy'].get(config, []) + [energy_serie[config]]
            iteration_series['true_rate'][config] = iteration_series['true_rate'].get(config, []) + [true_rate[config]]

        if kwargs.get('draw_plots', True):
            plot_series(mean_aoi_serie, median_aoi_serie, energy_serie, true_rate)

    return iteration_series


def pareto(series: list[dict], **kwargs):
    all_series = {
        'mean_aoi': {},
        'median_aoi': {},
        'energy': {},
        'time': {},
        'true_rate': {}
    }
    for serie in series:
        for metric in serie.keys():
            all_series[metric].update(serie[metric])

    mean_aoi = {}
    median_aoi = {}
    mean_energy = {}
    mean_time = {}
    true_rate = {}

    for config, value in all_series['mean_aoi'].items():
        mean_aoi[config] = np.mean(value, axis=0).tolist()

    for config, value in all_series['median_aoi'].items():
        median_aoi[config] = np.mean(value, axis=0).tolist()

    for config, value in all_series['energy'].items():
        mean_energy[config] = np.mean(value, axis=0).tolist()

    for config, value in all_series['time'].items():
        mean_time[config] = np.mean(value, axis=0).tolist()

    for config, value in all_series['true_rate'].items():
        true_rate[config] = np.mean(value, axis=0).tolist()

    plot_series(mean_aoi, median_aoi, mean_energy, true_rate, **kwargs)
    plot_pareto(mean_aoi, median_aoi, mean_energy, mean_time, **kwargs)


def fast_aoi(path: str = '../results/observer.json'):
    with open(path, 'r') as fin:
        timestamps = json.load(fin)

    # Compute AoI
    aoi_x, aoi_y = compute_aoi(timestamps)

    # Compute mean rates
    gen_rate, rx_rate = compute_rates(timestamps)

    # Compute mean AoI
    mean_aoi = compute_integral_mean_aoi(timestamps)

    # Compute median AoI
    median_aoi = np.median(aoi_y[:-1])
    above, below = np.round(check_median(timestamps, median_aoi), 3)

    # Queue time
    queue_time = []
    for ts in timestamps:
        if 'send_ts' not in ts:
            break
        queue_time.append((ts['send_ts'] - ts['gen_ts']) / 10 ** 6)

    print(f'{"# Messages":30}{len(timestamps)}')
    print(f'{"Observer Time":20}{"[ms]":10}{round((timestamps[-1]["rx_ts"] - timestamps[0]["rx_ts"]) / 10 ** 6, 2)}')
    print(f'{"Mean receive rate":20}{"[msg/s]":10}{round(rx_rate, 2)}')
    print(f'{"Peak AoI":20}{"[ms]":10}{round(max(aoi_y), 2)}')
    print(f'{"Mean AoI":20}{"[ms]":10}{round(mean_aoi, 2)}')
    print(f'{"Median AoI":20}{"[ms]":10}{round(median_aoi, 2)} (Above {above}, Below {below})')

    if len(queue_time) > 0:
        print(f'{"Queue time":20}{"[ms]":10}{round(np.mean(queue_time), 3)}')
    print('\n')

    # AoI time evolution
    plot_time_evolution('AoI time evolution', aoi_x, aoi_y, mean_aoi, median_aoi)
