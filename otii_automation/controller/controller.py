import json
import os
import time
import traceback
from time import sleep

from ..environment import Environment as Env
from ..rdt import Message
from .experiment import Experiment
from .otii import SimpleOtii
from .traffic_control import restore_bandwidth_and_delay, init_bandwidth_and_delay, set_bandwidth_and_delay
from .observer import Observer
from .util import download_results, logger, build_config_message, build_trace_name, download_device_logs
from ..rdt.exception import RdtException

otii: SimpleOtii
observer: Observer


def launch_config(params: dict) -> bool:
    trace = build_trace_name(params)

    logger.info(f'Start configuration: {trace}')

    # Set network bandwidth and delay
    set_bandwidth_and_delay(params['bandwidth'], params['bandwidth'], params['delay'])
    logger.info(f'Network constraints configured')

    # Start trace recording on Otii
    otii.start_recording()
    logger.info(f'Recording started')

    # Send configuration message to device via UART
    otii.send(Message.START_CONFIG, build_config_message(params, trace))
    logger.info(f'Configuration message sent')

    while True:

        results = {
            'trace_name': trace,
            'energy': None,
            'device': None,
            'messages': [],
            'config': params
        }

        # Clean observer buffer
        observer.clean()

        # Wait for device to complete configuration
        while True:
            message, timestamp = otii.receive(timeout=150)
            results['messages'].append({'timestamp': timestamp, 'message': message['code']})

            if message['code'] == Message.START_REQ.value:
                results['req_start'] = timestamp
                logger.info('Request start')
                sleep(50)
            elif message['code'] == Message.STOP_REQ.value:
                results['req_stop'] = timestamp
                logger.info('Request stop')
            elif message['code'] == Message.STOP_CONFIG.value:
                break
            else:
                raise Exception(f'Error on device: {message}')

        if len(results['messages']) == 3:
            break

    # Stop trace recording on Otii
    otii.stop_recording(trace)
    logger.info(f'Recording stopped')

    # Retrieve energy results
    results['energy'] = otii.get_energy(results['req_start'], results['req_stop'])
    # Needed by legacy code
    results['energy']['diff_t'] = results['req_stop'] - results['req_start']
    results['energy']['diff_ej'] = results['energy'].pop('energy')

    # Restore network bandwidth and delay
    restore_bandwidth_and_delay()

    # Download device results
    results['device'] = download_results(trace)

    # Download device logs
    download_device_logs()

    # Dump observed messages
    if Env.config['meta']['experiment'] == 'aoi':
        observer.dump_observed(os.path.join(Env.base_dir, f'{trace}_observer.json'))

    # Dump results
    summary_path = os.path.join(Env.base_dir, 'summary.json')
    summary = []
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as fp:
            summary = json.load(fp)

    with open(summary_path, 'w') as fp:
        summary.append(results)
        json.dump(summary, fp, indent=2)

    # Save Otii project
    otii.save_project(os.path.join(Env.otii_dir, f'Iteration_{Env.iteration}'))

    logger.info(f'Configuration completed: {trace}')

    return True


def controller() -> None:
    global otii
    global observer
    try:
        # Initialize components
        otii = SimpleOtii()
        observer = Observer()
        observer.start_observing()
        init_bandwidth_and_delay()
        experiment = Experiment()
        meta = Env.config['meta']
        meta['seed'] = experiment.seed
        meta['config'] = Env.config['params']
        meta['config'].update(Env.config['params'])
        with open(os.path.join(Env.base_dir, 'meta.json'), 'w') as fp:
            json.dump(meta, fp, indent=1)
        logger.info('Initialization completed')
    except Exception as ex:
        logger.error(f'Initialization failed: {ex}')
        logger.error(traceback.format_exc())
        return

    logger.info(f'Running {len(experiment)} configurations')

    try:
        # Run all iterations
        for it in range(0, Env.config['meta']['repetition']):
            otii.create_project()

            # Run all configurations
            for config in experiment:
                completed = False
                while completed is not True:
                    try:
                        completed = launch_config(config)
                    except Exception as ex:
                        logger.error(f'Configuration failed: {ex}')
                        logger.error(traceback.format_exc())

                        if not isinstance(ex, RdtException):
                            sleep(10)
                            project_path = os.path.join(Env.otii_dir, f'Iteration_{it}') if (
                                    (Env.trace_counter % len(experiment)) != 1) else None
                            otii.reset(project_path)

                Env.trace_counter += 1

            logger.info(f'Iteration {it} completed\n')
            Env.iteration += 1

        # End experiment
        for _ in range(3):
            otii.send(Message.END_EXPERIMENT, no_ack=True)

        # Download device logs
        time.sleep(5)
        download_device_logs()

        logger.info('Experiment completed')
    except Exception as ex:
        logger.error(f'Experiment failed: {ex}')
        logger.error(traceback.format_exc())
