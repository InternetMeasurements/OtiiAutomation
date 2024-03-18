import traceback
import logging
import os
import json
import time
import subprocess

from .util import sync_clock, logger, network_status, parse_payload_size, upload_results, upload_logs, iface_up_cmd
from .protocols import publish_rawmqtt
from .at_command import config_radio_5G, config_radio_4G
from .protocols.mqtt import aoi_rawmqtt
from ..rdt import Rdt
from ..rdt.message import Message
from ..rdt.udt.uart_serial import UdtUartSerial
from ..environment import Environment as Env

rdt = Rdt(UdtUartSerial('/dev/ttyS0'))

server_config = None


def start_configuration(config):
    """ Start configuration """

    global server_config

    timestamps = {
        'launch': int(time.time_ns())
    }

    # Results folder
    os.makedirs(config['results_dir'], exist_ok=True)

    # Set radio generation (4G/5G)
    if config['radio_generation'] == '4G':
        config_radio_4G()
    elif config['radio_generation'] == '5G':
        config_radio_5G()
    else:
        raise Exception(f'Unknown radio generation: {config["radio_generation"]}')

    # Network card information
    network_status(os.path.join(config['results_dir'], 'network_status.json'))

    logger.info(f'Start configuration {config}')

    # Start experiment timestamp
    timestamps['network_info'] = int(time.time_ns())

    # Parse MQTT message payload size
    payload_size = parse_payload_size(config['payload_size'])

    # Sync system clock
    clock_sync_res = []
    if config['experiment'] == 'aoi':
        clock_sync_res = sync_clock()

    # Start request
    rdt.send(Message.START_REQ)
    if config['experiment'] == 'plain_energy':
        # Publish message via MQTT
        timestamps['start_req'], timestamps['stop_req'], req_result = publish_rawmqtt(
            config['host'], config['port'], config['transport_protocol'], config['qos'], config['topic'], payload_size)
    else:
        # AoI session
        req_result = aoi_rawmqtt(config['host'], config['port'], config['transport_protocol'], config['qos'],
                                 config['topic'], payload_size, config['rate'], config['duration'], config['queue'])
    rdt.send(Message.STOP_REQ)
    if req_result.returncode != 0:
        raise Exception(f'Error on request: {req_result}')

    # Enable Ethernet interface
    iface_up_res = subprocess.run(iface_up_cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if iface_up_res.returncode != 0:
        raise Exception(f'Network interface up failed: {iface_up_res.stdout.decode()}')

    with open(os.path.join(config['results_dir'], 'timestamps.json'), 'w') as fout:
        json.dump(timestamps, fout, indent=2)

    req_result = {
        'return_code': req_result.returncode,
        'stdout': req_result.stdout.decode(),
        'stderr': req_result.stderr.decode()
    }
    with open(os.path.join(config['results_dir'], 'req_result.json'), 'w') as fout:
        json.dump(req_result, fout, indent=2)

    # Network card information
    net_status = network_status(os.path.join(config['results_dir'], 'network_status.json'))

    # Upload local results
    local_results = {
        'req_result': req_result,
        'timestamps': timestamps,
        'network_status': net_status,
        'clock_sync': clock_sync_res
    }

    upload_results(config['server'], local_results, config['results_dir'].split('/')[-1])
    upload_logs(config['server'], Env.log_file)
    server_config = config['server']

    # Wait for idle network card
    if config['experiment'] != 'aoi':
        time.sleep(25)

    # Notify configuration completed
    rdt.send(Message.STOP_CONFIG)
    logger.info("Configuration completed")


def device():
    logging.getLogger('paramiko.transport').setLevel(logging.WARNING)

    while True:
        try:
            logger.info("Waiting for UART messages...")
            message, _ = rdt.receive()

            if message['code'] == Message.START_CONFIG.value:
                start_configuration(message['payload'])
            elif message['code'] == Message.END_EXPERIMENT.value:
                logger.info('Experiment concluded')
                break
            else:
                raise Exception(f'Unknown command: {message}')
        except Exception as ex:
            subprocess.run(iface_up_cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            logger.error(f'Exception on device: {ex}')
            logger.error(traceback.format_exc())
            rdt.send(Message.ERROR)

    # Upload final logs
    if server_config is not None:
        upload_logs(server_config, Env.log_file)
