import json
import logging
import os
import subprocess
import ifcfg

from paramiko import SSHClient, AutoAddPolicy
from scp import SCPClient
from string import ascii_letters

from .at_command import send_commands

logger = logging.getLogger('device')


def parse_payload_size(payload_size):
    """ Build a random ascii payload of given size """

    size_units = {
        "B": 2 ** 0,
        "KB": 2 ** 10,
        "MB": 2 ** 20
    }

    unit = payload_size[-2:] if payload_size[-2] in ascii_letters else payload_size[-1:]
    size = int(payload_size[:-len(unit)]) * size_units[unit.upper()]

    return size


def sync_clock():
    """ Sync system clock with NTP server """
    query_sync_cmds = ['timedatectl', 'sudo ntpdate 169.254.250.244']
    sync_cmd = 'sudo ntpdate 169.254.250.244'
    iface_down_cmd = 'sudo ifconfig eth0 down'

    logger.info("Synchronizing system clocks ...")

    # Check clock offset before sync
    pre_sync_res = []
    for cmd in query_sync_cmds:
        pre_sync_res.append(
            subprocess.run(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.decode())

    # Force clock synchronization
    sync_res = subprocess.run(sync_cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # Check clock offset after sync
    post_sync_res = []
    for cmd in query_sync_cmds:
        post_sync_res.append(
            subprocess.run(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.decode())

    # Shutdown network interface
    iface_down_res = subprocess.run(iface_down_cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    logger.info("System clocks synchronized")

    return [
        pre_sync_res,
        sync_res.stdout.decode(),
        post_sync_res,
        iface_down_res.returncode
    ]


def check_connectivity():
    """ Check network status """

    logger.info("Checking connectivity ...")

    for iface in ifcfg.interfaces().values():
        logger.info(f'Network interface: {iface["device"]}')
        logger.info(f'Ipv4 address: {iface["inet"]}')
        logger.info(f'Subnet mask: {iface["netmask"]}')
        logger.info(f'Broadcast address: {iface["broadcast"]}')

    subprocess.run('ping -c 3 -I usb0 8.8.8.8'.split())


def network_status(output_path):
    """ Save network status information on the given file """

    status = []
    if os.path.exists(output_path):
        with open(output_path, 'r') as fin:
            status = json.load(fin)

    outputs = send_commands(['AT+CNMP?', 'AT+CREG?', 'AT+CPSI?', 'AT+CSQ'])
    status.append(outputs)

    with open(output_path, 'w') as fout:
        json.dump(status, fout, indent=2)

    return status


def upload_results(server, local_results, trace):
    with open('.tmp.json', 'w') as fp:
        json.dump(local_results, fp)

    try:
        with SSHClient() as ssh:
            ssh.set_missing_host_key_policy(AutoAddPolicy())
            ssh.connect(hostname=server['host'], username=server['username'], key_filename=server['key_file'])

            with SCPClient(ssh.get_transport()) as scp:
                scp.put('.tmp.json', os.path.join(server['remote_path'], f'{trace}.json'))

        os.remove('.tmp.json')

    except Exception as ex:
        logger.warning(f'Upload results {trace} failed: {ex}')


def upload_logs(server, log_file):
    try:
        with SSHClient() as ssh:
            ssh.set_missing_host_key_policy(AutoAddPolicy())
            ssh.connect(hostname=server['host'], username=server['username'], key_filename=server['key_file'])

            with SCPClient(ssh.get_transport()) as scp:
                scp.put(log_file, os.path.join(server['remote_path'], 'device.log'))

    except Exception as ex:
        logger.warning(f'Upload logs failed: {ex}')
