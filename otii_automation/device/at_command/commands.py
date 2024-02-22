from serial import Serial
import logging
import time

SERIAL_PORT = '/dev/ttyUSB2'
BAUDRATE = 115200

logging.getLogger('at-cmd')


def send_command(command, channel=None):
    if channel is None:
        channel = Serial(SERIAL_PORT, BAUDRATE)

    channel.write(bytes(f'{command}\r\n', 'UTF-8'))
    time.sleep(0.5)

    output = None
    if channel.in_waiting:
        time.sleep(0.01)
        output = channel.read(channel.in_waiting).decode('UTF-8')
        logging.debug(f'{command} - {output}')

    return output


def send_commands(commands):
    outputs = {}
    with Serial(SERIAL_PORT, BAUDRATE) as channel:
        for command in commands:
            outputs[command] = send_command(command, channel)

    return outputs


def config_radio_4G():
    """ Set radio generation to 4G (mode 38 - 4G only) """

    with Serial(SERIAL_PORT, BAUDRATE) as channel:
        out = send_command('"AT+CNMP?', channel)
        if '38' in out:
            logging.debug('Radio generation already set to 4G')
        else:
            send_command('AT+CNMP=38', channel)
            logging.debug('Radio generation set to 4G')

        # Wait for connectivity
        ret = "NO SERVICE"
        while "NO SERVICE" in ret:
            time.sleep(0.5)
            ret = send_command('AT+CPSI?', channel)
            logging.debug(ret)


def config_radio_5G():
    """ Set radio generation to 4G (mode 109 - LTE+NR5G) """

    with Serial(SERIAL_PORT, BAUDRATE) as channel:
        out = send_command('"AT+CNMP?', channel)
        if '109' in out:
            logging.debug('Radio generation already set to 5G')
        else:
            send_command('AT+CNMP=109', channel)
            logging.debug('Radio generation set to 5G')
            time.sleep(18)  # Needed to init module

        # Wait for connectivity
        ret = "NO SERVICE"
        while "NO SERVICE" in ret:
            time.sleep(0.5)
            ret = send_command('AT+CPSI?', channel)
            logging.debug(ret)
