import subprocess
import time


def publish_emqtt(host, port, transport, topic, payload):
    # Build transport flag, if empty TCP
    if transport == 'ssl' or transport == 'tls':
        enable_transport = '--enable-ssl'
    elif transport == 'quic':
        enable_transport = '--enable-quic'
    else:
        enable_transport = ''  # TCP by default

    # Build shell command
    cmd = f'emqtt pub --host {host} --port {port} --topic {topic} --payload {payload} ' \
          f'{enable_transport} --tls-version tlsv1.3'

    # Run command
    subprocess.run(cmd.split())


def publish_rawmqtt(host, port, transport, qos, topic, payload):
    # Build shell command
    cmd = f'raw-mqtt-cli pub --host {host} --port {port} --transport {transport} ' \
          f'--topic {topic} --size {payload} --insecure --qos {qos} --keep-alive 10'

    # Run command
    start_time = time.time_ns()

    result = subprocess.run(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)

    stop_time = time.time_ns()

    return start_time, stop_time, result


def aoi_rawmqtt(host, port, transport, qos, topic, payload, rate, duration, queue):
    nagle = ''
    if transport == 'tls-nagle-off':
        transport = 'tls'
        nagle = '--nagle-off'

    # Build shell command
    cmd = f'raw-mqtt-stream-cli publish --host {host} --port {port} --transport {transport} --topic {topic} ' \
          f'--size {payload} --qos {qos} --rate {rate} --duration {duration} --queue {queue} {nagle} ' \
          f'--insecure --keep-alive 10'

    # Run command
    result = subprocess.run(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)

    return result
