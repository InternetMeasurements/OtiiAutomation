import orjson
from string import ascii_letters


def payload_to_int(payload_size):
    unit = payload_size[-2:] if payload_size[-2] in ascii_letters else payload_size[-1:]
    size = int(payload_size[:-len(unit)]) * {
        'B': 2 ** 0,
        'KB': 2 ** 10,
        'MB': 2 ** 20
    }[unit.upper()]

    return size


def load_json(file_path):
    with open(file_path, 'rb') as fp:
        return orjson.loads(fp.read())


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
