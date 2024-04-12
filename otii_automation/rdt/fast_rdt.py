import json

from .exception import RdtException
from .message import Message
from .util import crc_8, logger

MAX_CTR = 2 ** 8


class FastRdt:
    def __init__(self, udt):
        self.udt = udt

    def udt_send(self, code: Message, payload: dict = None) -> None:
        if payload is None:
            msg = json.dumps({'code': code.value})
        else:
            msg = json.dumps({'code': code.value, 'payload': payload})

        logger.debug(f'Udt sent: {msg}')
        self.udt.send(msg)

    def udt_receive(self) -> [str, float]:
        msg, timestamp = self.udt.receive()
        logger.debug(f'Udt received: {msg}')

        return json.loads(msg), timestamp

    def send(self, code: Message, payload: dict = None) -> None:
        if payload is None:
            msg = json.dumps({'code': code.value})
        else:
            msg = json.dumps({'code': code.value, 'payload': payload})

        encoded_msg = msg.encode(encoding='utf-8')
        rdt_pkt = f'{msg}{crc_8(encoded_msg)}'

        self.udt.send(rdt_pkt)

        logger.debug(f'Sent: {msg}')

    def receive(self, timeout=None) -> [dict, float]:

        rdt_pkt, timestamp = self.udt.receive(timeout=timeout)

        # Check integrity (message format)
        if len(rdt_pkt) < 3:
            logger.warning(f'Received corrupted message or timeout expired {rdt_pkt}')
            raise RdtException(f'Received corrupted message or timeout expired {rdt_pkt}')

        # Check integrity (crc)
        crc: str = rdt_pkt[-2:]
        msg: str = rdt_pkt[:-2]
        if crc != crc_8(msg.encode(encoding='utf-8')):
            logger.warning(f'Invalid crc {rdt_pkt}')
            raise RdtException(f'Invalid crc {rdt_pkt}')

        logger.debug(f"Received: {msg}")

        # Parse message
        json_msg = json.loads(msg)

        return json_msg, timestamp
