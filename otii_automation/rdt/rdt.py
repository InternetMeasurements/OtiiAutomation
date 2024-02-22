import json

from .message import Message
from .util import crc_8, logger

MAX_CTR = 2 ** 16


class Rdt:
    def __init__(self, udt):
        self.udt = udt
        self.tx_ctr = 0
        self.rx_ctr = 0

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
        protected = self.tx_ctr.to_bytes(2, byteorder='big') + encoded_msg
        rdt_pkt = f'{msg}{crc_8(protected)}'
        ack = False
        while not ack:
            self.udt.send(rdt_pkt)
            ack = self._recv_ack()

        logger.debug(f'Sent: {msg}')
        self.tx_ctr = (self.tx_ctr + 1) % MAX_CTR

    def receive(self) -> [dict, float]:
        msg = ''
        while True:
            rdt_pkt, timestamp = self.udt.receive()
            if len(rdt_pkt) < 3:
                self._send_ack(nack=True)
                continue

            crc: str = rdt_pkt[-2:]
            msg: str = rdt_pkt[:-2]
            if crc == crc_8(self.rx_ctr.to_bytes(length=2, byteorder='big') + msg.encode()):
                self._send_ack()
                break
            else:
                logger.debug(f'Invalid crc: {crc}')
                self._send_ack(nack=True)

        logger.debug(f"Received: {msg}")
        return json.loads(msg), timestamp

    def _send_ack(self, nack=False) -> None:
        if nack:
            ctr = ((self.rx_ctr - 1) % MAX_CTR).to_bytes(2, byteorder='big')
        else:
            ctr = self.rx_ctr.to_bytes(2, byteorder='big')
            self.rx_ctr = (self.rx_ctr + 1) % MAX_CTR

        ack = crc_8(ctr)
        self.udt.send(ack)

        logger.debug(f'{"Nack" if nack else "Ack"} sent: {ctr}')

    def _recv_ack(self) -> bool:
        ack, _ = self.udt.receive(timeout=8)
        if len(ack) != 2:
            logger.debug(f'Duplicated message: {ack}')
            self._send_ack(nack=True)
            return False
        if ack == crc_8(self.tx_ctr.to_bytes(2, byteorder='big')):
            logger.debug(f'Ack received: {ack}')
            return True
        else:
            logger.debug(f'Invalid ack: {ack}')
            return False
