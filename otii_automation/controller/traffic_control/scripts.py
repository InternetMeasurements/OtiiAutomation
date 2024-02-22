def build_init():
    return [
        'sudo modprobe ifb',
        'sudo ip link set dev ifb0 up',

        'sudo tc qdisc del dev ifb0 root || true',
        'sudo tc qdisc del dev eth0 root || true',

        'sudo tc qdisc add dev eth0 handle ffff: ingress || true',
        'sudo tc filter add dev eth0 parent ffff: protocol ip u32 match u32 0 0 action mirred egress redirect dev '
        'ifb0 || true',

        'sudo tc qdisc add dev ifb0 root netem delay 0ms',
        'sudo tc qdisc add dev eth0 root netem delay 0ms'
    ]


def build_restore():
    return [
        'sudo tc qdisc del dev ifb0 root || true',
        'sudo tc qdisc del dev eth0 root || true'
    ]


def build_set_1(delay):
    return build_restore() + [
        f'sudo tc qdisc add dev ifb0 root netem delay {delay}ms',
        f'sudo tc qdisc add dev eth0 root netem delay {delay}ms'
    ]


def build_set_2(delay, bandwidth):
    return build_restore() + [
        f'sudo tc qdisc add dev ifb0 root netem delay {delay}ms',
        f'sudo tc qdisc add dev eth0 root netem delay {delay}ms rate {bandwidth}mbit'
    ]


def build_set_3(delay, dl_bandwidth, ul_bandwidth):
    return build_restore() + [
        f'sudo tc qdisc add dev ifb0 root netem delay {delay}ms rate {dl_bandwidth}mbit',
        f'sudo tc qdisc add dev eth0 root netem delay {delay}ms rate {ul_bandwidth}mbit'
    ]
