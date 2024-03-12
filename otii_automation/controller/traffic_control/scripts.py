def build_init():
    return [
        # Redirect incoming traffic to ifb0
        'sudo modprobe ifb',
        'sudo ip link set dev ifb0 up',

        'sudo tc qdisc del dev ifb0 root || true',
        'sudo tc qdisc del dev eth0 root || true',

        'sudo tc qdisc add dev eth0 handle ffff: ingress || true',
        'sudo tc filter add dev eth0 parent ffff: protocol ip u32 match u32 0 0 action mirred egress redirect dev '
        'ifb0 || true'
    ]


def build_restore():
    return [
        'sudo tc qdisc del dev ifb0 root || true',
        'sudo tc qdisc del dev eth0 root || true'
    ]


def build_prio():
    return [
        # Outgoing packets (eth0)
        'sudo tc qdisc add dev eth0 root handle 1: prio',
        'sudo tc filter add dev eth0 parent 1:0 protocol ip prio 1 u32 match ip dst 131.114.0.0/16 flowid 2:1',
        'sudo tc filter add dev eth0 parent 1:0 protocol ip prio 2 u32 match ip dst 0.0.0.0/0 flowid 2:2',

        # Incoming packets (ifb0)
        'sudo tc qdisc add dev ifb0 root handle 1: prio',
        'sudo tc filter add dev ifb0 parent 1:0 protocol ip prio 1 u32 match ip src 131.114.0.0/16 flowid 2:1',
        'sudo tc filter add dev ifb0 parent 1:0 protocol ip prio 2 u32 match ip src 0.0.0.0/0 flowid 2:2'
    ]


def build_set_1(delay):
    # return build_restore() + build_prio() + [
    #     f'sudo tc qdisc add dev ifb0 root netem delay {delay}ms',
    #     f'sudo tc qdisc add dev eth0 root netem delay {delay}ms'
    # ]
    return build_restore() + build_prio() + [
        f'sudo tc qdisc add dev eth0 parent 1:2 handle 2: netem delay {delay}ms',
        f'sudo tc qdisc add dev ifb0 parent 1:2 handle 2: netem delay {delay}ms'
    ]


def build_set_2(delay, bandwidth):
    return build_restore() + build_prio() + [
        f'sudo tc qdisc add dev ifb0 root netem delay {delay}ms',
        f'sudo tc qdisc add dev eth0 root netem delay {delay}ms rate {bandwidth}mbit'
    ]


def build_set_3(delay, dl_bandwidth, ul_bandwidth):
    return build_restore() + build_prio() + [
        f'sudo tc qdisc add dev ifb0 root netem delay {delay}ms rate {dl_bandwidth}mbit',
        f'sudo tc qdisc add dev eth0 root netem delay {delay}ms rate {ul_bandwidth}mbit'
    ]
