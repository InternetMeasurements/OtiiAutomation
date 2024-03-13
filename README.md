# OtiiAutomation

---

## Environment
The script has been tested on a Python 3.11 environment for the controller and Python 3.7 for the device.

### Required Packages

The following packages are required to run the code:

<table>
    <caption><b>Controller side packages and versions</b></caption>
    <thead>
        <tr><th>Package</th><th>Version</th><th>Package</th><th>Version</th></tr>
    </thead>
    <tr>
      <td>bcrypt</td>
      <td>4.1.2</td>
      <td>pycparser</td>
      <td>2.21</td>
    </tr>
    <tr>
      <td>cffi</td>
      <td>1.16.0</td>
      <td>PyNaCl</td>
      <td>1.5.0</td>
    </tr>
    <tr>
      <td>crc</td>
      <td>6.1.1</td>
      <td>python-dateutil</td>
      <td>2.9.0.post0</td>
    </tr>
    <tr>
      <td>cryptography</td>
      <td>42.0.5</td>
      <td>scp</td>
      <td>0.14.5</td>
    </tr>
    <tr>
      <td>otii-tcp-client</td>
      <td>1.0.7</td>
      <td>six</td>
      <td>1.16.0</td>
    </tr>
    <tr>
      <td>paho-mqtt</td>
      <td>1.6.1</td>
      <td>tomli</td>
      <td>2.0.1</td>
    </tr>
    <tr>
      <td>paramiko</td>
      <td>3.4.0</td>
      <td>win-precise-time</td>
      <td>1.4.2</td>
    </tr>
</table>

<table>
    <caption><b>Device side packages and versions</b></caption>
    <thead>
        <tr><th>Package</th><th>Version</th><th>Package</th><th>Version</th></tr>
    </thead>
    <tbody>
    <tr>
        <td>bcrypt</td>
        <td>4.1.2</td>
        <td>tomli</td>
        <td>2.0.1</td>
        </tr>
    <tr>
        <td>cffi</td>
        <td>1.15.1</td>
        <td>scp</td>
        <td>0.14.5</td>
    </tr>
    <tr>
        <td>crc</td>
        <td>3.0.1</td>
        <td>PyYAML</td>
        <td>6.0.1</td>
    </tr>
    <tr>
        <td>cryptography</td>
        <td>42.0.5</td>
        <td>pyserial</td>
        <td>3.5</td>
    </tr>
    <tr>
        <td>future</td>
        <td>1.0.0</td>
        <td>PyNaCl</td>
        <td>1.5.0</td>
    </tr>
    <tr>
        <td>ifcfg</td>
        <td>0.24</td>
        <td>pycparser</td>
        <td>2.21</td>
    </tr>
    <tr>
        <td>iso8601</td>
        <td>2.1.0</td>
        <td>pkg_resources</td>
        <td>0.0.0</td>
    </tr>
    <tr>
        <td>paramiko</td>
        <td>3.4.0</td>
    </tr>
  </tbody>
</table>


### Installation

You can install these packages using the following command:

Controller side
```bash
pip3 install -r requirements_controller.txt
```

Device side
```bash
python3 -m pip install -U pip
sudo apt-get install build-essential libssl-dev libffi-dev python3-dev cargo pkg-config
pip3 install -r requirements_device.txt
```
---

## Running experiments

### Controller side (laptop)
```bash
python3 main.py controller --config <path-to-the-config-file>
```

### Device side (Raspberry Pi)
```bash
nohup python3 main.py device
```