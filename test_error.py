#!/usr/bin/env python3
import sys
import traceback
sys.path.insert(0, '/home/pcnlab/working-humayun/working/lib/python3.10/site-packages')

from scapy.sendrecv import AsyncSniffer
from cicflowmeter.flow_session import generate_session_class

output_file = "test_error.csv"
output_mode = "flow"
url_model = None

NewFlowSession = generate_session_class(output_mode, output_file, url_model)

print("Creating sniffer...")
sniffer = AsyncSniffer(
    offline="ARP_Spoofing_train.pcap",
    filter="ip and (tcp or udp)",
    prn=None,
    session=NewFlowSession,
    store=False,
)

print("Starting sniffer...")
sniffer.start()

print("Waiting for sniffer to finish...")
try:
    sniffer.join()
except Exception as e:
    print(f"\n\nERROR: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
finally:
    print("\nDone!")
