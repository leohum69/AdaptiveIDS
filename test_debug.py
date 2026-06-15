#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/pcnlab/working-humayun/working/lib/python3.10/site-packages')

from scapy.sendrecv import AsyncSniffer
from cicflowmeter.flow_session import  generate_session_class
from cicflowmeter import flow_session

# Monkey patch to add debugging
original_on_packet = flow_session.FlowSession.on_packet_received
packet_count = [0]

def debug_on_packet(self, packet):
    packet_count[0] += 1
    if packet_count[0] % 1000 == 0:
        print(f"Processed {packet_count[0]} packets, flows: {len(self.flows)}")
    return original_on_packet(self, packet)

flow_session.FlowSession.on_packet_received = debug_on_packet

original_garbage_collect = flow_session.FlowSession.garbage_collect

def debug_garbage_collect(self, latest_time):
    print(f"Garbage collect called: flows={len(self.flows)}, csv_line={self.csv_line}")
    result = original_garbage_collect(self, latest_time)
    print(f"After garbage collect: flows={len(self.flows)}, csv_line={self.csv_line}")
    return result

flow_session.FlowSession.garbage_collect = debug_garbage_collect

output_file = "test_debug.csv"
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
except KeyboardInterrupt:
    print("\nInterrupted!")
    sniffer.stop()
finally:
    sniffer.join()

print(f"\nDone! Processed {packet_count[0]} packets")
print(f"Check {output_file} for results")
