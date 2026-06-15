#!/usr/bin/env python3
from scapy.all import rdpcap, AsyncSniffer
import sys

print("Loading pcap file...")
sniffer = AsyncSniffer(
    offline="ARP_Spoofing_train.pcap",
    filter="ip and (tcp or udp)",
    store=True,  # Store packets to count them
)

sniffer.start()
sniffer.join()

packets = sniffer.results
print(f"\nTotal packets captured with filter: {len(packets)}")

if len(packets) > 0:
    print("\nFirst 5 packets:")
    for i, pkt in enumerate(packets[:5]):
        print(f"  {i+1}. {pkt.summary()}")
else:
    print("\n WARNING: No packets matched the filter 'ip and (tcp or udp)'")
    print("Let's check all packets without filter...")
    
    all_pkts = rdpcap("ARP_Spoofing_train.pcap", count=20)
    print(f"\nFirst 20 packets (no filter):")
    for i, pkt in enumerate(all_pkts):
        print(f"  {i+1}. {pkt.summary()}")
