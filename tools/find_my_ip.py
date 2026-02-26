"""Find your PC's local LAN IP address for Quest 3 connection."""

import socket
import sys


def get_local_ip() -> str:
    """Return the LAN IP by connecting to an external address (no traffic sent)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def get_all_ips() -> list[str]:
    """Return all IPv4 addresses on this machine."""
    hostname = socket.gethostname()
    try:
        addrs = socket.getaddrinfo(hostname, None, socket.AF_INET)
        return list({addr[4][0] for addr in addrs if not addr[4][0].startswith("127.")})
    except Exception:
        return []


def main():
    primary = get_local_ip()
    all_ips = get_all_ips()

    print("=" * 50)
    print("  Conversational AI Avatar — IP Finder")
    print("=" * 50)
    print()
    print(f"  Primary LAN IP:  {primary}")
    print()

    if all_ips:
        print("  All network interfaces:")
        for ip in sorted(all_ips):
            marker = " ← (primary)" if ip == primary else ""
            print(f"    {ip}{marker}")
        print()

    print(f"  WebSocket URL for Quest 3:")
    print(f"    ws://{primary}:8765")
    print()
    print("  In Unity ConversationManager, set:")
    print(f'    Server Address = "{primary}"')
    print(f"    Server Port    = 8765")
    print()
    print("  Make sure your firewall allows port 8765:")
    if sys.platform == "win32":
        print('    netsh advfirewall firewall add rule name="AI Avatar" '
              'dir=in action=allow protocol=tcp localport=8765')
    else:
        print("    sudo ufw allow 8765/tcp")
    print("=" * 50)


if __name__ == "__main__":
    main()
