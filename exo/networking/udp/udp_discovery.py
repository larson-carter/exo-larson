import asyncio
import json
import socket
import time
import traceback
import aiohttp
from typing import List, Dict, Callable, Tuple, Coroutine
from exo.networking.discovery import Discovery
from exo.networking.peer_handle import PeerHandle
from exo.topology.device_capabilities import DeviceCapabilities, device_capabilities, UNKNOWN_DEVICE_CAPABILITIES
from exo.helpers import DEBUG, DEBUG_DISCOVERY, get_all_ip_addresses
from .stun_client import get_public_ip_and_port, is_behind_nat


class ListenProtocol(asyncio.DatagramProtocol):
    def __init__(self, on_message: Callable[[bytes, Tuple[str, int]], Coroutine]):
        super().__init__()
        self.on_message = on_message
        self.loop = asyncio.get_event_loop()

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        asyncio.create_task(self.on_message(data, addr))


class BroadcastProtocol(asyncio.DatagramProtocol):
    def __init__(self, message: str, broadcast_port: int):
        self.message = message
        self.broadcast_port = broadcast_port

    def connection_made(self, transport):
        sock = transport.get_extra_info("socket")
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        transport.sendto(self.message.encode("utf-8"), ("<broadcast>", self.broadcast_port))


class UDPDiscovery(Discovery):
    def __init__(
            self,
            node_id: str,
            node_port: int,
            listen_port: int,
            broadcast_port: int,
            create_peer_handle: Callable[[str, str, DeviceCapabilities, bool], PeerHandle],
            broadcast_interval: int = 1,
            discovery_timeout: int = 30,
            device_capabilities: DeviceCapabilities = UNKNOWN_DEVICE_CAPABILITIES,
            tracker_url: str = "http://localhost:8080",  # Add tracker URL
    ):
        self.node_id = node_id
        self.node_port = node_port
        self.listen_port = listen_port
        self.broadcast_port = broadcast_port
        self.create_peer_handle = create_peer_handle
        self.broadcast_interval = broadcast_interval
        self.discovery_timeout = discovery_timeout
        self.device_capabilities = device_capabilities
        self.known_peers: Dict[str, Tuple[PeerHandle, float, float]] = {}
        self.broadcast_task = None
        self.listen_task = None
        self.cleanup_task = None
        self.tracker_url = tracker_url
        self.public_ip, self.public_port = get_public_ip_and_port()
        self.is_behind_nat = is_behind_nat()
        self.heartbeat_task = None

    async def start(self):
        self.device_capabilities = device_capabilities()
        self.broadcast_task = asyncio.create_task(self.task_broadcast_presence())
        self.listen_task = asyncio.create_task(self.task_listen_for_peers())
        self.cleanup_task = asyncio.create_task(self.task_cleanup_peers())

        if self.is_behind_nat:
            await self.register_with_tracker()
            self.heartbeat_task = asyncio.create_task(self.task_heartbeat())

    async def stop(self):
        if self.broadcast_task: self.broadcast_task.cancel()
        if self.listen_task: self.listen_task.cancel()
        if self.cleanup_task: self.cleanup_task.cancel()
        if self.heartbeat_task: self.heartbeat_task.cancel()

        if self.is_behind_nat:
            await self.deregister_from_tracker()

        tasks = [task for task in [self.broadcast_task, self.listen_task, self.cleanup_task, self.heartbeat_task] if
                 task]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def discover_peers(self, wait_for_peers: int = 0) -> List[PeerHandle]:
        if wait_for_peers > 0:
            while len(self.known_peers) < wait_for_peers:
                if DEBUG_DISCOVERY >= 2: print(
                    f"Current peers: {len(self.known_peers)}/{wait_for_peers}. Waiting for more peers...")
                await asyncio.sleep(0.1)

        lan_peers = [peer_handle for peer_handle, _, _ in self.known_peers.values()]

        if self.is_behind_nat:
            wan_peers = await self.get_peers_from_tracker()
            return lan_peers + wan_peers

        return lan_peers

    async def task_broadcast_presence(self):
        message = json.dumps({
            "type": "discovery",
            "node_id": self.node_id,
            "grpc_port": self.node_port,
            "device_capabilities": self.device_capabilities.to_dict(),
            "public_ip": self.public_ip,
            "public_port": self.public_port,
        })

        if DEBUG_DISCOVERY >= 2:
            print("Starting task_broadcast_presence...")
            print(f"\nBroadcast message: {message}")

        while True:
            for addr in get_all_ip_addresses():
                transport = None
                try:
                    transport, _ = await asyncio.get_event_loop().create_datagram_endpoint(
                        lambda: BroadcastProtocol(message, self.broadcast_port),
                        local_addr=(addr, 0),
                        family=socket.AF_INET
                    )
                    if DEBUG_DISCOVERY >= 3:
                        print(f"Broadcasting presence at ({addr})")
                except Exception as e:
                    print(f"Error in broadcast presence ({addr}): {e}")
                finally:
                    if transport:
                        try:
                            transport.close()
                        except Exception as e:
                            if DEBUG_DISCOVERY >= 2: print(f"Error closing transport: {e}")
                            if DEBUG_DISCOVERY >= 2: traceback.print_exc()
            await asyncio.sleep(self.broadcast_interval)

    async def on_listen_message(self, data, addr):
        if not data:
            return

        decoded_data = data.decode("utf-8", errors="ignore")

        if not (decoded_data.strip() and decoded_data.strip()[0] in "{["):
            if DEBUG_DISCOVERY >= 2: print(f"Received invalid JSON data from {addr}: {decoded_data[:100]}")
            return

        try:
            decoder = json.JSONDecoder(strict=False)
            message = decoder.decode(decoded_data)
        except json.JSONDecodeError as e:
            if DEBUG_DISCOVERY >= 2: print(f"Error decoding JSON data from {addr}: {e}")
            return

        if DEBUG_DISCOVERY >= 2: print(f"received from peer {addr}: {message}")

        if message["type"] == "discovery" and message["node_id"] != self.node_id:
            peer_id = message["node_id"]
            peer_host = addr[0]
            peer_port = message["grpc_port"]
            device_capabilities = DeviceCapabilities(**message["device_capabilities"])
            is_wan = message.get("public_ip") is not None

            if is_wan:
                peer_host = message["public_ip"]
                peer_port = message["public_port"]

            if peer_id not in self.known_peers or self.known_peers[peer_id][0].addr() != f"{peer_host}:{peer_port}":
                new_peer_handle = self.create_peer_handle(peer_id, f"{peer_host}:{peer_port}", device_capabilities,
                                                          is_wan)
                if not await new_peer_handle.health_check():
                    if DEBUG >= 1: print(f"Peer {peer_id} at {peer_host}:{peer_port} is not healthy. Skipping.")
                    return
                if DEBUG >= 1: print(
                    f"Adding {peer_id=} at {peer_host}:{peer_port}. Replace existing peer_id: {peer_id in self.known_peers}")
                self.known_peers[peer_id] = (new_peer_handle, time.time(), time.time())
            else:
                if not await self.known_peers[peer_id][0].health_check():
                    if DEBUG >= 1: print(f"Peer {peer_id} at {peer_host}:{peer_port} is not healthy. Removing.")
                    if peer_id in self.known_peers: del self.known_peers[peer_id]
                    return
                self.known_peers[peer_id] = (self.known_peers[peer_id][0], self.known_peers[peer_id][1], time.time())

    async def task_listen_for_peers(self):
        await asyncio.get_event_loop().create_datagram_endpoint(lambda: ListenProtocol(self.on_listen_message),
                                                                local_addr=("0.0.0.0", self.listen_port))
        if DEBUG_DISCOVERY >= 2: print("Started listen task")

    async def task_cleanup_peers(self):
        while True:
            try:
                current_time = time.time()
                peers_to_remove = []
                for peer_id, (peer_handle, connected_at, last_seen) in self.known_peers.items():
                    if (
                            not await peer_handle.is_connected() and current_time - connected_at > self.discovery_timeout) or \
                            (current_time - last_seen > self.discovery_timeout) or \
                            (not await peer_handle.health_check()):
                        peers_to_remove.append(peer_id)

                if DEBUG_DISCOVERY >= 2: print("Peer statuses:", {
                    peer_handle.id(): f"is_connected={await peer_handle.is_connected()}, health_check={await peer_handle.health_check()}, {connected_at=}, {last_seen=}"
                    for peer_handle, connected_at, last_seen in self.known_peers.values()})

                for peer_id in peers_to_remove:
                    if peer_id in self.known_peers: del self.known_peers[peer_id]
                    if DEBUG_DISCOVERY >= 2: print(f"Removed peer {peer_id} due to inactivity or failed health check.")
            except Exception as e:
                print(f"Error in cleanup peers: {e}")
                print(traceback.format_exc())
            finally:
                await asyncio.sleep(self.broadcast_interval)

    async def register_with_tracker(self):
        async with aiohttp.ClientSession() as session:
            data = {
                "node_id": self.node_id,
                "ip": self.public_ip,
                "port": self.public_port,
                "device_capabilities": self.device_capabilities.to_dict()
            }
            try:
                async with session.post(f"{self.tracker_url}/register", json=data) as response:
                    if response.status == 201:
                        if DEBUG_DISCOVERY >= 1: print(f"Successfully registered with tracker: {await response.text()}")
                    else:
                        print(f"Failed to register with tracker: {await response.text()}")
            except Exception as e:
                print(f"Error registering with tracker: {e}")

    async def deregister_from_tracker(self):
        async with aiohttp.ClientSession() as session:
            data = {"node_id": self.node_id}
            try:
                async with session.post(f"{self.tracker_url}/deregister", json=data) as response:
                    if response.status == 200:
                        if DEBUG_DISCOVERY >= 1: print(
                            f"Successfully deregistered from tracker: {await response.text()}")
                    else:
                        print(f"Failed to deregister from tracker: {await response.text()}")
            except Exception as e:
                print(f"Error deregistering from tracker: {e}")

    async def get_peers_from_tracker(self):
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.tracker_url}/peers") as response:
                    if response.status == 200:
                        peers_data = await response.json()
                        wan_peers = []
                        for peer in peers_data:
                            if peer["node_id"] != self.node_id:
                                peer_handle = self.create_peer_handle(
                                    peer["node_id"],
                                    f"{peer['ip']}:{peer['port']}",
                                    DeviceCapabilities(**peer["device_capabilities"]),
                                    is_wan=True
                                )
                                wan_peers.append(peer_handle)
                        if DEBUG_DISCOVERY >= 1: print(f"Retrieved {len(wan_peers)} peers from tracker")
                        return wan_peers
                    else:
                        print(f"Failed to get peers from tracker: {await response.text()}")
                        return []
            except Exception as e:
                print(f"Error getting peers from tracker: {e}")
                return []

    async def task_heartbeat(self):
        while True:
            await asyncio.sleep(20)  # Send heartbeat every 20 seconds
            async with aiohttp.ClientSession() as session:
                data = {"id": self.node_id}
                try:
                    async with session.post(f"{self.tracker_url}/heartbeat", json=data) as response:
                        if response.status != 200:
                            print(f"Failed to send heartbeat to tracker: {await response.text()}")
                except Exception as e:
                    print(f"Error sending heartbeat to tracker: {e}")