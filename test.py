#!/usr/bin/env python3

# SPDX-FileCopyrightText: Bosch Rexroth AG
#
# SPDX-License-Identifier: MIT

import signal
import sys
import time
from datetime import datetime
from typing import List

import ctrlxdatalayer
import flatbuffers
from comm.datalayer import Metadata, SubscriptionProperties
from ctrlxdatalayer.variant import Result, Variant

from helper.ctrlx_datalayer_helper import get_client
import numpy as np
from queue import PriorityQueue

__close_app = False

def heuristic(a, b):
    """Calculate the Euclidean distance between two points."""
    return np.linalg.norm(np.array(a) - np.array(b))

def get_neighbors():
    """Generate all possible movements including diagonal in 3D space."""
    return [(dx, dy, dz) for dx in [-1, 0, 1] for dy in [-1, 0, 1] for dz in [-1, 0, 1] if not (dx == dy == dz == 0)]

def movement_cost(current, neighbor):
    """Calculate the cost of moving from current to neighbor, accounting for diagonal movements."""
    return np.linalg.norm(np.array(neighbor) - np.array(current))

def direction_vector(a, b):
    """Calculate the normalized direction vector from point a to b."""
    return tuple(np.array(b) - np.array(a))

def a_star(start, goal, obstacles):
    neighbors = get_neighbors()
    
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    direction_from = {}  # Store the direction leading to each node
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    while not open_set.empty():
        current_cost, current = open_set.get()

        if current == goal:
            path = []
            direction_path = []  # Track the directions for filtering straight lines
            while current in came_from:
                if current in direction_from:
                    prev_direction = direction_from[current]
                    if not direction_path or prev_direction != direction_path[-1]:
                        path.append(current)
                    direction_path.append(prev_direction)
                current = came_from[current]
            path.append(start)
            return path[::-1]  # Return reversed path
        
        for offset in neighbors:
            neighbor = tuple(np.array(current) + np.array(offset))
            
            # Skip through obstacles or if the neighbor is invalid
            if neighbor in obstacles:
                continue
            
            tentative_g_score = g_score[current] + movement_cost(current, neighbor)
            
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                direction_from[neighbor] = direction_vector(current, neighbor)
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                open_set.put((f_score[neighbor], neighbor))
    
    return None  # No path found

def handler(signum, frame):
    """handler"""
    global __close_app
    __close_app = True
    # print('Here you go signum: ', signum, __close_app, flush=True)


def main():
    """main"""
    print()
    print("=================================================================")
    print("sdk-py-datalayer-client - A ctrlX Data Layer Client App in Python")
    print(
        "=================================================================", flush=True
    )

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGABRT, handler)

    with ctrlxdatalayer.system.System("") as datalayer_system:
        datalayer_system.start(False)

        datalayer_client, datalayer_client_connection_string = get_client(
            datalayer_system, ip="10.0.2.2", ssl_port=8443
        )
        if datalayer_client is None:
            print(
                "ERROR Connecting",
                datalayer_client_connection_string,
                "failed.",
                flush=True,
            )
            sys.exit(1)

        with (
            datalayer_client
        ):  # datalayer_client is closed automatically when leaving with block
            if datalayer_client.is_connected() is False:
                print(
                    "ERROR ctrlX Data Layer is NOT connected:",
                    datalayer_client_connection_string,
                    flush=True,
                )
                sys.exit(1)

            i = 0
            start = (-400, -400, 0)
            goal = (400, 400, 0)
            obstacles = {(x, y, z) for x in range(-100, 101) for y in range(-100, 101) for z in range(-1, 2)}
            path = a_star(start, goal, obstacles)
            while datalayer_client.is_connected() and not __close_app:
                dt_str = datetime.now().strftime("%H:%M:%S.%f")

                # Float64 -------------------------------------------------------
                addr = "motion/axs/AxisX/state/values/actual/pos"
                result, float64_var = datalayer_client.read_sync(addr)

                with float64_var:
                    float64_valuex = float64_var.get_float64()
                    print(
                        f"INFO {dt_str} Sync read '{addr}': {float64_valuex}",
                        flush=True,
                    )
                    
                # Float64 -------------------------------------------------------
                addr = "motion/axs/AxisY/state/values/actual/pos"
                result, float64_var = datalayer_client.read_sync(addr)

                with float64_var:
                    float64_valuey = float64_var.get_float64()
                    print(
                        f"INFO {dt_str} Sync read '{addr}': {float64_valuey}",
                        flush=True,
                    )

                # Float64 READ -------------------------------------------------------
                addr = "motion/axs/AxisZ/state/values/actual/pos"
                result, float64_var = datalayer_client.read_sync(addr)

                with float64_var:
                    float64_valuez = float64_var.get_float64()
                    print(
                        f"INFO {dt_str} Sync read '{addr}': {float64_valuez}",
                        flush=True,
                    )

                if path[i][0] == float64_valuex and path[i][1] == float64_valuey and path[i][2] == float64_valuez:
                    i += 1
                    #Writing new position
                    addr = "plc/app/Application/sym/PLC_PRG/x"
                    with Variant() as data:
                        data.set_float64(path[i][0])
                        result, _ = datalayer_client.write_sync(addr, data)

                    addr = "plc/app/Application/sym/PLC_PRG/y"
                    with Variant() as data:
                        data.set_float64(path[i][1])
                        result, _ = datalayer_client.write_sync(addr, data)

                    addr = "plc/app/Application/sym/PLC_PRG/z"
                    with Variant() as data:
                        data.set_float64(path[i][2])
                        result, _ = datalayer_client.write_sync(addr, data)
                if path[i][0] == goal[0] and path[i][1] == goal[1] and path[i][2] == goal[2]:
                    break

            print("ERROR ctrlX Data Layer is NOT connected")
            print("INFO Closing subscription", flush=True)

        stop_ok = datalayer_system.stop(
            False
        )  # Attention: Doesn't return if any provider or client instance is still running
        print("System Stop", stop_ok, flush=True)


if _name_ == "_main_":
    main()