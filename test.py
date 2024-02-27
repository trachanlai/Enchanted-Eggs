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
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(a, b)))

def get_neighbors():
    """Generate all possible movements including diagonal."""
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue  # Skip the current node itself
                neighbors.append((dx, dy, dz))
    return neighbors

def movement_cost(current, neighbor):
    """Calculate the cost of moving from current to neighbor."""
    dx, dy, dz = np.abs(np.array(neighbor) - np.array(current))
    if dx + dy + dz == 3:
        return np.sqrt(3)  # Diagonal in 3D
    elif dx + dy + dz == 2:
        return np.sqrt(2)  # Diagonal in 2D (plane)
    else:
        return 1  # Straight movement

def a_star(start, goal, obstacles, grid_size):
    neighbors = get_neighbors()
    
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    while not open_set.empty():
        current = open_set.get()[1]

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]  # Return reversed path
        
        for dx, dy, dz in neighbors:
            neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
            
            # Skip out of bounds or through obstacles
            if not (0 <= neighbor[0] < grid_size[0] and 0 <= neighbor[1] < grid_size[1] and 0 <= neighbor[2] < grid_size[2]):
                continue
            if neighbor in obstacles:
                continue
            
            tentative_g_score = g_score[current] + movement_cost(current, neighbor)
            
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
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
                # CALCULATION OF NEW POSITIONS
                
                start = (0, 0, 0)
                goal = (9, 9, 9)
                grid_size = (10, 10, 10)
                obstacles = {(x, y, z) for x in range(3, 7) for y in range(3, 7) for z in range(3, 7)}  # Define a cube obstacle
                
                path = a_star(start, goal, obstacles, grid_size)
                print("Path:", path)

                # #Writing new position
                # addr = "plc/app/Application/sym/PLC_PRG/x"
                # with Variant() as data:
                #     data.set_float64(returnx)
                #     result, _ = datalayer_client.write_sync(addr, data)

                # addr = "plc/app/Application/sym/PLC_PRG/y"
                # with Variant() as data:
                #     data.set_float64(returny)
                #     result, _ = datalayer_client.write_sync(addr, data)

                # addr = "plc/app/Application/sym/PLC_PRG/z"
                # with Variant() as data:
                #     data.set_float64(returnz)
                #     result, _ = datalayer_client.write_sync(addr, data)

                time.sleep(2.0)

            print("ERROR ctrlX Data Layer is NOT connected")
            print("INFO Closing subscription", flush=True)

        stop_ok = datalayer_system.stop(
            False
        )  # Attention: Doesn't return if any provider or client instance is still running
        print("System Stop", stop_ok, flush=True)


if _name_ == "_main_":
    main()