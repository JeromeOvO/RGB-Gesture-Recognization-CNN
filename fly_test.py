"""
This script shows a simple scripted flight path using the MotionCommander class.

Simple example that connects to the crazyflie at `URI` and runs a sequence. Change the URI variable to your Crazyflie configuration.
"""
import logging
import time
import sys
import cflib.crtp
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie import Crazyflie
from cflib.positioning.motion_commander import MotionCommander
URI = 'radio://0/100/2M/E7E7E7E701'
cflib.crtp.init_drivers(enable_debug_driver=False)
cf = Crazyflie(rw_cache='./cache')
cf.open_link(URI)
print("URI Opened##########")

thrust_mult = 1
thrust_step = 500
thrust = 20000
pitch = 0
roll = 0
yawrate = 0

# Unlock startup thrust protection
cf.commander.send_setpoint(0, 0, 0, 0)
print('Take off!')
while thrust >= 20000:
    cf.commander.send_setpoint(roll, pitch, yawrate, thrust)
    time.sleep(0.1)
    if thrust >= 40000:
        thrust_mult = -1
    thrust += thrust_step * thrust_mult
print('landing')
cf.commander.send_setpoint(0, 0, 0, 0)