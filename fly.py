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
#count = [1, 1, 1, 1, 1, 1]
# Only output errors from the logging framework
# logging.basicConfig(level=logging.ERROR)


# if __name__ == '__main__':
#     # Initialize the low-level drivers (don't list the debug drivers)
#     cflib.crtp.init_drivers(enable_debug_driver=False)

    # with SyncCrazyflie(URI) as scf:
    #     # We take off when the commander is created
    #     with MotionCommander(scf) as mc:
    #         print('Taking off!')
    #         time.sleep(1)

            # # There is a set of functions that move a specific distance
            # # We can move in all directions
            # print('Moving forward 0.5m')
            # mc.forward(0.5)
            # # Wait a bit
            # time.sleep(1)

            # print('Moving up 0.2m')
            # mc.up(0.5, 0.01)
            # # Wait a bit
            # time.sleep(1)
            #
            # # print('Doing a 270deg circle');
            # # mc.circle_right(0.5, velocity=0.5, angle_degrees=270)
            #
            # print('Moving down 1m')
            # mc.down(0.5, 0.01)
            # # Wait a bit
            # time.sleep(1)

            # # print('Rolling left 0.2m at 0.6m/s')
            # # mc.left(0.2, velocity=0.6)
            # # # Wait a bit
            # # time.sleep(1)
            #
            # print('Moving forward 0.5m')
            # mc.forward(0.3)
            #
            # # We land when the MotionCommander goes out of scope
            # print('Landing!')


class control_UOV():
    def __init__(self):
        print("UOV Connected!!!!!!")

    def stop(self):
        #global cf
        cf.commander.send_setpoint(0, 0, 0, 0)

    def turnning_four_paddle(self):
        #global cf
        thrust_mult = 1
        thrust_step = 500
        thrust = 2000
        pitch = 0
        roll = 0
        yawrate = 0
        cf.commander.send_setpoint(0, 0, 0, 0)
        print('Take off!')
        while thrust >= 2000:
            cf.commander.send_setpoint(roll, pitch, yawrate, thrust)
            time.sleep(0.1)
            if thrust >= 4000:
                thrust_mult = -1
            thrust += thrust_step * thrust_mult
        print('landing')
        cf.commander.send_setpoint(0, 0, 0, 0)

    def move_process_1(self):
        thrust_mult = 1
        thrust_step = 500
        thrust = 20000
        pitch = 0
        roll = 0
        yawrate = 0
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


#
# def Monitor():
#     UOV = control_UOV()
#     filter_num = 10
#
#     while(True):
#         if count[0] % filter_num == 0:
#             count[0] = 1
#             print("######################################################################")
#             UOV.turnning_four_paddle()
#         elif count[1] % filter_num == 0:
#             count[1] = 1
#         elif count[2] % filter_num == 0:
#             count[2] = 1
#         elif count[3] % filter_num == 0:
#             count[3] = 1
#         elif count[4] % filter_num == 0:
#             count[4] = 1
#         elif count[5] % filter_num == 0:
#             count[5] = 1






    # cf = Crazyflie(rw_cache='./cache')
    # cf.open_link(URI)
    # print('aaaaaaaaaaaaaaa')

    # thrust_mult = 1
    # thrust_step = 500
    # thrust = 20000
    # pitch = 0
    # roll = 0
    # yawrate = 0
    #
    # # Unlock startup thrust protection
    # cf.commander.send_setpoint(0, 0, 0, 0)
    # print('Take off!')
    # while thrust >= 20000:
    #     cf.commander.send_setpoint(roll, pitch, yawrate, thrust)
    #     time.sleep(0.1)
    #     if thrust >= 40000:
    #         thrust_mult = -1
    #     thrust += thrust_step * thrust_mult
    # print('landing')
    # cf.commander.send_setpoint(0, 0, 0, 0)