#******************************************************************************
# File Name:   udp_client.py
#
# Description: A simple python based UDP client.
# 
#********************************************************************************
# Copyright 2020-2022, Cypress Semiconductor Corporation (an Infineon company) or
# an affiliate of Cypress Semiconductor Corporation.  All rights reserved.
#
# This software, including source code, documentation and related
# materials ("Software") is owned by Cypress Semiconductor Corporation
# or one of its affiliates ("Cypress") and is protected by and subject to
# worldwide patent protection (United States and foreign),
# United States copyright laws and international treaty provisions.
# Therefore, you may use this Software only as provided in the license
# agreement accompanying the software package from which you
# obtained this Software ("EULA").
# If no EULA applies, Cypress hereby grants you a personal, non-exclusive,
# non-transferable license to copy, modify, and compile the Software
# source code solely for use in connection with Cypress's
# integrated circuit products.  Any reproduction, modification, translation,
# compilation, or representation of this Software except as specified
# above is prohibited without the express written permission of Cypress.
#
# Disclaimer: THIS SOFTWARE IS PROVIDED AS-IS, WITH NO WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, NONINFRINGEMENT, IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. Cypress
# reserves the right to make changes to the Software without notice. Cypress
# does not assume any liability arising out of the application or use of the
# Software or any product or circuit described in the Software. Cypress does
# not authorize its products for use in any products where a malfunction or
# failure of the Cypress product may reasonably be expected to result in
# significant property damage, injury or death ("High Risk Product"). By
# including Cypress's product in a High Risk Product, the manufacturer
# of such system or application assumes all risk of such use and in doing
# so agrees to indemnify Cypress against all liability.
#********************************************************************************

#!/usr/bin/env python
import socket
import optparse
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
from helpers.DopplerAlgo import *
from helpers.DistanceAlgo import *

#49158 
BUFFER_SIZE = 49158 

# IP details for the UDP server
DEFAULT_IP   = '192.168.137.34'  # IP address of the UDP server
DEFAULT_PORT = 57345             # Port of the UDP server for data
DEFAULT_MODE = "data"


class Draw:
    # Represents drawing for example
    #
    # Draw is done for each antenna, and each antenna is represented for
    # other subplot

    def __init__(self, max_speed_m_s, max_range_m, num_ant):
        # max_range_m:   maximum supported range
        # max_speed_m_s: maximum supported speed
        # num_ant:      number of available antennas
        
        self._max_speed_m_s = max_speed_m_s
        self._max_range_m = max_range_m
        self._num_ant = num_ant
        self._h=0
        
        plt.ion()

        self._fig, ax = plt.subplots(nrows=1, ncols=num_ant, figsize=((num_ant + 1) // 2, 2))
        if (num_ant == 1):
            self._ax = [ax]
        else:
            self._ax = ax
        
        # Add labels and legend
        self._ax[0].set_title("Distance Data with Peak Highlighted")
        self._ax[0].set_xlabel("Distance (m)")
        self._ax[0].set_ylabel("Amplitude")
        self._ax[0].legend()
        self._ax[0].grid(True)
        self._fig.canvas.manager.set_window_title("Distance")
        self._fig.set_size_inches(3 * num_ant + 1, 3 + 1 / num_ant)
        self._fig.canvas.mpl_connect('close_event', self.close)
        self._is_window_open = True



    def _draw_first_time(self, data_all_antennas,range_bin_length,distance_peak_m,skip):
        # First time draw
        #
        # It computes minimal, maximum value and draw data for all antennas
        # in same scale
        # data_all_antennas: array of raw data for each antenna

 

        
        data = data_all_antennas
        x_axis = np.arange(len(data)) * range_bin_length
        self._h,= self._ax[0].plot(x_axis, data, label="Distance Data")
        # Highlight the peak
        self._ax[0].axvline(x=distance_peak_m, color='r', linestyle='--', label=f"Peak at {distance_peak_m:.2f} m")
        self._ax[0].scatter([distance_peak_m], [data[skip + np.argmax(data[skip:])]], color='r')
        self._ax[0].set_ylim(0, 1000)
        self._ax[0].set_ylabel("Amplitude (FFT)")
        self._ax[0].set_xlabel("Distance (m)")

        

    def _draw_next_time(self, data_all_antennas,range_bin_length,distance_peak_m,skip):
        # data_all_antennas: array of raw data for each antenna
        self._ax[0].cla()
        x_axis = np.arange(len(data_all_antennas)) * range_bin_length
        # Now re-plot the data and assign the new Line2D object to self._h
        self._h, = self._ax[0].plot(x_axis, data_all_antennas, label="Distance Data")
        # Highlight the peak
        self._ax[0].axvline(x=distance_peak_m, color='r', linestyle='--', label=f"Peak at {distance_peak_m:.2f} m")
        self._ax[0].scatter([distance_peak_m], [data_all_antennas[skip + np.argmax(data_all_antennas[skip:])]], color='r')
        self._ax[0].set_ylim(0, 1000)
        self._ax[0].set_ylabel("Amplitude (FFT)")
        self._ax[0].set_xlabel("Distance (m)")

    def draw(self, data_all_antennas,range_bin_length,distance_peak_m,skip):
        # Draw plots for all antennas
        # data_all_antennas: array of raw data for each antenna
        
        if self._is_window_open:
            if self._h == 0:  # handle the first run
                self._draw_first_time(data_all_antennas,range_bin_length,distance_peak_m,skip)
                print("draw_first_time")
            else:
                self._draw_next_time(data_all_antennas,range_bin_length,distance_peak_m,skip)

            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()

            

    def close(self, event=None):
        if self.is_open():
            self._is_window_open = False
            plt.close(self._fig)
            plt.close('all')  # Needed for Matplotlib ver: 3.4.0 and 3.4.1
            print('Application closed!')

    def is_open(self):
        return self._is_window_open
    
def linear_to_dB(x):
    abs_x = np.abs(x)
    abs_x[abs_x <= 0] = 1e-12 
    return 20 * np.log10(abs_x)


def deinterleave_antennas(buffer,num_samples_per_chirp, num_chirps_per_frame, num_rx_antennas):
    #norm_factor = 1.0

    # Initialize the gesture_frame array with the desired shape
    gesture_frame = np.zeros(
        (num_rx_antennas,num_chirps_per_frame, num_samples_per_chirp),dtype=np.uint16)
    

    for i in range(num_samples_per_chirp * num_chirps_per_frame * num_rx_antennas):
        antenna = i % num_rx_antennas
        chirp = (i // num_rx_antennas) // num_samples_per_chirp
        sample = (i // num_rx_antennas) % num_samples_per_chirp

        gesture_frame[antenna, chirp, sample] = buffer[i] 

    return gesture_frame

def udp_client_radar( server_ip, server_port):
        """
         server_ip: IP address of the udp server
         server_port: port on which the server is listening

        This functions intializes the connection to udp server and starts radar device with
        given configuration. The radar raw data is read from the socket and frame number is
        shown on the terminal. 
        """
        
    
        print("================================================================================")
        print("UDP Client for Radar data")
        print("================================================================================")
        print("Sending radar configuration. IP Address:",server_ip, " Port:",server_port)

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # radar data tranmission mode with presence application settings
        print("Start radar device with data tranmission enabled")
        s.sendto('{"radar_transmission":"enable"}'.encode(), (server_ip, server_port))
        max_range_m = 1.451
        max_speed_m_s = 2.45
        num_rx_antennas = 1

        draw = Draw(
        max_speed_m_s,
        max_range_m,
        num_rx_antennas)

       
        num_samples = 64
        num_chirps_per_frame = 32
        num_ant = 3
        start_frequency_Hz =58500000000
        end_frequency_Hz =62500000000

        #num_frame_collect = 10
        #doppler = DopplerAlgo(num_samples, num_chirps_per_frame, num_ant)
        algo = DistanceAlgo(num_samples,end_frequency_Hz,start_frequency_Hz,num_chirps_per_frame)
        
        # 初始化存储数据的数组
        #data_collection = np.zeros((num_ant, num_chirps_per_frame, num_samples), dtype=np.float32)
        while True:
                try:    
                     
                     
                    data, adr  = s.recvfrom(BUFFER_SIZE)
                    data = np.frombuffer(data, dtype=np.uint16)
  
                    radar_data = data[3:]
                    
                    gesture_frame = deinterleave_antennas(radar_data,num_samples,num_chirps_per_frame,num_ant)
                    

                    distance_peaks = []
                    for i in range(num_ant):  # Loop over the first dimension (frames)
                        mat = gesture_frame[i, :, :]  # Select the ith frame (2D slice)
                        
                        # Compute distance for each frame
                        distance_peak_m, distance_data, range_bin_length, skip = algo.compute_distance(mat)
                            # Store the computed distance_peak_m
                        distance_peaks.append(distance_peak_m)
                    # After the loop, calculate the average of distance_peak_m
                    average_distance_peak_m = sum(distance_peaks) / len(distance_peaks)
                    print("Distance:" + format(average_distance_peak_m, "^05.3f") + "m")
                    #print(range_bin_length)
                        
                    draw.draw(distance_data,range_bin_length,average_distance_peak_m,skip)
                    
                    #print("Received data frame number: ", int.from_bytes(data[2:6], 'little'))
                except KeyboardInterrupt:
                        
                        plt.ioff()
                        draw.close()
                        break
               


                        
                        
               
	
if __name__ == '__main__':
        parser = optparse.OptionParser()
        parser.add_option("-p", "--port", dest="port", type="int", default=DEFAULT_PORT, help="Port to listen on [default: %default].")
        parser.add_option("--hostname", dest="hostname", default=DEFAULT_IP, help="Hostname or IP address of the server to connect to.")
        parser.add_option("-m", "--mode", dest="mode", type="string", default=DEFAULT_MODE, help="Mode for radar: test, data.")
        (options, args) = parser.parse_args()
        #start udp client to connect to radar device

        if options.mode == "test":
                udp_client_radar_test(options.hostname, options.port)
        else:
                udp_client_radar(options.hostname, options.port)    
        


