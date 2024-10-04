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
from scipy.signal import butter, filtfilt

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
        self._h = []
        self._max_speed_m_s = max_speed_m_s
        self._max_range_m = max_range_m
        self._num_ant = num_ant
        
        plt.ion()

        self._fig, ax = plt.subplots(nrows=1, ncols=num_ant, figsize=((num_ant + 1) // 2, 2))
        if (num_ant == 1):
            self._ax = [ax]
        else:
            self._ax = ax

        self._fig.canvas.manager.set_window_title("Doppler")
        self._fig.set_size_inches(3 * num_ant + 1, 3 + 1 / num_ant)
        self._fig.canvas.mpl_connect('close_event', self.close)
        self._is_window_open = True



    def _draw_first_time(self, data_all_antennas):
        # First time draw
        #
        # It computes minimal, maximum value and draw data for all antennas
        # in same scale
        # data_all_antennas: array of raw data for each antenna

        minmin = min([np.min(data) for data in data_all_antennas])
        maxmax = max([np.max(data) for data in data_all_antennas])
        """
                        extent=(-self._max_speed_m_s,
                        self._max_speed_m_s,
                        0,
                        self._max_range_m)

                        vmin=minmin, vmax=maxmax,
        """
        for i_ant in range(self._num_ant):
            data = data_all_antennas[i_ant]
            h = self._ax[i_ant].imshow(
                data,
                aspect='auto',
                cmap='jet',
                extent=(-self._max_speed_m_s,
                        self._max_speed_m_s,
                        0,
                        self._max_range_m),
                origin='lower')
            self._h.append(h)

            self._ax[i_ant].set_xlabel("velocity (m/s)")
            self._ax[i_ant].set_ylabel("distance (m)")
            self._ax[i_ant].set_title("antenna #" + str(i_ant))
        self._fig.subplots_adjust(right=0.8)
        cbar_ax = self._fig.add_axes([0.85, 0.0, 0.03, 1])

        cbar = self._fig.colorbar(self._h[0], cax=cbar_ax)
        cbar.ax.set_ylabel("magnitude (dB)")
        

    def _draw_next_time(self, data_all_antennas):
        # data_all_antennas: array of raw data for each antenna
        for i_ant in range(0, self._num_ant):
            data = data_all_antennas[i_ant]
            self._h[i_ant].set_data(data)

        
    def draw(self, data_all_antennas):
        # Draw plots for all antennas
        # data_all_antennas: array of raw data for each antenna
        
        if self._is_window_open:
            if len(self._h) == 0:  # handle the first run
                self._draw_first_time(data_all_antennas)
                print("draw_first_time")
            else:
                self._draw_next_time(data_all_antennas)

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

def highpass_filter(data, cutoff_freq, sample_rate, order=5):
    nyquist = 0.5 * sample_rate  # 奈奎斯特频率
    normal_cutoff = cutoff_freq / nyquist  # 归一化截止频率
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def lowpass_filter(data, cutoff_freq, sample_rate, order=5):
    nyquist = 0.5 * sample_rate  # 奈奎斯特频率
    normal_cutoff = cutoff_freq / nyquist  # 归一化截止频率
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data
"""
def deinterleave_antennas(buffer_ptr, num_samples_per_chirp, num_chirps_per_frame, num_rx_antennas):
    norm_factor = 1.0
    # 初始化二维列表，每个子列表对应一个天线的数据
    gesture_frame = [[] for _ in range(num_rx_antennas)]

    antenna = 0
    index = 0

    for i in range(num_samples_per_chirp * num_chirps_per_frame * num_rx_antennas):
        # 将数据添加到对应天线的子列表中
        gesture_frame[antenna].append(buffer_ptr[i] * norm_factor)
        antenna += 1
        if antenna == num_rx_antennas:
            antenna = 0
            index += 1

    return gesture_frame
"""
def udp_client_radar_test(server_ip, server_port):
        """
         server_ip: IP address of the udp server
         server_port: port on which the server is listening

        This functions intializes the connection to udp server and starts radar device in test
        mode. The status is shown on the terminal.
        
        """
        print("================================================================================")
        print("UDP Client for Radar data test")
        print("================================================================================")
        print("Sending radar configuration. IP Address:",server_ip, " Port:",server_port)
        
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
         
        print("Start radar device in test mode")
        s.sendto('{"radar_transmission":"test"}'.encode(), (server_ip, server_port))

        while True:
                try:
                        msg, adr  = s.recvfrom(BUFFER_SIZE)
                        print(msg.decode())
                except KeyboardInterrupt:
                        break




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
        num_rx_antennas = 3

        draw = Draw(
        max_speed_m_s,
        max_range_m,
        num_rx_antennas)

       
        num_samples = 64
        num_chirps_per_frame = 32
        num_ant = 3
   

        #num_frame_collect = 10
        doppler = DopplerAlgo(num_samples, num_chirps_per_frame, num_ant)
        
        # 初始化存储数据的数组
        #data_collection = np.zeros((num_ant, num_chirps_per_frame, num_samples), dtype=np.float32)
        while True:
                try:    
                     
                     
                    data, adr  = s.recvfrom(BUFFER_SIZE)
                    data = np.frombuffer(data, dtype=np.uint16)
                    #print("data length :",len(data))
                    #print("data :",data[3:6])
                        
                    radar_data = data[3:]
                    
                    gesture_frame = deinterleave_antennas(radar_data,num_samples,num_chirps_per_frame,num_ant)
                    data_all_antennas = []

                    if not draw.is_open():
                            break
                    for i_ant in range(0, num_rx_antennas):  # for each antenna
                        mat = gesture_frame[i_ant,:,:]
                        #mat = highpass_filter(mat, cutoff_freq_hp, sample_rate)
                        #mat = lowpass_filter(mat, cutoff_freq_lp, sample_rate)
                        dfft_dbfs = linear_to_dB(doppler.compute_doppler_map(mat, i_ant))
                        data_all_antennas.append(dfft_dbfs)

                    draw.draw(data_all_antennas)
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
        


