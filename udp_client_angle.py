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
from helpers.DigitalBeamForming import *
import radarsimpy.processing as proc


#49158 
BUFFER_SIZE = 49158 

# IP details for the UDP server
DEFAULT_IP   = '192.168.137.34'  # IP address of the UDP server
DEFAULT_PORT = 57345             # Port of the UDP server for data
DEFAULT_MODE = "data"



            


class LivePlot:
    def __init__(self, max_angle_degrees: float, max_range_m: float):
        # max_angle_degrees: maximum supported speed
        # max_range_m:   maximum supported range
        self.h = None
        self.max_angle_degrees = max_angle_degrees
        self.max_range_m = max_range_m

        plt.ion()

        self._fig, self._ax = plt.subplots(nrows=1, ncols=1)

        self._fig.canvas.manager.set_window_title("Range-Angle-Map using Digital Beam Forming")
        self._fig.canvas.mpl_connect('close_event', self.close)
        self._is_window_open = True

    def _draw_first_time(self, data: np.ndarray):
        # First time draw

        minmin = -60
        maxmax = 0

        self.h = self._ax.imshow(
            data,
            vmin=minmin, vmax=maxmax,
            cmap='viridis',
            extent=(-self.max_angle_degrees,
                    self.max_angle_degrees,
                    0,
                    self.max_range_m),
            origin='lower')

        self._ax.set_xlabel("angle (degrees)")
        self._ax.set_ylabel("distance (m)")
        self._ax.set_aspect("auto")

        self._fig.subplots_adjust(right=0.8)
        cbar_ax = self._fig.add_axes([0.85, 0.0, 0.03, 1])

        cbar = self._fig.colorbar(self.h, cax=cbar_ax)
        cbar.ax.set_ylabel("magnitude (a.u.)")

    def _draw_next_time(self, data: np.ndarray):
        # Update data for each antenna

        self.h.set_data(data)

    def draw(self, data: np.ndarray, title: str):
        if self._is_window_open:
            if self.h:
                self._draw_next_time(data)
            else:
                self._draw_first_time(data)
            self._ax.set_title(title)

            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()

    def close(self, event=None):
        if not self.is_closed():
            self._is_window_open = False
            plt.close(self._fig)
            plt.close('all')
            print('Application closed!')

    def is_closed(self):
        return not self._is_window_open

    
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
        max_range_m = 0.87
        num_rx_antennas = 3

        num_samples = 64
        num_chirps_per_frame = 32
        num_ant = 2
   
        num_beams = 27  # number of beams
        max_angle_degrees_a = 45  # maximum angle, angle ranges from -40 to +40 degrees
        max_angle_degrees_e = 40
        doppler = DopplerAlgo(num_samples, num_chirps_per_frame, num_ant)
        dbf_a = DigitalBeamForming(num_ant , num_beams=num_beams, max_angle_degrees=max_angle_degrees_a)
        dbf_e = DigitalBeamForming(num_ant , num_beams=num_beams, max_angle_degrees=max_angle_degrees_e)
        plot_e = LivePlot(max_angle_degrees_e, max_range_m)
        plot_a = LivePlot(max_angle_degrees_a, max_range_m)

        
        while True:
                
            try:
                     
                data, adr  = s.recvfrom(BUFFER_SIZE)
                data = np.frombuffer(data, dtype=np.uint16)
                #print("data length :",len(data))
                #print("data :",data[3:6])
                            
                radar_data = data[3:]
                        
                gesture_frame = deinterleave_antennas(radar_data,num_samples,num_chirps_per_frame,num_rx_antennas)
            
                rd_spectrum_e = np.zeros((num_samples, 2 * num_chirps_per_frame, num_ant), dtype=complex)
                rd_spectrum_a = np.zeros((num_samples, 2 * num_chirps_per_frame, num_ant), dtype=complex)

                beam_range_energy_e = np.zeros((num_samples, num_beams))
                beam_range_energy_a = np.zeros((num_samples, num_beams))

                elevation_angle = gesture_frame[[1, 2],:,:]
                azimuth_angle = gesture_frame[[0, 2],:,:]
                
                for i_ant in range(num_ant):  # For each antenna
                    # Current RX antenna (num_samples_per_chirp x num_chirps_per_frame)
                    mat_e = elevation_angle[i_ant, :, :]
                    mat_a = azimuth_angle[i_ant, :, :]

                    # Compute Doppler spectrum
                    dfft_dbfs_e = doppler.compute_doppler_map(mat_e, i_ant)
                    dfft_dbfs_a = doppler.compute_doppler_map(mat_a, i_ant)
                    rd_spectrum_e[:, :, i_ant] = dfft_dbfs_e
                    rd_spectrum_a[:, :, i_ant] = dfft_dbfs_a
        
                # Compute Range-Angle map
                rd_beam_formed_e = dbf_e.run(rd_spectrum_e)
                rd_beam_formed_a = dbf_a.run(rd_spectrum_a)         
                
                for i_beam in range(num_beams):
                    doppler_i_e = rd_beam_formed_e[:, :, i_beam]
                    doppler_i_a = rd_beam_formed_a[:, :, i_beam]
                    beam_range_energy_e[:, i_beam] += np.linalg.norm(doppler_i_e, axis=1) / np.sqrt(num_beams)
                    beam_range_energy_a[:, i_beam] += np.linalg.norm(doppler_i_a, axis=1) / np.sqrt(num_beams)
                
                # Maximum energy in Range-Angle map
                beam_range_energy_e[0:10,:]=0
                beam_range_energy_a[0:10,:]=0
                max_energy_e = np.max(beam_range_energy_e)
                max_energy_a = np.max(beam_range_energy_a)

                scale = 150
                beam_range_energy_e = scale * (beam_range_energy_e / max_energy_e - 1)
                beam_range_energy_a = scale * (beam_range_energy_a / max_energy_a - 1)

                #re_avg = np.abs(beam_range_energy_e)
                #ra_avg = np.abs(beam_range_energy_a)
                """
                cfar_e = proc.cfar_os_2d(
                    re_avg, guard=3, trailing=45, pfa=1e-4, k=1500, detector="linear"
                    )
                cfar_a = proc.cfar_os_2d(
                    ra_avg, guard=3, trailing=45, pfa=1e-4, k=1500, detector="linear"
                )

                beam_range_energy_e =20 * np.log10(cfar_e)   
                beam_range_energy_a =20 * np.log10(cfar_a)            
                """

                            # Find dominant angle of target
                _, idx_e = np.unravel_index(beam_range_energy_e.argmax(), beam_range_energy_e.shape)
                angle_degrees_e = np.linspace(-max_angle_degrees_e, max_angle_degrees_e, num_beams)[idx_e]

                _, idx_a = np.unravel_index(beam_range_energy_a.argmax(), beam_range_energy_a.shape)
                angle_degrees_a = np.linspace(-max_angle_degrees_a, max_angle_degrees_a, num_beams)[idx_a]

                            # And plot...
                plot_e.draw(beam_range_energy_e, f"Range-Angle-elevation map using DBF, angle={angle_degrees_e:+02.0f} degrees")
                plot_a.draw(beam_range_energy_a, f"Range-Angle-azimuth map using DBF, angle={angle_degrees_a:+02.0f} degrees")
        
        
            except KeyboardInterrupt:
                plot_e.close()
                plot_a.close()
                break
                    
            
              
               


                        
                        
               
	
if __name__ == '__main__':
        parser = optparse.OptionParser()
        parser.add_option("-p", "--port", dest="port", type="int", default=DEFAULT_PORT, help="Port to listen on [default: %default].")
        parser.add_option("--hostname", dest="hostname", default=DEFAULT_IP, help="Hostname or IP address of the server to connect to.")
        parser.add_option("-m", "--mode", dest="mode", type="string", default=DEFAULT_MODE, help="Mode for radar: test, data.")
        (options, args) = parser.parse_args()
        #start udp client to connect to radar device

    
            
        
        udp_client_radar(options.hostname, options.port)    
        


