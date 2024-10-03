# Radar and WiFi Sensor Fusion Project

This project focuses on the integration of radar and WiFi data for enhanced sensing capabilities. It allows users to customize radar parameters, communicate via WiFi, and process radar data through UDP. The project generates various outputs, including range, range-Doppler maps, range angles (elevation angles), and point clouds.

## Features

- **Customizable Radar Parameters**: Users can configure radar settings to suit specific application needs.
  
- **WiFi Communication**: The project supports WiFi communication for data transmission and device control.
  
- **Radar Data via UDP**: Efficient data transmission using the User Datagram Protocol (UDP) for radar data.
  
- **Generated Outputs**:
  - **Range**: Calculate the distance to detected objects.
  - **Range-Doppler Map**: Visualize object velocity and distance information.
  - **Range Angles (Elevation Angles)**: Compute and display elevation angles for better spatial awareness.
  - **Point Cloud Generation**: Generate point clouds for 3D visualization and analysis of detected objects.

## Hardware Requirements

For this code example, the following hardware is required:

- **XENSIV™ BGT60TR13C Radar Wing Board**: This is part of the connected sensor kit.
- **CYSBSYSKIT-DEV-01 Kit**: This kit is used to connect the radar wing board to your PC.

### Connections

1. **Connect the Radar Wing Board**: Connect the XENSIV™ BGT60TR13C radar wing board to the CYSBSYSKIT-DEV-01 kit through the pin headers.
2. **Connect the CYSBSYSKIT-DEV-01 Kit to Your PC**: Use a USB cable to connect the CYSBSYSKIT-DEV-01 kit to your PC.


