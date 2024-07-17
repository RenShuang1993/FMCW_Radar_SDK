import numpy as np
import helper_functions


def _apply_range_fft(cube, fft_length):
    """
    Applies range fft on the cube
    :param cube: cube to apply fft
    :param fft_length: length of the fft (in the best case: 2^N)
    :return: cube with applied fft in range duration
    """
    # cube = abs(np.fft.fft(a=cube, n=fft_length, axis=0))
    cube = np.fft.fft(a=cube, n=fft_length, axis=0)
    # cube = cube / np.max(cube, axis=0)
    return cube


def _apply_doppler_fft(cube, fft_length):
    """
    Applies doppler fft and fft-shift on second dimension
    :param cube: cube to apply fft
    :param fft_length: length of the fft (in the best case: 2^N)
    :return: cube with applied fft in chirp duration
    """
    # cube = abs(np.fft.fft(a=cube, n=fft_length, axis=1))
    cube = np.fft.fft(a=cube, n=fft_length, axis=1)
    cube = np.fft.fftshift(cube, axes=1)
    # cube = cube / cube.max()
    return cube


def _apply_angle_fft(cube, fft_length):
    """
    Applies angle FFT on the third dimension of the cube
    :param cube: cube to apply fft
    :param fft_length: length of the fft (in the best case: 2^N)
    :return: cube with applied fft in rx antenna_name duration
    """
    # cube = abs(np.fft.fft(a=cube, n=fft_length, axis=2))
    cube = np.fft.fft(a=cube, n=fft_length, axis=2)
    # cube = cube / cube.max()
    return cube


def apply_3d_fft(cube, fft_range: int, fft_doppler: int, fft_angle: int) -> np.array:
    cube = _apply_range_fft(cube, fft_range)
    cube = _apply_doppler_fft(cube, fft_doppler)
    cube = _apply_angle_fft(cube, fft_angle)
    return cube


def add_ampl_noise(radar_cube, mu_ampl_noise, sigma_ampl_noise):  # Fixme --> really necessary?!
    """
    Adds amplitude noise after aüülied
    :param radar_cube:
    :param mu_ampl_noise:
    :param sigma_ampl_noise:
    :return:
    """
    max_ampl = np.abs(radar_cube).max()
    # for i_chirp in range(radar_cube.shape[1]):
    #     real = np.random.normal(mu_ampl_noise * max_ampl,
    #                             sigma_ampl_noise * max_ampl,
    #                             (radar_cube.shape[0], radar_cube.shape[2]))
    #     imag = np.random.normal(mu_ampl_noise * max_ampl,
    #                             sigma_ampl_noise * max_ampl,
    #                             (radar_cube.shape[0], radar_cube.shape[2]))
    #     radar_cube[:, i_chirp, :] = radar_cube[:, i_chirp, :] + real + 1j * imag

    real = np.random.normal(mu_ampl_noise * max_ampl,
                            sigma_ampl_noise * max_ampl,
                            (radar_cube.shape[0], radar_cube.shape[1], radar_cube.shape[2]))
    imag = np.random.normal(mu_ampl_noise * max_ampl,
                            sigma_ampl_noise * max_ampl,
                            (radar_cube.shape[0], radar_cube.shape[1], radar_cube.shape[2]))
    return radar_cube + real + 1j * imag


def cfar_range(range_map_lin: np.array, offset: float = 0) -> tuple[np.array, np.array]:
    """
    cfar on the range plot
    @see: # https://github.com/tooth2/2D-CFAR/blob/main/radar-target-generation-and-detection.m
    :param range_map_lin: input of range map
    :param offset: offset that is necessary between threshold and detection
    :return: cfar detections [W], threshold area [W]
    """
    # https://github.com/tooth2/2D-CFAR/blob/main/radar-target-generation-and-detection.m

    assert len(range_map_lin.shape) == 1, "'cfar_range_doppler': Wrong size of the range doppler map"
    assert (range_map_lin > 0).any(), "Range map input for cfar is dB --> must be linear"

    cells_train_ran = 8
    cells_guard_ran = 4

    res = np.zeros(range_map_lin.shape)
    thres_area = np.copy(res)

    for i_range in range(cells_train_ran + cells_guard_ran,
                         int(range_map_lin.shape[0]) - (cells_train_ran + cells_guard_ran)):  # iterate over range
        noise_level = 0
        for i_p in range(i_range - (cells_train_ran + cells_guard_ran), i_range + (cells_train_ran + cells_guard_ran)):
            if abs(i_p - i_range) > cells_guard_ran:
                noise_level += range_map_lin[i_p]

        threshold = helper_functions.pow2db(  # Todo here pow2dB or lin2dB?!
            noise_level / (2 * (cells_train_ran + cells_guard_ran + 1) - cells_guard_ran - 1))
        threshold = helper_functions.db2pow(threshold + offset)  # Todo here pow2dB or lin2dB?!

        cut = range_map_lin[i_range]
        thres_area[i_range] = threshold

        res[i_range] = 0 if cut < threshold else 1

    return range_map_lin * res, thres_area


def cfar_range_doppler(range_doppler_map_lin: np.array, offset: float = 0) -> tuple[np.array, np.array]:
    """
    cfar on the range doppler plot
    @see: # https://github.com/tooth2/2D-CFAR/blob/main/radar-target-generation-and-detection.m
    :param range_doppler_map_lin: input of range doppler map
    :param offset: offset that is necessary between threshold and detection
    :return: cfar detections [W], threshold area [W]
    """

    assert len(range_doppler_map_lin.shape) == 2, "'cfar_range_doppler': Wrong size of the range doppler map"
    assert (range_doppler_map_lin > 0).any(), "Range map input for cfar is dB --> must be linear"

    cells_train_ran = 8
    cells_train_dop = 2
    cells_guard_ran = 4
    cells_guard_dop = 2

    res = np.zeros(range_doppler_map_lin.shape)
    thres_area = np.copy(res)

    for i_range in range(cells_train_ran + cells_guard_ran,
                         int(range_doppler_map_lin.shape[0]) - (
                                 cells_train_ran + cells_guard_ran)):  # iterate over range
        for i_doppler in range(cells_train_dop + cells_guard_dop,
                               range_doppler_map_lin.shape[1] - (
                                       cells_train_dop + cells_guard_dop)):  # iterate over doppler
            noise_level = 0
            for i_p in range(i_range - (cells_train_ran + cells_guard_ran),
                             i_range + (cells_train_ran + cells_guard_ran)):
                for i_q in range(i_doppler - (cells_train_dop + cells_guard_dop),
                                 i_doppler + (cells_train_dop + cells_guard_dop)):
                    if (abs(i_p - i_range) > cells_guard_ran) or (abs(i_q - i_doppler) > cells_guard_dop):
                        noise_level += range_doppler_map_lin[i_p, i_q]

            threshold = helper_functions.pow2db(noise_level / (  # Todo here pow2dB or lin2dB?!
                    2 * (cells_train_dop + cells_guard_dop + 1) * 2 * (cells_train_ran + cells_guard_ran + 1) - (
                    cells_guard_ran * cells_guard_dop) - 1))
            threshold = helper_functions.db2pow(threshold + offset)  # Todo here pow2dB or lin2dB?!

            cut = range_doppler_map_lin[i_range, i_doppler]
            thres_area[i_range, i_doppler] = threshold

            res[i_range, i_doppler] = 0 if cut < threshold else 1

    return range_doppler_map_lin * res, thres_area


def cfar_range_angular(range_angular_map_lin: np.array, offset: float = 0):
    """
    cfar on the range doppler plot
    @see: # https://github.com/tooth2/2D-CFAR/blob/main/radar-target-generation-and-detection.m
    :param range_angular_map_lin: input of range doppler map
    :param offset: offset that is necessary between threshold and detection
    :return: cfar detections [W], threshold area [W]
    """

    assert len(range_angular_map_lin.shape) == 2, "'cfar_range_doppler': Wrong size of the range angular map"
    assert (range_angular_map_lin > 0).any(), "Range map input for cfar is dB --> must be linear"

    cells_train_ran = 8
    cells_train_ang = 1
    cells_guard_ran = 4
    cells_guard_ang = 0

    res = np.zeros(range_angular_map_lin.shape)
    thres_area = np.copy(res)

    for i_range in range(cells_train_ran + cells_guard_ran,
                         int(range_angular_map_lin.shape[0]) - (
                                 cells_train_ran + cells_guard_ran)):  # iterate over range
        for i_antenna in range(cells_train_ang + cells_guard_ang,
                               range_angular_map_lin.shape[1] - (
                                       cells_train_ang + cells_guard_ang)):  # iterate over doppler
            noise_level = 0
            for i_p in range(i_range - (cells_train_ran + cells_guard_ran),
                             i_range + (cells_train_ran + cells_guard_ran)):
                for i_q in range(i_antenna - (cells_train_ang + cells_guard_ang),
                                 i_antenna + (cells_train_ang + cells_guard_ang)):
                    if (abs(i_p - i_range) > cells_guard_ran) or (abs(i_q - i_antenna) > cells_guard_ang):
                        noise_level += range_angular_map_lin[i_p, i_q]

            threshold = helper_functions.pow2db(noise_level / (  # Todo here pow2dB or lin2dB?!
                    2 * (cells_train_ang + cells_guard_ang + 1) * 2 * (cells_train_ran + cells_guard_ran + 1) - (
                    cells_guard_ran * cells_guard_ang) - 1))
            threshold = helper_functions.db2pow(threshold + offset)  # Todo here pow2dB or lin2dB?!

            cut = range_angular_map_lin[i_range, i_antenna]
            thres_area[i_range, i_antenna] = threshold

            res[i_range, i_antenna] = 0 if cut < threshold else 1

    return range_angular_map_lin * res, thres_area


if __name__ == "__main__":
    import pathlib
    import radar
    import matplotlib.pyplot as plt

    from src import config_param
    from src import helper_functions

    """ radar part """
    files = [pathlib.Path("..") / "data_files" / "Example" / "Site  1 Antenna 1 Sub-Channel Rx-1 Rays.str",
             pathlib.Path("..") / "data_files" / "Example" / "Site  1 Antenna 1 Sub-Channel Rx-2 Rays.str",
             pathlib.Path("..") / "data_files" / "Example" / "Site  1 Antenna 1 Sub-Channel Rx-3 Rays.str",
             pathlib.Path("..") / "data_files" / "Example" / "Site  1 Antenna 1 Sub-Channel Rx-4 Rays.str"]

    radar_cube = radar.generate_radar_cube(file_location_list=files,
                                           sampling_frequency=config_param.sampling_frequency,
                                           fft_range=config_param.fft_samples_range,
                                           number_of_chirps=config_param.number_chirps,
                                           fft_chirps=config_param.fft_samples_chirps,
                                           number_of_antennas=config_param.number_rx_antennas,
                                           fft_antennas=config_param.fft_samples_antenna,
                                           ampl_tx=config_param.transmit_amplitude,
                                           chirp_duration=config_param.chirp_time,
                                           bandwidth=config_param.bandwidth,
                                           phi_rx=config_param.phi_rx,
                                           phi_0=config_param.phi_0,
                                           wanted_timestep=0,
                                           dead_time=0,
                                           lp_deadband_gain=0)

    # add noise
    radar_cube_noisy = radar.add_phase_noise(radar_cube, mu_phase_noise=config_param.mu_phase_noise,
                                             sigma_phase_noise=config_param.sigma_phase_noise)

    # window radar cube
    radar_cube_windowed_range = radar.window_direction_range(radar_cube_noisy)
    radar_cube_windowed_doppler = radar.window_direction_doppler(radar_cube_noisy)
    radar_cube_windowed_ran_dop = radar.window_direction_doppler(radar_cube_windowed_range)

    # Digital Signal Processing

    # FFT and noise
    radar_cube = apply_3d_fft(cube=radar_cube,
                              fft_range=config_param.fft_samples_range,
                              fft_doppler=config_param.fft_samples_chirps,
                              fft_angle=config_param.fft_samples_antenna)

    radar_cube_noisy = apply_3d_fft(cube=radar_cube_noisy,
                                    fft_range=config_param.fft_samples_range,
                                    fft_doppler=config_param.fft_samples_chirps,
                                    fft_angle=config_param.fft_samples_antenna)
    # radar_cube_noisy = add_ampl_noise(radar_cube=radar_cube_noisy,
    #                                   mu_ampl_noise=config_param.mu_ampl_noise,
    #                                   sigma_ampl_noise=config_param.sigma_ampl_noise)

    radar_cube_windowed_range = apply_3d_fft(cube=radar_cube_windowed_range,
                                             fft_range=config_param.fft_samples_range,
                                             fft_doppler=config_param.fft_samples_chirps,
                                             fft_angle=config_param.fft_samples_antenna)
    # radar_cube_windowed_range = add_ampl_noise(radar_cube=radar_cube_windowed_range,
    #                                            mu_ampl_noise=config_param.mu_ampl_noise,
    #                                            sigma_ampl_noise=config_param.sigma_ampl_noise)

    radar_cube_windowed_doppler = apply_3d_fft(cube=radar_cube_windowed_doppler,
                                               fft_range=config_param.fft_samples_range,
                                               fft_doppler=config_param.fft_samples_chirps,
                                               fft_angle=config_param.fft_samples_antenna)
    # radar_cube_windowed_doppler = add_ampl_noise(radar_cube=radar_cube_windowed_doppler,
    #                                              mu_ampl_noise=config_param.mu_ampl_noise,
    #                                              sigma_ampl_noise=config_param.sigma_ampl_noise)

    radar_cube_windowed_ran_dop = apply_3d_fft(cube=radar_cube_windowed_ran_dop,
                                               fft_range=config_param.fft_samples_range,
                                               fft_doppler=config_param.fft_samples_chirps,
                                               fft_angle=config_param.fft_samples_antenna)
    # radar_cube_windowed_ran_dop = add_ampl_noise(radar_cube=radar_cube_windowed_ran_dop,
    #                                              mu_ampl_noise=config_param.mu_ampl_noise,
    #                                              sigma_ampl_noise=config_param.sigma_ampl_noise)

    # Create x and y values
    f_carr = radar.get_carrier_freq(files[0])
    range_m, doppler_range, doppler_speed, angle_range, angle_degree = (
        helper_functions.create_radar_labels(carrier_freq=f_carr,
                                             sampling_frequency=config_param.sampling_frequency,
                                             bandwidth=config_param.bandwidth,
                                             chirp_time=config_param.chirp_time,
                                             distance_antennas=config_param.distance_antennas,
                                             fft_samples_range=config_param.fft_samples_range,
                                             fft_samples_doppler=config_param.fft_samples_chirps,
                                             fft_samples_antenna=config_param.fft_samples_antenna))

    # CFAR Range
    rm_cfar = np.abs(radar_cube_windowed_range[:, 0, 0])
    rm_cfar = rm_cfar / rm_cfar.max()
    rm_cfar_detections, rm_cfar_threshold = cfar_range(rm_cfar, 10)

    # CFAR Range Doppler
    rdm_cfar = np.abs(radar_cube_windowed_ran_dop[:, :, 0])
    rdm_cfar = rdm_cfar / rdm_cfar.max()
    rdm_cfar_detections, rdm_cfar_threshold = cfar_range_doppler(rdm_cfar, 13)

    # CFAR Range Angular
    ram_cfar = np.abs(radar_cube_windowed_range[:, 0, :])
    ram_cfar = ram_cfar / ram_cfar.max()
    ram_cfar_detections, ram_cfar_threshold = cfar_range_angular(ram_cfar, 14.5)

    """ 
    
    Visualization 
    
    """
    chirp_number = 0
    anten_number = 0

    fig, axs = plt.subplots(1, 4)
    # Plotting range map
    rm = np.abs(radar_cube[:, chirp_number, anten_number])
    rm = helper_functions.pow2db(rm / rm.max())
    axs[0].plot(range_m, rm)
    axs[0].set_title("Range map")
    axs[0].grid(True)

    rm = np.abs(radar_cube_noisy[:, chirp_number, anten_number])
    rm = helper_functions.pow2db(rm / rm.max())
    axs[1].plot(range_m, rm)
    axs[1].set_title("Noisy range map")
    axs[1].grid(True)

    rm = np.abs(radar_cube_windowed_range[:, chirp_number, anten_number])
    rm = helper_functions.pow2db(rm / rm.max())
    axs[2].plot(range_m, rm)
    axs[2].set_title("Windowed and noisy range map")
    axs[2].grid(True)

    axs[3].plot(range_m, helper_functions.pow2db(rm_cfar), label="Range map", color="tab:blue")
    axs[3].plot(range_m, helper_functions.pow2db(rm_cfar_threshold), label="CFAR threshold", color="green",
                linestyle="-")
    axs[3].plot(range_m, helper_functions.pow2db(rm_cfar_detections), label="CFAR detections", color="red",
                linestyle=None, marker="x", markersize=10)
    axs[3].set_title("CFAR range map")
    axs[3].grid(True)
    axs[3].legend()

    # Range Doppler map
    anten_number = 0

    fig, axs = plt.subplots(2, 2, subplot_kw={"projection": "3d"})
    # Plotting range doppler map
    rdm = np.abs(radar_cube[:, :, anten_number])
    rdm = helper_functions.pow2db(rdm / rdm.max())
    axs[0][0].plot_surface(doppler_range, doppler_speed, rdm.T, cmap=config_param.cmap)
    axs[0][0].set_title("Range doppler map")
    axs[0][0].grid(True)

    rdm = np.abs(radar_cube_noisy[:, :, anten_number])
    rdm = helper_functions.pow2db(rdm / rdm.max())
    axs[0][1].plot_surface(doppler_range, doppler_speed, rdm.T, cmap=config_param.cmap)
    axs[0][1].set_title("Noisy range doppler map")
    axs[0][1].grid(True)

    rdm = np.abs(radar_cube_windowed_ran_dop[:, :, anten_number])
    rdm = helper_functions.pow2db(rdm / rdm.max())
    axs[1][0].plot_surface(doppler_range, doppler_speed, rdm.T, cmap=config_param.cmap)
    axs[1][0].set_title("Windowed and noisy range doppler map")
    axs[1][0].grid(True)

    axs[1][1].plot_surface(doppler_range, doppler_speed, helper_functions.pow2db(rdm_cfar).T, label="Range map",
                           color="tab:blue", alpha=0.1)
    axs[1][1].plot_surface(doppler_range, doppler_speed, helper_functions.pow2db(rdm_cfar_threshold).T,
                           label="CFAR threshold", color="green", alpha=0.3)
    axs[1][1].plot_surface(doppler_range, doppler_speed, helper_functions.pow2db(rdm_cfar_detections).T,
                           label="CFAR detections", color="red", antialiased=False)
    axs[1][1].set_title("CFAR range doppler map")
    axs[1][1].grid(True)
    axs[1][1].legend()

    # Range angular map
    chrip_number = 0

    fig, axs = plt.subplots(2, 2, subplot_kw={"projection": "3d"})
    # Plotting range doppler map
    ram = np.abs(radar_cube[:, chirp_number, :])
    ram = helper_functions.pow2db(ram / ram.max())
    axs[0][0].plot_surface(angle_range, angle_degree, ram.T, cmap=config_param.cmap)
    axs[0][0].set_title("Range angular map")
    axs[0][0].grid(True)

    ram = np.abs(radar_cube_noisy[:, chirp_number, :])
    ram = helper_functions.pow2db(ram / ram.max())
    axs[0][1].plot_surface(angle_range, angle_degree, ram.T, cmap=config_param.cmap)
    axs[0][1].set_title("Noisy range angular map")
    axs[0][1].grid(True)

    ram = np.abs(radar_cube_windowed_range[:, chirp_number, :])
    ram = helper_functions.pow2db(ram / ram.max())
    axs[1][0].plot_surface(angle_range, angle_degree, ram.T, cmap=config_param.cmap)
    axs[1][0].set_title("Range windowed and noisy range angular map")
    axs[1][0].grid(True)

    axs[1][1].plot_surface(angle_range, angle_degree, helper_functions.pow2db(ram_cfar).T, label="Range map",
                           color="tab:blue", alpha=0.1)
    axs[1][1].plot_surface(angle_range, angle_degree, helper_functions.pow2db(ram_cfar_threshold).T,
                           label="CFAR threshold", color="green", alpha=0.3)
    axs[1][1].plot_surface(angle_range, angle_degree, helper_functions.pow2db(ram_cfar_detections).T,
                           label="CFAR detections", color="red", antialiased=False)
    axs[1][1].set_title("CFAR range angular map")
    axs[1][1].grid(True)
    axs[1][1].legend()

    plt.show()
