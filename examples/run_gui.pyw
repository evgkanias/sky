from sky.prague import PragueSkyModel, AvailableData
from sky.render import render, image2texture, pixel2dir
from sky.render import SPECTRUM_CHANNELS, SPECTRUM_WAVELENGTHS, SPECTRUM_STEP, MODES

from PIL import Image

import PySimpleGUI as sg
import pandas as pd
import numpy as np
import time
import os
import io

sg.theme("DarkGrey10")
load_path = os.path.abspath(os.path.join("..", "src", "data", "SWIR.dat"))
save_path = os.path.join("..", "src", "data", "new_test.png")
save_path_changed = False

loading_time = 0
saving_time = 0
rendering_time = 0
drawing_time = 0
render_command = False
draw_command = False
result: {np.ndarray, None} = None
is_rendering = False
rendering_success = False
is_loading = False
loading_success = False
saving_success = False
img_tmp: Image = None

values = {"load_path": load_path, "save_path": save_path}

available = AvailableData(
    albedo_min=0.0,
    albedo_max=1.0,
    altitude_min=0.0,
    altitude_max=15000.0,
    elevation_min=-4.2,
    elevation_max=90.0,
    visibility_min=20.0,
    visibility_max=131.0,
    polarisation=True,
    channels=SPECTRUM_CHANNELS,
    channel_start=SPECTRUM_WAVELENGTHS[0] - 0.5 * SPECTRUM_STEP,
    channel_width=SPECTRUM_STEP
)

modes = [m.capitalize() for m in MODES]
default_mode = 0
default_albedo = 0.50
default_altitude = 0.0
default_azimuth = 0.0
default_elevation = 0.0
default_resolution = 150  # 693
default_visibility = 59.4
default_wavelength = 280
default_exposure = 0.0
default_auto_update = False

text_size = 9
val_size = 9


def get_file_name(path):
    return path.split(os.sep)[-1]


def get_folder_path(path):
    return os.sep.join(path.split(os.sep)[:-1])


def draw_figure(rgb):
    global img_tmp

    exposure = values["exposure"] if "exposure" in values else default_exposure
    if rgb.ndim > 2:
        rgb_ = np.transpose(rgb, axes=(1, 2, 0))
    else:
        rgb_ = rgb
    text = image2texture(rgb_, float(exposure))

    img_tmp = Image.fromarray(np.transpose(text, axes=(1, 0, 2)), mode="RGBA")
    img_resize = img_tmp.resize((600, 600))
    buf = io.BytesIO()
    img_resize.save(buf, format="PNG")
    window["canvas"].update(buf.getvalue())


rgb_init = -np.ones((3, default_resolution, default_resolution), dtype='float32')

left_column = [
    [
        sg.Text("Dataset:")
    ],
    [
        sg.Button(get_file_name(load_path),
                  button_type=sg.BUTTON_TYPE_BROWSE_FILE,
                  file_types=(
                                 ("DAT Files", "*.dat *.data"),
                             ) + sg.FILE_TYPES_ALL_FILES,
                  initial_folder=os.path.join("..", "data"),
                  enable_events=True,
                  target=(1, 0),
                  border_width=0,
                  focus=False,
                  key="load_path",
                  bind_return_key=False,
                  size=(50, 1)),
        sg.Text("file")
    ],
    [
        sg.Button("Load", border_width=0, key="load"),
        sg.Text("", key="load_ok")
    ],
    [
        sg.HSeparator()
    ],
    [
        sg.Text("Configuration:")
    ],
    [
        sg.Slider((available.albedo_min, available.albedo_max),
                  default_value=default_albedo,
                  orientation="horizontal",
                  disable_number_display=True,
                  resolution=0.01,
                  size=(51, 20),
                  change_submits=True,
                  enable_events=True,
                  disabled=True,
                  key="albedo"),
        sg.Text(f"albedo", size=(text_size, 1), key="albedo_tooltip",
                tooltip=f"Ground albedo, value in range [{available.albedo_min:.1f}, {available.albedo_max:.1f}]."),
        sg.Text(f"({default_albedo:.2f})", key="albedo_text", size=(val_size, 1))
    ],
    [
        sg.Slider((available.altitude_min, available.altitude_max),
                  default_value=default_altitude,
                  orientation="horizontal",
                  disable_number_display=True,
                  resolution=1,
                  size=(51, 20),
                  change_submits=True,
                  enable_events=True,
                  disabled=True,
                  key="altitude"),
        sg.Text(f"altitude", size=(text_size, 1), key="altitude_tooltip",
                tooltip=f"Altitude of view point in meters, "
                        f"value in range [{available.altitude_min:.1f}, {available.altitude_max:.1f}]."),
        sg.Text(f"({default_altitude:.0f} m)", key="altitude_text", size=(val_size, 1))
    ],
    [
        sg.Slider((0.0, 360.0),
                  default_value=default_azimuth,
                  orientation="horizontal",
                  disable_number_display=True,
                  resolution=0.1,
                  size=(51, 20),
                  change_submits=True,
                  enable_events=True,
                  disabled=True,
                  key="azimuth"),
        sg.Text(f"azimuth", size=(text_size, 1), key="azimuth_tooltip",
                tooltip=f"Sun azimuth at view point in degrees, value in range [0, 360]."),
        sg.Text(f"({default_azimuth:.1f}°)", key="azimuth_text", size=(val_size, 1))
    ],
    [
        sg.Slider((available.elevation_min, available.elevation_max),
                  default_value=default_elevation,
                  orientation="horizontal",
                  disable_number_display=True,
                  resolution=0.1,
                  size=(51, 20),
                  change_submits=True,
                  enable_events=True,
                  disabled=True,
                  key="elevation"),
        sg.Text(f"elevation", size=(text_size, 1), key="elevation_tooltip",
                tooltip=f"Sun elevation at view point in degrees, "
                        f"value in range [{available.elevation_min:.1f}, {available.elevation_max:.1f}]."),
        sg.Text(f"({default_elevation:.1f}°)", key="elevation_text", size=(val_size, 1))
    ],
    [
        sg.Slider((1, 10000),
                  default_value=default_resolution,
                  orientation="horizontal",
                  disable_number_display=True,
                  resolution=1,
                  size=(51, 20),
                  change_submits=True,
                  enable_events=True,
                  disabled=True,
                  key="resolution"),
        sg.Text(f"resolution", size=(text_size, 1), key="resolution_tooltip",
                tooltip=f"Length of resulting square image size in pixels, "
                        f"value in range [1, 10000]."),
        sg.Text(f"({default_resolution:.0f} px)", key="resolution_text", size=(val_size, 1))
    ],
    [
        sg.Slider((available.visibility_min, available.visibility_max),
                  default_value=default_visibility,
                  orientation="horizontal",
                  disable_number_display=True,
                  resolution=0.1,
                  size=(51, 20),
                  change_submits=False,
                  enable_events=True,
                  disabled=True,
                  key="visibility"),
        sg.Text(f"visibility", size=(text_size, 1), key="visibility_tooltip",
                tooltip=f"Horizontal visibility (meteorological range) at ground level in kilometers, "
                        f"value in range [{available.visibility_min:.1f}, {available.visibility_max:.1f}]."),
        sg.Text(f"({default_visibility:.1f} km)", key="visibility_text", size=(val_size, 1))
    ],
    [
        sg.DropDown(modes,
                    default_value=modes[default_mode],
                    readonly=True,
                    disabled=True,
                    key="mode",
                    size=(49, 1)),
        sg.Text("mode", key="mode_tooltip", tooltip="Rendering quantity.")
    ],
    [
        sg.Button("Render", border_width=0, key="render", disabled=default_auto_update or True),
        sg.Checkbox("Auto-update", key="auto-update", default=default_auto_update, disabled=True, enable_events=True),
        sg.Text("", key="render-status")
    ],
    [
        sg.HSeparator()
    ],
    [
        sg.Text("Channels:")
    ],
    [
        sg.Radio("all visible range in one sRGB image", 2, default=True, enable_events=True, key="wl-visible",
                 disabled=True)
    ],
    [
        sg.Radio("individual wavelength bins", 2, default=False, enable_events=True, key="wl-manual", disabled=True),
    ],
    [
        sg.Slider((280, 2460),
                  default_value=default_wavelength,
                  orientation="horizontal",
                  disable_number_display=True,
                  resolution=20,
                  size=(51, 20),
                  enable_events=True,
                  key="wavelength",
                  disabled=True),
        sg.Text(f"wavelength", size=(text_size, 1), key="wavelength_tooltip",
                tooltip=f"Wavelength determining the displayed wavelength bin."),
        sg.Text(f"({default_wavelength:.0f} nm)", key="wavelength_text", size=(val_size, 1))
    ],
    [
        sg.HSeparator()
    ],
    [
        sg.Text("Display:")
    ],
    [
        sg.Slider((-25, 25),
                  default_value=default_exposure,
                  orientation="horizontal",
                  disable_number_display=True,
                  enable_events=True,
                  resolution=0.1,
                  disabled=True,
                  size=(51, 20),
                  key="exposure"),
        sg.Text("exposure", size=(text_size, 1), key="exposure_tooltip",
                tooltip="Multiplication factor of displayed image values, value in range [-25, 25]."),
        sg.Text(f"({default_exposure:.1f})", key="exposure_text", size=(val_size, 1))
    ],
    [
        sg.HSeparator()
    ],
    [
        sg.Text("Save:")
    ],
    [
        sg.Button(f"{get_file_name(save_path)}",
                  button_type=sg.BUTTON_TYPE_SAVEAS_FILE,
                  file_types=(
                                 ("PNG Files", "*.png"),
                                 ("JPEG Files", "*.jpg *.jpeg"),
                                 ("CSV Files", "*.csv"),
                                 ("EXCEL Files", "*.xls *.xlsx")
                             ) + sg.FILE_TYPES_ALL_FILES,
                  # initial_folder=f"{get_folder_path(save_path)}",
                  initial_folder=save_path,
                  default_extension="PNG",
                  enable_events=True,
                  target=(sg.ThisRow, 0),
                  border_width=0,
                  focus=False,
                  key="save_path",
                  disabled=True,
                  bind_return_key=False,
                  size=(50, 1)),
        sg.Text("file")
    ],
    [
        sg.Button("Save", border_width=0, key="save", disabled=True),
        sg.Text("", key="save_ok")
    ]
]

right_column = [
    [
        sg.Image(filename="", key="canvas", size=(600, 600))
    ]
]

layout = [
    [
        sg.Column(left_column),
        sg.VSeperator(),
        sg.Column(right_column)
    ]
]


# Create the window
window = sg.Window("Prague Sky Model",
                   layout,
                   finalize=True,
                   element_justification="centre",
                   font="Mundial 9")

sky = PragueSkyModel()


def load_local():
    global loading_time, loading_success, is_loading

    is_loading = True
    start = time.time()
    try:
        sky.reset(values["load_path"])
        loading_success = True
    except Exception as e:
        loading_success = False
        print(e)
    end = time.time()
    is_loading = False

    loading_time = end - start


def save_local():
    global saving_success, img_tmp

    if img_tmp is not None and values["save_path"] is not None:
        try:
            extension = save_path.split(".")[-1].lower()
            if extension in ("png", "jpg", "jpeg"):
                img_tmp.save(save_path)
                saving_success = True
            elif extension in ("csv", "xlsx", "xls"):
                xs, ys = np.meshgrid(np.arange(result.shape[1]), np.arange(result.shape[2]))
                resolution = np.maximum(result.shape[1], result.shape[2])

                views_dir = pixel2dir(xs, ys, resolution)
                pixel_map = ~np.all(np.isclose(views_dir, 0), axis=2)
                raw_data = np.hstack([views_dir[pixel_map, :], result[:, pixel_map].T])
                columns = ["x", "y", "z", "R", "G", "B"] + [f"wl-{wl}" for wl in SPECTRUM_WAVELENGTHS]
                df = pd.DataFrame(raw_data, columns=columns)
                if extension in ("csv",):
                    df.to_csv(save_path, sep=",")
                else:
                    df.to_excel(save_path)
                saving_success = True
            else:
                saving_success = False
                print(f"Unsupported file extension: '*.{extension}'.")
        except Exception as e:
            saving_success = False
            print(e)
    else:
        saving_success = False


def render_local():
    global result, rendering_time, is_rendering, rendering_success

    is_rendering = True
    start = time.time()
    try:
        print(values)
        result = render(sky_model=sky, albedo=float(values["albedo"]), altitude=float(values["altitude"]),
                        azimuth=np.deg2rad(values["azimuth"]), elevation=np.deg2rad(values["elevation"]),
                        visibility=float(values["visibility"]),
                        resolution=int(values["resolution"]), mode=values["mode"])
        rendering_success = True
    except Exception as e:
        rendering_success = False
        print(e)
    end = time.time()
    is_rendering = False

    rendering_time = end - start


def load_complete():
    global render_command

    if loading_success:
        window["load_ok"].update(f"Done. ({loading_time:.1f} sec)")
    else:
        window["load_ok"].update(f"Failed. ({loading_time:.1f} sec)")
        return

    # sg.popup_ok("Data loaded!")
    update_available_data(disable=False)

    if values["auto-update"]:
        render_command = True


def save_complete():
    if saving_success:
        window["save_ok"].update(f"Done. ({saving_time:.1f} sec)")
    else:
        window["save_ok"].update(f"Failed. ({saving_time:.1f} sec)")


def render_complete():
    global draw_command, is_rendering

    if rendering_success:
        window["render-status"].update(f"Done. ({rendering_time:.1f} sec)")
    else:
        window["render-status"].update(f"Failed. ({rendering_time:.1f} sec)")

    draw_command = True


def draw_complete():
    pass


def update_available_data(disable=True):
    global available

    available = sky.available_data

    # Update ranges and values, disable or enable changes
    window["albedo"].update(range=(available.albedo_min, available.albedo_max), disabled=disable)
    window["altitude"].update(range=(available.altitude_min, available.altitude_max), disabled=disable)
    window["azimuth"].update(disabled=disable)
    window["elevation"].update(range=(available.elevation_min, available.elevation_max), disabled=disable)
    window["resolution"].update(disabled=disable)
    window["visibility"].update(range=(available.visibility_min, available.visibility_max), disabled=disable)

    if not available.polarisation and "Polarisation" in modes:
        modes.remove("Polarisation")
    elif available.polarisation and "Polarisation" not in modes:
        modes.insert(2, "Polarisation")

    window["mode"].update(values=modes, value=modes[default_mode], disabled=disable)
    window["render"].update(disabled=disable or default_auto_update)
    window["auto-update"].update(disabled=disable)
    window["wl-visible"].update(disabled=disable)
    window["wl-manual"].update(disabled=disable)
    window["wavelength"].update(range=(available.channel_start,
                                       available.channel_start + available.channels * available.channel_width - 1),
                                disabled=disable or values["wl-visible"])
    window["exposure"].update(disabled=disable)
    window["save_path"].update(disabled=disable)
    window["save"].update(disabled=disable)

    # Update tooltips
    window["albedo_tooltip"].TooltipObject.text = (
        f"Ground albedo, value in range [{available.albedo_min:.1f}, {available.albedo_max:.1f}].")
    window["altitude_tooltip"].TooltipObject.text = (
        f"Altitude of view point in meters, "
        f"value in range [{available.altitude_min:.1f}, {available.altitude_max:.1f}].")
    window["azimuth_tooltip"].TooltipObject.text = (
        f"Sun azimuth at view point in degrees, value in range [0, 360].")
    window["elevation_tooltip"].TooltipObject.text = (
        f"Sun elevation at view point in degrees, "
        f"value in range [{available.elevation_min:.1f}, {available.elevation_max:.1f}].")
    window["visibility_tooltip"].TooltipObject.text = (
        f"Horizontal visibility (meteorological range) at ground level in kilometers, "
        f"value in range [{available.visibility_min:.1f}, {available.visibility_max:.1f}].")
    window["resolution_tooltip"].TooltipObject.text = (
        f"Length of resulting square image size in pixels, value in range [1, 10000].")
    window["albedo_tooltip"].TooltipObject.text = (
        f"Ground albedo, value in range [{available.albedo_min:.1f}, {available.albedo_max:.1f}].")
    window["wavelength_tooltip"].TooltipObject.text = (
        f"Wavelength determining the displayed wavelength bin.")


def main(*args):
    global result, rgb_init, values, load_path, save_path, available, render_command, draw_command, is_rendering

    draw_figure(rgb_init)

    while True:
        event, values = window.read()
        # print(event, values)
        render_command = False
        draw_command = False

        # End program if user closes window or presses the OK button
        if event == sg.WIN_CLOSED:
            break
        elif event == "render":
            render_command = loading_success
        elif event == "wl-manual":
            window["wavelength"].update(disabled=False)
            draw_command = rendering_success
        elif event == "wl-visible":
            window["wavelength"].update(disabled=True)
            draw_command = rendering_success
        elif event == "auto-update":
            window["render"].update(disabled=values["auto-update"])
        elif event == "albedo":
            window["albedo_text"].update(f"({values['albedo']:.2f})")
            if values["auto-update"]:
                render_command = loading_success
        elif event == "altitude":
            window["altitude_text"].update(f"({values['altitude']:.0f} m)")
            if values["auto-update"]:
                render_command = loading_success
        elif event == "azimuth":
            window["azimuth_text"].update(f"({values['azimuth']:.1f}°)")
            if values["auto-update"]:
                render_command = loading_success
        elif event == "elevation":
            window["elevation_text"].update(f"({values['elevation']:.1f}°)")
            if values["auto-update"]:
                render_command = loading_success
        elif event == "resolution":
            window["resolution_text"].update(f"({values['resolution']:.0f} px)")
            if values["auto-update"]:
                render_command = loading_success
        elif event == "visibility":
            window["visibility_text"].update(f"({values['visibility']:.1f} km)")
            if values["auto-update"]:
                render_command = loading_success
        elif event == "wavelength":
            window["wavelength_text"].update(f"({values['wavelength']:.0f} nm)")
            draw_command = rendering_success
        elif event == "exposure":
            window["exposure_text"].update(f"({values['exposure']:.1f})")
            draw_command = rendering_success
        elif event == "load_path":
            load_path = values["load_path"].replace("/", os.sep)  # correct for the "linux-only" bug
            window["load_path"].update(text=get_file_name(load_path))
            window["load_ok"].update("")
        elif event == "load":
            if values["load_path"] == "":  # default path
                values["load_path"] = load_path
                window["load_ok"].update(f"Loading...")
            try:
                window.perform_long_operation(load_local, "LOAD COMPLETE")
            except Exception as e:
                window["load_ok"].update("FAILED")
                print(e)
        elif event == "save_path":
            save_path = values["save_path"].replace("/", os.sep)  # correct for the "linux-only" bug
            window["save_path"].update(text=get_file_name(save_path))
            window["save_ok"].update("")
        elif event == "save":
            if values["save_path"] == "":  # default path
                values["save_path"] = save_path
            try:
                window.perform_long_operation(save_local, "SAVE COMPLETE")  # save the data
            except Exception as e:
                window["save_ok"].update("Failed.")
                print(e)
        elif event == "LOAD COMPLETE":
            load_complete()
        elif event == "SAVE COMPLETE":
            save_complete()
        elif event == "RENDER COMPLETE":
            render_complete()
        elif event == "DRAW COMPLETE":
            draw_complete()

        if render_command and not is_rendering:
            window["render-status"].update("Rendering...")
            window.perform_long_operation(render_local, "RENDER COMPLETE")
            is_rendering = True

        if draw_command:
            if values["wl-manual"]:
                wl_idx = np.searchsorted(SPECTRUM_WAVELENGTHS, values["wavelength"], side="right")
                draw_f = lambda: draw_figure(result[3 + wl_idx])
            else:
                draw_f = lambda: draw_figure(result[:3])
            window.perform_long_operation(draw_f, "DRAW COMPLETE")

    window.close()


if __name__ == "__main__":
    import sys

    main(*sys.argv)
