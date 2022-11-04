from sky.prague import PragueSkyModel
from sky.render import render, image2texture, SPECTRUM_WAVELENGTHS

from PIL import Image

import PySimpleGUI as sg
import numpy as np
import time
import os
import io

sg.theme("DarkGrey10")
load_path = os.path.join("..", "data", "SWIR.dat")
# load_path = os.path.join("..", "data", "full.dat")
save_path = os.path.join("..", "data", "new_test.png")
save_path_changed = False

values = {"load_path": load_path, "save_path": save_path}

default_albedo = 0.50
default_altitude = 0
default_azimuth = 0
default_elevation = 0.0
default_resolution = 150  # 693
default_visibility = 59.4
default_wavelength = 280
default_exposure = 0.0
default_zoom = 1
auto_update = False

text_size = 9
val_size = 9


def get_file_name(path):
    return path.split(os.sep)[-1]


def draw_figure(rgb):
    exposure = values["exposure"] if "exposure" in values else default_exposure
    if rgb.ndim > 2:
        rgb_ = np.transpose(rgb, axes=(1, 2, 0))
    else:
        rgb_ = rgb
    text = image2texture(rgb_, float(exposure))
    print(text.min(), text.max(), text.shape)
    img = Image.fromarray(np.transpose(text, axes=(1, 0, 2)), mode="RGBA")
    img_resize = img.resize((600, 600))
    buf = io.BytesIO()
    img_resize.save(buf, format="PNG")
    window["canvas"].update(buf.getvalue())


rgb_init = np.zeros((3, default_resolution, default_resolution), dtype='float32')

left_column = [
    [
        sg.Text("Dataset:")
    ],
    [
        sg.Button(get_file_name(load_path),
                  button_type=sg.BUTTON_TYPE_BROWSE_FILE,
                  file_types=sg.FILE_TYPES_ALL_FILES,
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
        sg.DropDown(["Everything", "Sky radiance", "Sky transmittance", "Sky polarisation"],
                    "Everything",
                    size=(49, 1),
                    readonly=True,
                    key="part_to_load"),
        sg.Text("part to load")
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
        sg.DropDown(["Sky radiance", "Sun radiance", "Polarisation", "Transmittance"],
                    default_value="Sky radiance",
                    readonly=True,
                    key="mode",
                    size=(49, 1)),
        sg.Text("mode")
    ],
    [
        sg.Slider((0, 1),
                  default_value=default_albedo,
                  orientation="horizontal",
                  disable_number_display=True,
                  resolution=0.01,
                  size=(51, 20),
                  change_submits=True,
                  enable_events=True,
                  key="albedo"),
        sg.Text(f"albedo", size=(text_size, 1)),
        sg.Text(f"({default_albedo:.2f})", key="albedo_text", size=(val_size, 1))
    ],
    [
        sg.Slider((0, 15000),
                  default_value=default_altitude,
                  orientation="horizontal",
                  disable_number_display=True,
                  resolution=1,
                  size=(51, 20),
                  change_submits=True,
                  enable_events=True,
                  key="altitude"),
        sg.Text(f"altitude", size=(text_size, 1)),
        sg.Text(f"({default_altitude:.0f} m)", key="altitude_text", size=(val_size, 1))
    ],
    [
        sg.Slider((0, 360),
                  default_value=default_azimuth,
                  orientation="horizontal",
                  disable_number_display=True,
                  resolution=0.1,
                  size=(51, 20),
                  change_submits=True,
                  enable_events=True,
                  key="azimuth"),
        sg.Text(f"azimuth", size=(text_size, 1)),
        sg.Text(f"({default_azimuth:.1f}°)", key="azimuth_text", size=(val_size, 1))
    ],
    [
        sg.Slider((-4.2, 90),
                  default_value=default_elevation,
                  orientation="horizontal",
                  disable_number_display=True,
                  resolution=0.1,
                  size=(51, 20),
                  change_submits=True,
                  enable_events=True,
                  key="elevation"),
        sg.Text(f"elevation", size=(text_size, 1)),
        sg.Text(f"({default_elevation:.1f}°)", key="elevation_text", size=(val_size, 1))
    ],
    [
        sg.Slider((0, 1024),
                  default_value=default_resolution,
                  orientation="horizontal",
                  disable_number_display=True,
                  resolution=1,
                  size=(51, 20),
                  change_submits=True,
                  enable_events=True,
                  key="resolution"),
        sg.Text(f"resolution", size=(text_size, 1)),
        sg.Text(f"({default_resolution:.0f} px)", key="resolution_text", size=(val_size, 1))
    ],
    [
        sg.Slider((20, 131.8),
                  default_value=default_visibility,
                  orientation="horizontal",
                  disable_number_display=True,
                  resolution=0.1,
                  size=(51, 20),
                  change_submits=True,
                  enable_events=True,
                  key="visibility"),
        sg.Text(f"visibility", size=(text_size, 1)),
        sg.Text(f"({default_visibility:.1f} km)", key="visibility_text", size=(val_size, 1))
    ],
    [
        sg.Button("Render", border_width=0, key="render", disabled=auto_update),
        sg.Checkbox("Auto-update", key="auto-update", default=auto_update, enable_events=True),
        sg.Text("", key="render-status")
    ],
    [
        sg.HSeparator()
    ],
    [
        sg.Text("Channels:")
    ],
    [
        sg.Radio("all visible range in one sRGB image", 2, default=True, enable_events=True, key="wl-visible")
    ],
    [
        sg.Radio("individual wavelength bins", 2, default=False, enable_events=True, key="wl-manual"),
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
        sg.Text(f"wavelength", size=(text_size, 1)),
        sg.Text(f"({default_wavelength:.0f} nm)", key="wavelength_text", size=(val_size, 1))
    ],
    [
        sg.HSeparator()
    ],
    [
        sg.Text("Display:")
    ],
    [
        sg.Slider((-10, 10),
                  default_value=default_exposure,
                  orientation="horizontal",
                  disable_number_display=True,
                  enable_events=True,
                  resolution=0.1,
                  size=(51, 20),
                  key="exposure"),
        sg.Text("exposure", size=(text_size, 1)),
        sg.Text(f"({default_exposure:.1f})", key="exposure_text", size=(val_size, 1))
    ],
    [
        sg.Slider((0, 10),
                  default_value=default_zoom,
                  orientation="horizontal",
                  disable_number_display=True,
                  resolution=1,
                  enable_events=True,
                  size=(51, 20),
                  key="zoom"),
        sg.Text("zoom", size=(text_size, 1)),
        sg.Text(f"({default_zoom:.1f}x)", key="zoom_text", size=(val_size, 1))
    ],
    [
        sg.HSeparator()
    ],
    [
        sg.Text("Save:")
    ],
    [
        sg.Button(f"{save_path.split(os.sep)[-1]}",
                  button_type=sg.BUTTON_TYPE_BROWSE_FILE,
                  file_types=sg.FILE_TYPES_ALL_FILES,
                  enable_events=True,
                  target=(1, 0),
                  border_width=0,
                  focus=False,
                  key="save_path",
                  bind_return_key=False,
                  size=(50, 1)),
        sg.Text("file")
    ],
    [
        sg.Button("Save", border_width=0, key="save"),
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
window = sg.Window("Sky Model",
                   layout,
                   finalize=True,
                   element_justification="centre",
                   font="Mundial 9")

sky = PragueSkyModel()


def load_func():
    global loading_time

    start = time.time()
    sky.reset(values["load_path"])
    end = time.time()
    loading_time = end - start


def render_local():
    global result, rendering_time

    start = time.time()
    result = render(sky_model=sky, albedo=float(values["albedo"]), altitude=float(values["altitude"]),
                    azimuth=float(values["azimuth"]), elevation=float(values["elevation"]),
                    visibility=float(values["visibility"]),
                    resolution=int(values["resolution"]), mode=values["mode"])
    end = time.time()

    rendering_time = end - start


result = None
loading_time = 0
saving_time = 0
rendering_time = 0
drawing_time = 0

draw_figure(rgb_init)

while True:
    event, values = window.read()
    print(event, values)
    render_command = False
    draw_command = False

    # End program if user closes window or presses the OK button
    if event == sg.WIN_CLOSED:
        break
    elif event == "render":
        render_command = True
    elif event == "wl-manual":
        window["wavelength"].update(disabled=False)
        draw_command = True
    elif event == "wl-visible":
        window["wavelength"].update(disabled=True)
        draw_command = True
    elif event == "auto-update":
        window["render"].update(disabled=values["auto-update"])
    elif event == "albedo":
        window["albedo_text"].update(f"({values['albedo']:.2f})")
        if values["auto-update"]:
            render_command = True
    elif event == "altitude":
        window["altitude_text"].update(f"({values['altitude']:.0f} m)")
        if values["auto-update"]:
            render_command = True
    elif event == "azimuth":
        window["azimuth_text"].update(f"({values['azimuth']:.1f}°)")
        if values["auto-update"]:
            render_command = True
    elif event == "elevation":
        window["elevation_text"].update(f"({values['elevation']:.1f}°)")
        if values["auto-update"]:
            render_command = True
    elif event == "resolution":
        window["resolution_text"].update(f"({values['resolution']:.0f} px)")
        if values["auto-update"]:
            render_command = True
    elif event == "visibility":
        window["visibility_text"].update(f"({values['visibility']:.1f} km)")
        if values["auto-update"]:
            render_command = True
    elif event == "wavelength":
        window["wavelength_text"].update(f"({values['wavelength']:.0f} nm)")
        draw_command = True
    elif event == "exposure":
        window["exposure_text"].update(f"({values['exposure']:.1f})")
        draw_command = True
    elif event == "zoom":
        window["zoom_text"].update(f"({values['zoom']:.1f}x)")
    elif event == "data_path":
        load_path = values["data_path"].replace("/", os.sep)  # correct for the "linux-only" bug
        window["data_path"].update(text=get_file_name(load_path))
        window["load_ok"].update("")
    elif event == "load":
        if values["load_path"] == "":  # default path
            values["load_path"] = load_path
            window["load_ok"].update(f"Loading...")
        try:
            window.perform_long_operation(load_func, "LOAD COMPLETE")
        except Exception as e:
            window["load_ok"].update("FAILED")
            print(e)
    elif event == "save":
        if values["save_path"] == "":  # default path
            values["save_path"] = save_path
        try:
            pass
            # window.perform_long_operation(..., "SAVE COMPLETE")  # save the data
        except Exception as e:
            window["save_ok"].update("Failed.")
            print(e)
    elif event == "LOAD COMPLETE":
        # sg.popup_ok("Data loaded!")
        window["load_ok"].update(f"Done. ({loading_time:.1f} sec)")
        if values["auto-update"]:
            render_command = True
    elif event == "SAVE COMPLETE":
        window["save_ok"].update(f"Done. ({saving_time:.1f} sec)")
    elif event == "RENDER COMPLETE":
        window["render-status"].update(f"Done. ({rendering_time:.1f} sec)")
        draw_command = True
    elif event == "DRAW COMPLETE":
        print("Drawing completed!")

    if render_command and window["render-status"].get() != "Rendering...":
        window["render-status"].update("Rendering...")
        window.perform_long_operation(render_local, "RENDER COMPLETE")

    if draw_command:
        if values["wl-manual"]:
            wl_idx = np.searchsorted(SPECTRUM_WAVELENGTHS, values["wavelength"], side="right")
            draw_f = lambda: draw_figure(result[3 + wl_idx])
        else:
            draw_f = lambda: draw_figure(result[:3])
        window.perform_long_operation(draw_f, "DRAW COMPLETE")


window.close()
