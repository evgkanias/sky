from prague.prague import PragueSkyModel, AvailableData
from sky.prague.render import render, image2texture, pixel2dir
from sky.prague.render import SPECTRUM_CHANNELS, SPECTRUM_WAVELENGTHS, SPECTRUM_STEP, MODES

from PIL import Image

import PySimpleGUI as sg
import pandas as pd
import numpy as np
import time
import os
import io

# use theme
sg.theme("DarkBlack")


class SkyModelGUI:

    def __init__(self):

        # load icon
        icon_img: Image = Image.open(os.path.join("..", "src", "data", "icon.png"))
        self.icon = io.BytesIO()
        icon_img.save(self.icon, format="PNG")

        # default load and save paths
        self.load_path = os.path.expanduser("PragueSkyModelDatasetGroundInfra.dat")
        # self.load_path = os.path.abspath(os.path.join("..", "src", "data", "PragueSkyModelDatasetGroundInfra.dat"))
        self.save_path = os.path.expanduser("new_test.png")
        self.save_path_changed = False

        self.loading_time = 0
        self.saving_time = 0
        self.rendering_time = 0
        self.drawing_time = 0
        self.render_command = False
        self.draw_command = False
        self.result: {np.ndarray, None} = None
        self.is_rendering = False
        self.rendering_success = False
        self.rendering_error = None
        self.is_loading = False
        self.loading_success = False
        self.loading_error = None
        self.saving_success = False
        self.saving_error = None
        self.img_tmp: Image = None

        self.values = {"load_path": self.load_path, "save_path": self.save_path}

        self.available = AvailableData(
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

        self.modes = [m.capitalize() for m in MODES]
        self.default_mode = 0
        self.default_albedo = 0.50
        self.default_altitude = 0.0
        self.default_azimuth = 0.0
        self.default_elevation = 0.0
        self.default_resolution = 300
        self.default_visibility = 59.4
        self.default_wavelength = 280
        self.default_exposure = 0.0
        self.default_auto_update = False

        self.text_size = 9
        self.val_size = 9

        rgb_init = -np.ones((3, self.default_resolution, self.default_resolution), dtype='float32')

        left_column = [
            [
                sg.Text("Dataset:")
            ],
            [
                sg.Button(get_file_name(self.load_path),
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
                sg.Slider((self.available.albedo_min, self.available.albedo_max),
                          default_value=self.default_albedo,
                          orientation="horizontal",
                          disable_number_display=True,
                          resolution=0.01,
                          size=(51, 20),
                          change_submits=True,
                          enable_events=True,
                          disabled=True,
                          key="albedo"),
                sg.Text(f"albedo", size=(self.text_size, 1), key="albedo_tooltip",
                        tooltip=f"Ground albedo, value in range [{self.available.albedo_min:.1f}, "
                                f"{self.available.albedo_max:.1f}]."),
                sg.Text(f"({self.default_albedo:.2f})", key="albedo_text", size=(self.val_size, 1))
            ],
            [
                sg.Slider((self.available.altitude_min, self.available.altitude_max),
                          default_value=self.default_altitude,
                          orientation="horizontal",
                          disable_number_display=True,
                          resolution=1,
                          size=(51, 20),
                          change_submits=True,
                          enable_events=True,
                          disabled=True,
                          key="altitude"),
                sg.Text(f"altitude", size=(self.text_size, 1), key="altitude_tooltip",
                        tooltip=f"Altitude of view point in meters, "
                                f"value in range [{self.available.altitude_min:.1f}, "
                                f"{self.available.altitude_max:.1f}]."),
                sg.Text(f"({self.default_altitude:.0f} m)", key="altitude_text", size=(self.val_size, 1))
            ],
            [
                sg.Slider((0.0, 360.0),
                          default_value=self.default_azimuth,
                          orientation="horizontal",
                          disable_number_display=True,
                          resolution=0.1,
                          size=(51, 20),
                          change_submits=True,
                          enable_events=True,
                          disabled=True,
                          key="azimuth"),
                sg.Text(f"azimuth", size=(self.text_size, 1), key="azimuth_tooltip",
                        tooltip=f"Sun azimuth at view point in degrees, value in range [0, 360]."),
                sg.Text(f"({self.default_azimuth:.1f}°)", key="azimuth_text", size=(self.val_size, 1))
            ],
            [
                sg.Slider((self.available.elevation_min, self.available.elevation_max),
                          default_value=self.default_elevation,
                          orientation="horizontal",
                          disable_number_display=True,
                          resolution=0.1,
                          size=(51, 20),
                          change_submits=True,
                          enable_events=True,
                          disabled=True,
                          key="elevation"),
                sg.Text(f"elevation", size=(self.text_size, 1), key="elevation_tooltip",
                        tooltip=f"Sun elevation at view point in degrees, "
                                f"value in range [{self.available.elevation_min:.1f}, "
                                f"{self.available.elevation_max:.1f}]."),
                sg.Text(f"({self.default_elevation:.1f}°)", key="elevation_text", size=(self.val_size, 1))
            ],
            [
                sg.Slider((1, 10000),
                          default_value=self.default_resolution,
                          orientation="horizontal",
                          disable_number_display=True,
                          resolution=1,
                          size=(51, 20),
                          change_submits=True,
                          enable_events=True,
                          disabled=True,
                          key="resolution"),
                sg.Text(f"resolution", size=(self.text_size, 1), key="resolution_tooltip",
                        tooltip=f"Length of resulting square image size in pixels, "
                                f"value in range [1, 10000]."),
                sg.Text(f"({self.default_resolution:.0f} px)", key="resolution_text", size=(self.val_size, 1))
            ],
            [
                sg.Slider((self.available.visibility_min, self.available.visibility_max),
                          default_value=self.default_visibility,
                          orientation="horizontal",
                          disable_number_display=True,
                          resolution=0.1,
                          size=(51, 20),
                          change_submits=False,
                          enable_events=True,
                          disabled=True,
                          key="visibility"),
                sg.Text(f"visibility", size=(self.text_size, 1), key="visibility_tooltip",
                        tooltip=f"Horizontal visibility (meteorological range) at ground level in kilometers, "
                                f"value in range [{self.available.visibility_min:.1f}, "
                                f"{self.available.visibility_max:.1f}]."),
                sg.Text(f"({self.default_visibility:.1f} km)", key="visibility_text", size=(self.val_size, 1))
            ],
            [
                sg.DropDown(self.modes,
                            default_value=self.modes[self.default_mode],
                            readonly=True,
                            disabled=True,
                            key="mode",
                            size=(49, 1)),
                sg.Text("mode", key="mode_tooltip", tooltip="Rendering quantity.")
            ],
            [
                sg.Button("Render", border_width=0, key="render", disabled=self.default_auto_update or True),
                sg.Checkbox("Auto-update", key="auto-update", default=self.default_auto_update, disabled=True,
                            enable_events=True),
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
                sg.Radio("individual wavelength bins", 2, default=False, enable_events=True, key="wl-manual",
                         disabled=True),
            ],
            [
                sg.Slider((280, 2460),
                          default_value=self.default_wavelength,
                          orientation="horizontal",
                          disable_number_display=True,
                          resolution=20,
                          size=(51, 20),
                          enable_events=True,
                          key="wavelength",
                          disabled=True),
                sg.Text(f"wavelength", size=(self.text_size, 1), key="wavelength_tooltip",
                        tooltip=f"Wavelength determining the displayed wavelength bin."),
                sg.Text(f"({self.default_wavelength:.0f} nm)", key="wavelength_text", size=(self.val_size, 1))
            ],
            [
                sg.HSeparator()
            ],
            [
                sg.Text("Display:")
            ],
            [
                sg.Slider((-25, 25),
                          default_value=self.default_exposure,
                          orientation="horizontal",
                          disable_number_display=True,
                          enable_events=True,
                          resolution=0.1,
                          disabled=True,
                          size=(51, 20),
                          key="exposure"),
                sg.Text("exposure", size=(self.text_size, 1), key="exposure_tooltip",
                        tooltip="Multiplication factor of displayed image values, value in range [-25, 25]."),
                sg.Text(f"({self.default_exposure:.1f})", key="exposure_text", size=(self.val_size, 1))
            ],
            [
                sg.HSeparator()
            ],
            [
                sg.Text("Save:")
            ],
            [
                sg.Button(f"{get_file_name(self.save_path)}",
                          button_type=sg.BUTTON_TYPE_SAVEAS_FILE,
                          file_types=(
                                         ("PNG Files", "*.png"),
                                         ("JPEG Files", "*.jpg *.jpeg"),
                                         ("CSV Files", "*.csv"),
                                         ("EXCEL Files", "*.xlsx")
                                     ) + sg.FILE_TYPES_ALL_FILES,
                          # initial_folder=f"{get_folder_path(save_path)}",
                          initial_folder=self.save_path,
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
        right_column = [[sg.Image(filename="", key="canvas", size=(600, 600))]]
        layout = [
            [
                sg.Column(left_column),
                # sg.VSeperator(),
                sg.Column(right_column)
            ]
        ]

        # Create the window
        self.window = sg.Window("Prague Sky Model",
                                layout,
                                finalize=True,
                                element_justification="centre",
                                font="Mundial 9",
                                icon=self.icon.getvalue(),
                                grab_anywhere_using_control=True)

        self.sky = PragueSkyModel()

        self.draw_figure(rgb_init)

    def __call__(self):

        while True:
            event, self.values = self.window.read()
            # print(event, values)
            self.render_command = False
            self.draw_command = False

            # End program if user closes window or presses the OK button
            if event == sg.WIN_CLOSED:
                break
            elif event == "render":
                self.render_command = self.loading_success
            elif event == "wl-manual":
                self.window["wavelength"].update(disabled=False)
                self.draw_command = self.rendering_success
            elif event == "wl-visible":
                self.window["wavelength"].update(disabled=True)
                self.draw_command = self.rendering_success
            elif event == "auto-update":
                self.window["render"].update(disabled=self.values["auto-update"])
            elif event == "albedo":
                self.window["albedo_text"].update(f"({self.values['albedo']:.2f})")
                if self.values["auto-update"]:
                    self.render_command = self.loading_success
            elif event == "altitude":
                self.window["altitude_text"].update(f"({self.values['altitude']:.0f} m)")
                if self.values["auto-update"]:
                    self.render_command = self.loading_success
            elif event == "azimuth":
                self.window["azimuth_text"].update(f"({self.values['azimuth']:.1f}°)")
                if self.values["auto-update"]:
                    self.render_command = self.loading_success
            elif event == "elevation":
                self.window["elevation_text"].update(f"({self.values['elevation']:.1f}°)")
                if self.values["auto-update"]:
                    self.render_command = self.loading_success
            elif event == "resolution":
                self.window["resolution_text"].update(f"({self.values['resolution']:.0f} px)")
                if self.values["auto-update"]:
                    self.render_command = self.loading_success
            elif event == "visibility":
                self.window["visibility_text"].update(f"({self.values['visibility']:.1f} km)")
                if self.values["auto-update"]:
                    self.render_command = self.loading_success
            elif event == "wavelength":
                self.window["wavelength_text"].update(f"({self.values['wavelength']:.0f} nm)")
                self.draw_command = self.rendering_success
            elif event == "exposure":
                self.window["exposure_text"].update(f"({self.values['exposure']:.1f})")
                self.draw_command = self.rendering_success
            elif event == "load_path":
                self.load_path = self.values["load_path"].replace("/", os.sep)  # correct for the "linux-only" bug
                self.window["load_path"].update(text=get_file_name(self.load_path))
                self.window["load_ok"].update("")
                event = "load"
            elif event == "save_path":
                self.save_path = self.values["save_path"].replace("/", os.sep)  # correct for the "linux-only" bug
                self.window["save_path"].update(text=get_file_name(self.save_path))
                self.window["save_ok"].update("")
                event = "save"

            if event == "load":
                if self.values["load_path"] == "":  # default path
                    self.values["load_path"] = self.load_path
                self.loading_error = None
                try:
                    self.window.perform_long_operation(self.load_file, "LOAD COMPLETE")
                except Exception as e:
                    self.window["load_ok"].update("Failed.")
                    self.loading_error = e
                    event = "LOAD COMPLETE"
            elif event == "save":
                if self.values["save_path"] == "":  # default path
                    self.values["save_path"] = self.save_path
                self.saving_error = None
                try:
                    self.window.perform_long_operation(self.save_image, "SAVE COMPLETE")  # save the data
                except Exception as e:
                    self.window["save_ok"].update("Failed.")
                    self.saving_error = e
                    event = "SAVE COMPLETE"

            if event == "LOAD COMPLETE":
                self.load_complete()
            elif event == "SAVE COMPLETE":
                self.save_complete()
            elif event == "RENDER COMPLETE":
                self.render_complete()
            elif event == "DRAW COMPLETE":
                self.draw_complete()

            if self.render_command and not self.is_rendering:
                self.window["render-status"].update("Rendering...")
                self.window.perform_long_operation(self.render, "RENDER COMPLETE")
                self.is_rendering = True

            if self.draw_command:
                if self.values["wl-manual"]:
                    wl_idx = np.searchsorted(SPECTRUM_WAVELENGTHS, self.values["wavelength"], side="right")
                    draw_f = lambda: self.draw_figure(self.result[3 + wl_idx])
                else:
                    draw_f = lambda: self.draw_figure(self.result[:3])
                self.window.perform_long_operation(draw_f, "DRAW COMPLETE")

        self.window.close()

    def draw_figure(self, rgb):

        exposure = self.values["exposure"] if "exposure" in self.values else self.default_exposure
        if rgb.ndim > 2:
            rgb_ = np.transpose(rgb, axes=(1, 2, 0))
        else:
            rgb_ = rgb
        text = image2texture(rgb_, float(exposure))

        self.img_tmp = Image.fromarray(np.transpose(text, axes=(1, 0, 2)), mode="RGBA")
        img_resize = self.img_tmp.resize((600, 600))
        buf = io.BytesIO()
        img_resize.save(buf, format="PNG")
        self.window["canvas"].update(buf.getvalue())

    def load_file(self):
        self.is_loading = True
        start = time.time()
        try:
            self.window["load_ok"].update(f"Loading ...")
            self.sky.initialise(self.values["load_path"])
            self.loading_success = True
            self.loading_error = None
        except Exception as e:
            self.window["load_ok"].update(f"")
            self.loading_success = False
            self.loading_error = e

        end = time.time()
        self.is_loading = False

        self.loading_time = end - start

    def save_image(self):
        if self.img_tmp is not None and self.values["save_path"] is not None:
            try:
                self.window["save_ok"].update("Saving ...")
                extension = self.save_path.split(".")[-1].lower()
                if extension in ("png", "jpg", "jpeg"):
                    self.img_tmp.save(self.save_path)
                    self.saving_success = True
                elif extension in ("csv", "xlsx", "xls"):
                    xs, ys = np.meshgrid(np.arange(self.result.shape[1]), np.arange(self.result.shape[2]))
                    resolution = np.maximum(self.result.shape[1], self.result.shape[2])

                    views_dir = pixel2dir(xs, ys, resolution)
                    pixel_map = ~np.all(np.isclose(views_dir, 0), axis=2)
                    raw_data = np.hstack([views_dir[pixel_map, :], self.result[:, pixel_map].T])
                    columns = ["x", "y", "z", "R", "G", "B"] + [f"wl-{wl}" for wl in SPECTRUM_WAVELENGTHS]
                    df = pd.DataFrame(raw_data, columns=columns)
                    if extension in ("csv",):
                        df.to_csv(self.save_path, sep=",")
                    else:
                        df.to_excel(self.save_path)
                    self.saving_success = True
                else:
                    self.window["save_ok"].update("")
                    self.saving_success = False
                    self.saving_error = [f"Unsupported file extension: '*.{extension}'."]
            except Exception as e:
                self.window["save_ok"].update("")
                self.saving_success = False
                self.saving_error = e
        else:
            self.saving_success = False

    def render(self):

        self.is_rendering = True
        start = time.time()
        try:
            self.result = render(sky_model=self.sky,
                                 albedo=float(self.values["albedo"]),
                                 altitude=float(self.values["altitude"]),
                                 azimuth=np.deg2rad(self.values["azimuth"]),
                                 elevation=np.deg2rad(self.values["elevation"]),
                                 visibility=float(self.values["visibility"]),
                                 resolution=int(self.values["resolution"]),
                                 mode=self.values["mode"])
            self.rendering_success = True
            self.rendering_error = None
        except Exception as e:
            self.rendering_success = False
            self.rendering_error = e
        end = time.time()
        self.is_rendering = False

        self.rendering_time = end - start

    def load_complete(self):
        if self.loading_success:
            self.window["load_ok"].update(f"Done. ({self.loading_time:.1f} sec)")
            self.update_available_data(render_disable=False)

            if self.values["auto-update"]:
                self.render_command = True
        else:
            self.window["load_ok"].update(f"Failed. ({self.loading_time:.1f} sec)")

            if self.loading_error is not None:
                sg.popup_error_with_traceback("Loading error", self.loading_error)

    def save_complete(self):
        if self.saving_success:
            self.window["save_ok"].update(f"Done. ({self.saving_time:.1f} sec)")
        else:
            self.window["save_ok"].update(f"Failed. ({self.saving_time:.1f} sec)")

            if self.saving_error is not None:
                sg.popup_error_with_traceback("Saving error", self.saving_error)

    def render_complete(self):
        if self.rendering_success:
            self.window["render-status"].update(f"Done. ({self.rendering_time:.1f} sec)")
            self.update_available_data(render_disable=False, display_disable=False)
            self.draw_command = True
        else:
            self.window["render-status"].update(f"Failed. ({self.rendering_time:.1f} sec)")

            if self.rendering_error is not None:
                sg.popup_error_with_traceback("Rendering error", self.rendering_error)

    def draw_complete(self):
        pass

    def update_available_data(self, render_disable=True, display_disable=True):

        self.available = self.sky.available_data

        # Update ranges and values, disable or enable changes
        self.window["albedo"].update(range=(self.available.albedo_min, self.available.albedo_max),
                                     disabled=render_disable)
        self.window["altitude"].update(range=(self.available.altitude_min, self.available.altitude_max),
                                       disabled=render_disable)
        self.window["azimuth"].update(disabled=render_disable)
        self.window["elevation"].update(range=(self.available.elevation_min, self.available.elevation_max),
                                        disabled=render_disable)
        self.window["resolution"].update(disabled=render_disable)
        self.window["visibility"].update(range=(self.available.visibility_min, self.available.visibility_max),
                                         disabled=render_disable)

        if not self.available.polarisation and "Polarisation" in self.modes:
            self.modes.remove("Polarisation")
        elif self.available.polarisation and "Polarisation" not in self.modes:
            self.modes.insert(2, "Polarisation")

        self.window["mode"].update(values=self.modes, value=self.modes[self.default_mode], disabled=render_disable)
        self.window["render"].update(disabled=render_disable or self.default_auto_update)
        self.window["auto-update"].update(disabled=render_disable)
        self.window["wl-visible"].update(disabled=render_disable or display_disable)
        self.window["wl-manual"].update(disabled=render_disable or display_disable)
        self.window["wavelength"].update(range=(
            self.available.channel_start,
            self.available.channel_start + self.available.channels * self.available.channel_width - 1),
            disabled=render_disable or display_disable or self.values["wl-visible"])
        self.window["exposure"].update(disabled=render_disable or display_disable)
        self.window["save_path"].update(disabled=render_disable or display_disable)
        self.window["save"].update(disabled=render_disable or display_disable)

        # Update tooltips
        self.window["albedo_tooltip"].TooltipObject.text = (
            f"Ground albedo, value in range [{self.available.albedo_min:.1f}, {self.available.albedo_max:.1f}].")
        self.window["altitude_tooltip"].TooltipObject.text = (
            f"Altitude of view point in meters, "
            f"value in range [{self.available.altitude_min:.1f}, {self.available.altitude_max:.1f}].")
        self.window["azimuth_tooltip"].TooltipObject.text = (
            f"Sun azimuth at view point in degrees, value in range [0, 360].")
        self.window["elevation_tooltip"].TooltipObject.text = (
            f"Sun elevation at view point in degrees, "
            f"value in range [{self.available.elevation_min:.1f}, {self.available.elevation_max:.1f}].")
        self.window["visibility_tooltip"].TooltipObject.text = (
            f"Horizontal visibility (meteorological range) at ground level in kilometers, "
            f"value in range [{self.available.visibility_min:.1f}, {self.available.visibility_max:.1f}].")
        self.window["resolution_tooltip"].TooltipObject.text = (
            f"Length of resulting square image size in pixels, value in range [1, 10000].")
        self.window["albedo_tooltip"].TooltipObject.text = (
            f"Ground albedo, value in range [{self.available.albedo_min:.1f}, {self.available.albedo_max:.1f}].")
        self.window["wavelength_tooltip"].TooltipObject.text = (
            f"Wavelength determining the displayed wavelength bin.")


def get_file_name(path):
    return path.split(os.sep)[-1]


def get_folder_path(path):
    return os.sep.join(path.split(os.sep)[:-1])
