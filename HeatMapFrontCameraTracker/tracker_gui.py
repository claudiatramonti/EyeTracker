"""Tkinter camera selection GUI."""

import tkinter as tk
from tkinter import ttk

import eye_tracker


def camera_label(index):
    return str(index)


def other_camera_options(cameras, *reserved):
    reserved_set = {index for index in reserved if index is not None}
    options = ["None"]
    for cam in cameras:
        if cam not in reserved_set:
            options.append(camera_label(cam))
    return options


def default_camera(cameras, *reserved):
    reserved_set = {index for index in reserved if index is not None}
    for cam in cameras:
        if cam not in reserved_set:
            return camera_label(cam)
    return "None"


def parse_camera_choice(value):
    if value is None or value == "None":
        return None
    return int(value)


def selection_gui():
    cameras = eye_tracker.detect_cameras()

    root = tk.Tk()
    root.title("HeatMap Stereo Tracker")
    tk.Label(
        root,
        text="2× IR eye cameras + optional front camera + heatmap",
        font=("Arial", 12, "bold"),
    ).pack(pady=10)

    left_var = tk.StringVar(value=camera_label(cameras[0]) if cameras else "0")
    right_var = tk.StringVar(
        value=camera_label(cameras[1]) if len(cameras) > 1 else "None"
    )
    front_var = tk.StringVar(value="None")

    controls = ttk.Frame(root)
    controls.pack(pady=5)

    def add_row(row, label, variable):
        tk.Label(controls, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=4)
        combo = ttk.Combobox(
            controls,
            textvariable=variable,
            state="readonly" if cameras else "disabled",
            width=12,
        )
        combo.grid(row=row, column=1, padx=6, pady=4)
        return combo

    left_dropdown = add_row(0, "Left eye IR:", left_var)
    right_dropdown = add_row(1, "Right eye IR:", right_var)
    front_dropdown = add_row(2, "Front camera:", front_var)

    flip_left_var = tk.BooleanVar(value=True)
    flip_right_var = tk.BooleanVar(value=False)
    mirror_left_var = tk.BooleanVar(value=False)
    mirror_right_var = tk.BooleanVar(value=False)
    flip_front_var = tk.BooleanVar(value=False)
    mirror_front_var = tk.BooleanVar(value=False)
    flip_frame = ttk.Frame(root)
    flip_frame.pack(pady=2)
    ttk.Checkbutton(flip_frame, text="Flip left (V)", variable=flip_left_var).pack(side="left", padx=6)
    ttk.Checkbutton(flip_frame, text="Mirror left (L/R)", variable=mirror_left_var).pack(side="left", padx=6)
    ttk.Checkbutton(flip_frame, text="Flip right (V)", variable=flip_right_var).pack(side="left", padx=6)
    ttk.Checkbutton(flip_frame, text="Mirror right (L/R)", variable=mirror_right_var).pack(side="left", padx=6)
    ttk.Checkbutton(flip_frame, text="Flip front (V)", variable=flip_front_var).pack(side="left", padx=6)
    ttk.Checkbutton(flip_frame, text="Mirror front (L/R)", variable=mirror_front_var).pack(side="left", padx=6)

    def current_indices():
        return (
            parse_camera_choice(left_var.get()),
            parse_camera_choice(right_var.get()),
            parse_camera_choice(front_var.get()),
        )

    def refresh(*_):
        if not cameras:
            for dropdown in (left_dropdown, right_dropdown, front_dropdown):
                dropdown.configure(values=["None"])
            left_var.set("0")
            right_var.set("None")
            front_var.set("None")
            return

        left_index, right_index, front_index = current_indices()

        left_dropdown.configure(values=[camera_label(cam) for cam in cameras])
        right_dropdown.configure(values=other_camera_options(cameras, left_index))
        front_dropdown.configure(values=other_camera_options(cameras, left_index, right_index))

        if left_var.get() not in left_dropdown["values"]:
            left_var.set(camera_label(cameras[0]))
        if right_var.get() not in right_dropdown["values"]:
            right_var.set(default_camera(cameras, left_index))
        if front_var.get() not in front_dropdown["values"]:
            front_var.set(default_camera(cameras, left_index, right_index))

    for variable in (left_var, right_var):
        variable.trace_add("write", refresh)
    refresh()

    tk.Label(
        root,
        text="Single window: V toggles camera previews (center) | click IR preview to lock sphere",
        font=("Arial", 9),
    ).pack(pady=4)

    choice = {
        "left": None,
        "right": None,
        "front": None,
        "flip_left": True,
        "flip_right": True,
        "mirror_left": False,
        "mirror_right": False,
        "flip_front": True,
        "mirror_front": True,
    }

    def start():
        choice["left"] = parse_camera_choice(left_var.get())
        choice["right"] = parse_camera_choice(right_var.get())
        choice["front"] = parse_camera_choice(front_var.get())
        choice["flip_left"] = flip_left_var.get()
        choice["flip_right"] = flip_right_var.get()
        choice["mirror_left"] = mirror_left_var.get()
        choice["mirror_right"] = mirror_right_var.get()
        choice["flip_front"] = flip_front_var.get()
        choice["mirror_front"] = mirror_front_var.get()
        root.destroy()

    tk.Button(root, text="Start", command=start).pack(pady=8)
    root.mainloop()
    return choice
