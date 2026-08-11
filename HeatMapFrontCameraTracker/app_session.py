"""Main tracking session: cameras, heatmap loop, ArUco overlays."""

import os
import sys
import time

import cv2

import aruco_screen
import eye_tracker
import gaze_screen
from camera_io import CAMERA_OPEN_DELAY_SEC, CameraReader, stop_all_readers
from camera_panel import CameraPanelOverlay
from input_poll import KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_UP, poll_control_key

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def _format_fusion_status(fusion_eyes, active_eyes):
    if fusion_eyes == active_eyes or set(fusion_eyes) == set(active_eyes):
        return "Gaze: both eyes (1=left 2=right)"
    if fusion_eyes == ("left",):
        return "Gaze: LEFT only (0=both 2=right)"
    if fusion_eyes == ("right",):
        return "Gaze: RIGHT only (0=both 1=left)"
    return f"Gaze: {','.join(fusion_eyes)}"


def _on_heatmap_mouse(event, x, y, flags, param):
    """Forward clicks on embedded IR previews to the eye tracker sphere lock."""
    panel = param["panel"]
    hit = panel.hit_test(x, y)
    if hit is None:
        return
    if hit["slot"] not in eye_tracker.EYE_IDS:
        return
    eye_tracker.on_mouse_frame_with_rays(
        event,
        hit["frame_x"],
        hit["frame_y"],
        flags,
        hit["slot"],
    )


def run(
    left_index,
    right_index,
    front_index,
    flip_left=True,
    flip_right=True,
    mirror_left=False,
    mirror_right=False,
    flip_front=True,
    mirror_front=True,
):
    os.chdir(ROOT_DIR)
    eye_tracker.set_show_separate_tracking_windows(False)

    if left_index is None and right_index is None:
        print("Select at least one eye camera.")
        return

    if left_index is None:
        left_index, right_index = right_index, None
    if left_index == right_index:
        right_index = None

    active_eyes = tuple(
        eye_id
        for eye_id, index in (("left", left_index), ("right", right_index))
        if index is not None
    )
    fusion_eyes = active_eyes

    screen_w, screen_h, win_x, win_y = gaze_screen.get_window_placement()
    heatmap_window = "Gaze Heatmap"
    heatmap_session = gaze_screen.GazeHeatmapSession(screen_w, screen_h)
    gaze_screen.create_main_window(heatmap_window, screen_w, screen_h, win_x, win_y)

    corner_markers = aruco_screen.ScreenCornerMarkers(screen_w, screen_h)
    show_corner_overlay = front_index is not None
    camera_panel = CameraPanelOverlay(screen_w, screen_h)

    active_slots = list(active_eyes)
    if front_index is not None:
        active_slots.append("front")
    camera_panel.set_active_slots(active_slots)

    mouse_context = {"panel": camera_panel}
    cv2.setMouseCallback(heatmap_window, _on_heatmap_mouse, mouse_context)

    heatmap_fps = 0.0
    heatmap_last_frame_time = time.perf_counter()

    print(f"Left eye IR:  {left_index}")
    print(f"Right eye IR: {right_index if right_index is not None else 'disabled'}")
    print(f"Front camera: {front_index if front_index is not None else 'disabled'}")
    print("Fusion: average of available eye gaze directions")
    print("Calibrazione heatmap: C=centro, poi frecce su/giu/sin/des ai bordi (5 punti)")
    print("Tasti: V anteprime | 0/1/2 fusione | M marker | C + frecce calib | Q esci")
    print("Una sola finestra (sopra la taskbar). V nasconde le camere al centro.")
    if front_index is not None:
        print("ArUco: marker agli angoli attivi sulla heatmap (M toggle)")
        print("Con ArUco: usa C + frecce per calibrazione IR (ignora mapping ArUco durante la calib)")
    if sys.platform == "win32":
        print("Su Windows: se i tasti non rispondono, clicca sulla finestra 'Gaze Heatmap'.")

    readers = {}
    for eye_id, index in (("left", left_index), ("right", right_index)):
        if index is None:
            continue
        reader = CameraReader(index, width=640, height=480)
        reader.start()
        if not reader.is_opened():
            print(f"Warning: {eye_id} eye camera {index} not ready yet (will retry in background).")
        else:
            print(f"{eye_id.title()} eye camera {index} ready ({reader.backend_name}).")
        readers[eye_id] = reader
        eye_tracker.reset_eye_tracking_state(eye_id)
        time.sleep(CAMERA_OPEN_DELAY_SEC)

    front_reader = None
    if front_index is not None and front_index not in {left_index, right_index}:
        front_reader = CameraReader(front_index, width=640, height=480)
        front_reader.start()
        if not front_reader.is_opened():
            print(f"Warning: front camera {front_index} not ready yet (will retry in background).")
        else:
            print(f"Front camera {front_index} ready ({front_reader.backend_name}).")
        time.sleep(CAMERA_OPEN_DELAY_SEC)

    eye_tracker.calibrated = False
    eye_tracker.reset_gaze_smoothing()

    aruco_tracker = aruco_screen.ArucoScreenTracker(screen_w, screen_h) if front_reader else None

    try:
        while True:
            for eye_id, reader in readers.items():
                ret, frame = reader.read()
                if not ret:
                    continue

                flip_v = flip_left if eye_id == "left" else flip_right
                mirror = mirror_left if eye_id == "left" else mirror_right
                eye_tracker.process_frame(
                    frame,
                    eye_id=eye_id,
                    flip_vertical=flip_v,
                    flip_horizontal=mirror,
                )
                camera_panel.set_frame(eye_id, eye_tracker.get_preview_frame(eye_id))

            combined_gaze = eye_tracker.refresh_combined_gaze(fusion_eyes)
            calibration_gaze = eye_tracker.get_raw_combined_gaze_dir()

            use_calibrated_mapping = heatmap_session.mapping_ready
            ir_calibrating = heatmap_session.calibrating
            use_aruco_mapping = (
                not use_calibrated_mapping
                and not ir_calibrating
                and not heatmap_session.center_calibrated
                and aruco_tracker is not None
                and aruco_tracker.ready
                and eye_tracker.calibrated
            )
            aruco_available = aruco_tracker is not None and aruco_tracker.ready

            if front_reader is not None:
                ret_ext, ext_frame = front_reader.read()
                if ret_ext:
                    fh, fw = ext_frame.shape[:2]
                    if fw != eye_tracker.EXT_WIDTH or fh != eye_tracker.EXT_HEIGHT:
                        eye_tracker.configure_external_viewport(fw, fh)

                    display = ext_frame
                    if (fw, fh) != (eye_tracker.EXT_WIDTH, eye_tracker.EXT_HEIGHT):
                        display = cv2.resize(ext_frame, (eye_tracker.EXT_WIDTH, eye_tracker.EXT_HEIGHT))

                    if aruco_tracker is not None:
                        display = aruco_tracker.process(display)

                    if use_aruco_mapping and combined_gaze is not None:
                        uv = aruco_screen.gaze_to_front_camera_pixels(
                            combined_gaze,
                            eye_tracker.R_gaze_to_cam,
                            eye_tracker.EXT_WIDTH,
                            eye_tracker.EXT_HEIGHT,
                            eye_tracker.EXT_CX,
                            eye_tracker.EXT_CY,
                            eye_tracker.EXT_FX,
                            eye_tracker.EXT_FY,
                        )
                        if uv is not None:
                            cv2.circle(display, (int(uv[0]), int(uv[1])), 7, (0, 0, 255), -1)

                    if flip_front:
                        display = cv2.flip(display, 0)
                    if mirror_front:
                        display = cv2.flip(display, 1)
                    camera_panel.set_frame("front", display)

            if use_aruco_mapping:
                heatmap_session.update_via_aruco(
                    combined_gaze,
                    aruco_tracker.homography,
                    eye_tracker.R_gaze_to_cam,
                    eye_tracker.EXT_WIDTH,
                    eye_tracker.EXT_HEIGHT,
                    eye_tracker.EXT_CX,
                    eye_tracker.EXT_CY,
                    eye_tracker.EXT_FX,
                    eye_tracker.EXT_FY,
                )
            else:
                heatmap_session.update(combined_gaze)
            now = time.perf_counter()
            heatmap_fps = 0.9 * heatmap_fps + 0.1 / max(now - heatmap_last_frame_time, 1e-6)
            heatmap_last_frame_time = now

            camera_status = {eye_id: reader.snapshot_status() for eye_id, reader in readers.items()}
            if front_reader is not None:
                camera_status["front"] = front_reader.snapshot_status()

            hud_extra_lines = []
            if use_calibrated_mapping:
                hud_extra_lines.append(
                    f"Mapping: piecewise (5 punti) | "
                    f"{_format_fusion_status(fusion_eyes, active_eyes)}"
                )
            elif ir_calibrating:
                hud_extra_lines.append(
                    f"Mapping: calibrazione IR ({len(heatmap_session.calibrated_edges)}/4 bordi) | "
                    f"{_format_fusion_status(fusion_eyes, active_eyes)}"
                )
            elif use_aruco_mapping:
                hud_extra_lines.append(f"Mapping: ArUco | {_format_fusion_status(fusion_eyes, active_eyes)}")
            elif heatmap_session.ready:
                hud_extra_lines.append(f"Mapping: partial C/B/R | {_format_fusion_status(fusion_eyes, active_eyes)}")
            elif aruco_available:
                hud_extra_lines.append(
                    f"Mapping: premi C per calibrazione IR (ArUco disponibile) | "
                    f"{_format_fusion_status(fusion_eyes, active_eyes)}"
                )
            else:
                hud_extra_lines.append(_format_fusion_status(fusion_eyes, active_eyes))
            hud_extra_lines.append(camera_panel.status_line())
            if show_corner_overlay:
                hud_extra_lines.append(corner_markers.status_line())
            if aruco_tracker is not None:
                hud_extra_lines.append(aruco_tracker.status_line())

            heatmap_display = heatmap_session.compose_display(
                heatmap_fps,
                aruco_mapping=aruco_available and not heatmap_session.center_calibrated,
            )
            if show_corner_overlay:
                heatmap_display = corner_markers.overlay_on(heatmap_display)
            heatmap_display = camera_panel.overlay_on(heatmap_display)

            hud_safe_zone = None
            if show_corner_overlay and corner_markers.visible:
                hud_safe_zone = corner_markers.hud_safe_zone()

            heatmap_session.draw_hud(
                heatmap_display,
                heatmap_fps,
                camera_status=camera_status,
                extra_lines=hud_extra_lines,
                safe_zone=hud_safe_zone,
            )

            cv2.imshow(heatmap_window, heatmap_display)
            gaze_screen.focus_window(heatmap_window)

            key = poll_control_key(heatmap_window)
            if key == ord("q"):
                break
            if key == ord(" "):
                cv2.waitKey(0)
            elif key == ord("v"):
                visible = camera_panel.toggle()
                print(f"Camera previews: {'ON' if visible else 'OFF'}")
            elif key == ord("m") and show_corner_overlay:
                visible = corner_markers.toggle()
                print(f"ArUco corner markers on screen: {'ON' if visible else 'OFF'}")
            elif key == ord("c"):
                if calibration_gaze is None:
                    print("No combined gaze vector yet.")
                else:
                    eye_tracker.calibrated = False
                    _, message = heatmap_session.set_center_calibration(calibration_gaze)
                    eye_tracker.reset_gaze_smoothing()
                    print(message)
                    print("Guarda il MONITOR fisico. I punti verdi sono gli ancoraggi salvati.")
            elif key == ord("h"):
                eye_tracker.calibrated = False
                _, message = heatmap_session.handle_key(key, combined_gaze)
                print(message)
            elif key == KEY_UP:
                _, message = heatmap_session.calibrate_top(calibration_gaze)
                print(message)
            elif key == KEY_DOWN:
                _, message = heatmap_session.calibrate_bottom(calibration_gaze)
                print(message)
            elif key == KEY_LEFT:
                _, message = heatmap_session.calibrate_left(calibration_gaze)
                print(message)
            elif key == KEY_RIGHT:
                _, message = heatmap_session.calibrate_right(calibration_gaze)
                print(message)
            elif key == ord("s"):
                out_path = os.path.join(ROOT_DIR, "gaze_heatmap.png")
                heatmap_session.save_png(out_path)
                print(f"Saved {out_path}")
            elif key == ord("0"):
                fusion_eyes = active_eyes
                eye_tracker.reset_gaze_smoothing()
                heatmap_session.reset_calibration()
                heatmap_session.reset_heatmap()
                print("Gaze fusion: both eyes. Calibrazione resettata: premi C.")
            elif key == ord("1") and "left" in active_eyes:
                fusion_eyes = ("left",)
                eye_tracker.reset_gaze_smoothing()
                heatmap_session.reset_calibration()
                heatmap_session.reset_heatmap()
                print("Gaze fusion: LEFT eye only. Calibrazione resettata: premi C.")
            elif key == ord("2") and "right" in active_eyes:
                fusion_eyes = ("right",)
                eye_tracker.reset_gaze_smoothing()
                heatmap_session.reset_calibration()
                heatmap_session.reset_heatmap()
                print("Gaze fusion: RIGHT eye only. Calibrazione resettata: premi C.")
            else:
                handled, message = heatmap_session.handle_key(key, combined_gaze)
                if handled and message:
                    print(message)
    finally:
        stop_all_readers(readers)
        if front_reader is not None:
            front_reader.stop()
        corner_markers.close_corner_window()
        cv2.destroyAllWindows()
