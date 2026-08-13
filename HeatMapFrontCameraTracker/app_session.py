"""Main loop: IR cameras, fusion, heatmap, optional front/ArUco.

Each frame: track each eye, fuse directions, then map to the monitor.
Primary heatmap path is the 5-point IR map in gaze_screen (C + arrows).
Front camera + ArUco are optional (FOV/homography or experimental 6DoF).
"""

import os
import sys
import time
from collections import deque

import cv2
import numpy as np

import aruco_screen
import eye_tracker
import gaze_screen
from camera_io import CAMERA_OPEN_DELAY_SEC, CameraReader, stop_all_readers
from camera_panel import CameraPanelOverlay
from input_poll import KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_UP, poll_control_key

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# IR gaze → front FOV projection → ArUco homography → monitor.
# Primary when front camera + C (R_gaze_to_cam) + markers are ready.
ENABLE_FOV_SCREEN_MAPPING = True
# Legacy experimental 6DoF delta path (kept off; FOV+homography is preferred).
ENABLE_HEAD_POSE_MAPPING = False
FRONT_FOV_DEG = 60.0


def _format_fusion_status(fusion_eyes, active_eyes):
    if fusion_eyes == active_eyes or set(fusion_eyes) == set(active_eyes):
        return "Gaze: both eyes (1=left 2=right)"
    if fusion_eyes == ("left",):
        return "Gaze: LEFT only (0=both 2=right)"
    if fusion_eyes == ("right",):
        return "Gaze: RIGHT only (0=both 1=left)"
    return f"Gaze: {','.join(fusion_eyes)}"


def _current_eye_gazes(eye_ids):
    gazes = {}
    for eye_id in eye_ids:
        direction = eye_tracker.get_eye_gaze_dir(eye_id)
        if direction is not None:
            gazes[eye_id] = direction
    return gazes


class PoseGazeHistory:
    """Rolling per-eye gaze + pose samples from the live loop (no UI freeze)."""

    def __init__(self, maxlen=None):
        self.maxlen = (
            aruco_screen.CALIBRATION_HISTORY_FRAMES if maxlen is None else maxlen
        )
        self._by_eye = {}

    def clear(self):
        self._by_eye.clear()

    def push(self, eye_ids, aruco_tracker):
        if aruco_tracker is None or not aruco_tracker.pose_ready:
            return
        rotation = aruco_tracker.pose_rotation
        translation = aruco_tracker.pose_translation
        if rotation is None or translation is None:
            return
        for eye_id in eye_ids:
            direction = eye_tracker.get_eye_gaze_dir(eye_id)
            if direction is None:
                continue
            bucket = self._by_eye.setdefault(
                eye_id,
                deque(maxlen=self.maxlen),
            )
            bucket.append(
                (
                    np.asarray(direction, dtype=np.float64).copy(),
                    rotation.copy(),
                    translation.copy(),
                )
            )

    def snapshot(self, eye_ids, screen_point, min_samples=None):
        """Average recent gaze, camera-frame target, and ArUco pose."""
        min_samples = (
            aruco_screen.CALIBRATION_HISTORY_MIN
            if min_samples is None
            else min_samples
        )
        point_screen = np.array(
            [float(screen_point[0]), float(screen_point[1]), 0.0],
            dtype=np.float64,
        )
        gaze_by_eye = {}
        target_sums = []
        rotations = []
        translations = []
        for eye_id in eye_ids:
            bucket = self._by_eye.get(eye_id)
            if not bucket or len(bucket) < min_samples:
                continue
            directions = [entry[0] for entry in bucket]
            averaged = aruco_screen.average_unit_directions(directions)
            if averaged is None:
                continue
            gaze_by_eye[eye_id] = averaged
            for _, rotation, translation in bucket:
                target_sums.append(rotation @ point_screen + translation)
                rotations.append(rotation)
                translations.append(translation)

        if not gaze_by_eye or not target_sums:
            return {}, None, None, None
        target = np.mean(np.asarray(target_sums, dtype=np.float64), axis=0)
        translation = np.mean(np.asarray(translations, dtype=np.float64), axis=0)
        blended = np.mean(np.asarray(rotations, dtype=np.float64), axis=0)
        try:
            u, _, vt = np.linalg.svd(blended)
            rotation = u @ vt
            if np.linalg.det(rotation) < 0.0:
                u[:, -1] *= -1.0
                rotation = u @ vt
        except np.linalg.LinAlgError:
            rotation = rotations[-1]
        return gaze_by_eye, target, rotation, translation


def _calibration_target(heatmap_session, label):
    if label == "center":
        return (heatmap_session.cx, heatmap_session.cy)
    return gaze_screen.calibration_targets(
        heatmap_session.width,
        heatmap_session.height,
    )[label]


def _record_pose_sample_from_history(pose_mapper, history, heatmap_session, label, eye_ids):
    if pose_mapper is None:
        return "Posa 6DoF non attiva."
    target_uv = _calibration_target(heatmap_session, label)
    gaze_by_eye, target_camera, pose_rotation, pose_translation = history.snapshot(
        eye_ids,
        target_uv,
    )
    if not gaze_by_eye or target_camera is None:
        return (
            "Posa 6DoF non salvata: servi almeno "
            f"{aruco_screen.CALIBRATION_HISTORY_MIN} frame recenti con pose ArUco "
            "(tieni ≥3 marker visibili un momento, poi ripremi)."
        )
    _, message = pose_mapper.add_calibration_sample_with_target(
        label,
        gaze_by_eye,
        target_camera,
        pose_rotation=pose_rotation,
        pose_translation=pose_translation,
    )
    return message


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
    """Open cameras and run the heatmap loop (fusion + C/arrows mapping)."""
    os.chdir(ROOT_DIR)
    eye_tracker.set_show_separate_tracking_windows(False)
    eye_tracker.ensure_gl_sphere_window()

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
    eye_tracker.configure_external_viewport(
        eye_tracker.EXT_WIDTH,
        eye_tracker.EXT_HEIGHT,
        fov_deg=FRONT_FOV_DEG,
    )
    print(
        f"Front gaze projection: FOV {FRONT_FOV_DEG:.0f}° "
        f"(fx={eye_tracker.EXT_FX:.1f}, fy={eye_tracker.EXT_FY:.1f})."
    )

    aruco_tracker = (
        aruco_screen.ArucoScreenTracker(
            screen_w,
            screen_h,
            marker_size=corner_markers.marker_size,
            marker_margin=corner_markers.margin,
            cam_width=eye_tracker.EXT_WIDTH,
            cam_height=eye_tracker.EXT_HEIGHT,
            cam_fx=eye_tracker.EXT_FX,
            cam_fy=eye_tracker.EXT_FY,
            cam_cx=eye_tracker.EXT_CX,
            cam_cy=eye_tracker.EXT_CY,
        )
        if front_reader
        else None
    )
    pose_mapper = None
    head_pose_enabled = False
    fov_screen_enabled = (
        ENABLE_FOV_SCREEN_MAPPING and aruco_tracker is not None
    )
    if aruco_tracker is not None:
        if aruco_tracker.camera_calibrated:
            print(
                "Front camera chessboard calibration loaded "
                f"(RMS {aruco_tracker.camera_calibration_rms:.3f}px) "
                "for ArUco pose; gaze→front still uses FOV "
                f"{FRONT_FOV_DEG:.0f}°."
            )
        if ENABLE_HEAD_POSE_MAPPING and aruco_tracker.camera_calibrated:
            pose_mapper = aruco_screen.GazePoseMapper(eye_ids=active_eyes)
            head_pose_enabled = True
            print("Head compensation 6DoF: ON (legacy delta path).")
        if fov_screen_enabled:
            print(
                "Screen mapping: FOV + ArUco. Premi C guardando il CENTRO "
                "monitor (link IR↔front), tieni 4 marker visibili."
            )
    if not fov_screen_enabled and not head_pose_enabled:
        print("Using static 5-point mapping. Keep head still after calibration.")

    pose_history = PoseGazeHistory()

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

            if front_reader is not None:
                ret_ext, ext_frame = front_reader.read()
                if ret_ext:
                    fh, fw = ext_frame.shape[:2]
                    if fw != eye_tracker.EXT_WIDTH or fh != eye_tracker.EXT_HEIGHT:
                        eye_tracker.configure_external_viewport(
                            fw,
                            fh,
                            fov_deg=FRONT_FOV_DEG,
                        )
                    if aruco_tracker is not None:
                        # ArUco pose keeps chessboard K if loaded; gaze uses FOV.
                        aruco_tracker.configure_camera(
                            fw,
                            fh,
                            fx=eye_tracker.EXT_FX,
                            fy=eye_tracker.EXT_FY,
                            cx=eye_tracker.EXT_CX,
                            cy=eye_tracker.EXT_CY,
                        )

                    display = ext_frame
                    if (fw, fh) != (eye_tracker.EXT_WIDTH, eye_tracker.EXT_HEIGHT):
                        display = cv2.resize(ext_frame, (eye_tracker.EXT_WIDTH, eye_tracker.EXT_HEIGHT))

                    if aruco_tracker is not None:
                        display = aruco_tracker.process(display)

                    if eye_tracker.calibrated:
                        eye_tracker.update_gaze_circle_from_current_gaze()
                        cv2.circle(
                            display,
                            (int(eye_tracker.circle_x), int(eye_tracker.circle_y)),
                            8,
                            (0, 0, 255),
                            -1,
                        )

                    if flip_front:
                        display = cv2.flip(display, 0)
                    if mirror_front:
                        display = cv2.flip(display, 1)
                    camera_panel.set_frame("front", display)
                elif aruco_tracker is not None:
                    aruco_tracker.ready = False
                    aruco_tracker.pose_ready = False
                    aruco_tracker.homography = None

            pose_history.push(fusion_eyes, aruco_tracker)
            eye_tracker.pump_gl_sphere_events()

            ir_calibrating = heatmap_session.calibrating
            pose_available = aruco_tracker is not None and aruco_tracker.pose_ready
            pose_calibrated = pose_mapper is not None and pose_mapper.anchored_ready
            aruco_available = aruco_tracker is not None and aruco_tracker.ready
            homography_ready = (
                aruco_tracker is not None and aruco_tracker.homography is not None
            )
            use_fov_mapping = (
                fov_screen_enabled
                and eye_tracker.calibrated
                and homography_ready
                and combined_gaze is not None
            )
            use_pose_mapping = (
                head_pose_enabled
                and heatmap_session.mapping_ready
                and pose_calibrated
                and pose_available
                and not use_fov_mapping
            )
            use_calibrated_mapping = (
                heatmap_session.mapping_ready
                and not use_fov_mapping
                and not use_pose_mapping
            )

            fov_status = None
            if use_fov_mapping:
                ok, fov_status = heatmap_session.update_via_aruco(
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
                if not ok:
                    if fov_status == "off_screen":
                        heatmap_session.last_point = None
                    elif heatmap_session.mapping_ready:
                        heatmap_session.update(combined_gaze)
            elif use_pose_mapping:
                static_point = heatmap_session.project_to_screen(combined_gaze)
                gaze_by_eye = _current_eye_gazes(fusion_eyes)
                pose_point = pose_mapper.project_anchored(
                    gaze_by_eye,
                    aruco_tracker,
                    static_point,
                )
                if pose_point is not None:
                    heatmap_session.update_screen_point(pose_point)
                elif static_point is not None:
                    heatmap_session.update_screen_point(static_point)
                else:
                    heatmap_session.update(combined_gaze)
            else:
                heatmap_session.update(combined_gaze)
            now = time.perf_counter()
            heatmap_fps = 0.9 * heatmap_fps + 0.1 / max(now - heatmap_last_frame_time, 1e-6)
            heatmap_last_frame_time = now

            camera_status = {eye_id: reader.snapshot_status() for eye_id, reader in readers.items()}
            if front_reader is not None:
                camera_status["front"] = front_reader.snapshot_status()

            hud_extra_lines = []
            if use_fov_mapping:
                if fov_status == "off_screen":
                    hud_extra_lines.append(
                        f"Mapping: FOV {FRONT_FOV_DEG:.0f}°+ArUco | FUORI SCHERMO | "
                        f"{_format_fusion_status(fusion_eyes, active_eyes)}"
                    )
                else:
                    hud_extra_lines.append(
                        f"Mapping: FOV {FRONT_FOV_DEG:.0f}°+ArUco (front→monitor) | "
                        f"{_format_fusion_status(fusion_eyes, active_eyes)}"
                    )
            elif fov_screen_enabled and not eye_tracker.calibrated:
                hud_extra_lines.append(
                    "Mapping: premi C al CENTRO (link IR↔front, FOV 60°) | "
                    f"{_format_fusion_status(fusion_eyes, active_eyes)}"
                )
            elif fov_screen_enabled and eye_tracker.calibrated and not homography_ready:
                hud_extra_lines.append(
                    "Mapping: FOV OK, serve omografia ArUco (4 marker) | "
                    f"{_format_fusion_status(fusion_eyes, active_eyes)}"
                )
            elif use_pose_mapping:
                hud_extra_lines.append(
                    f"Mapping: statico + delta testa 6DoF | "
                    f"{_format_fusion_status(fusion_eyes, active_eyes)}"
                )
            elif pose_calibrated and not pose_available:
                hud_extra_lines.append(
                    "Mapping: 6DoF fallback statico (marker non visibili)"
                )
            elif use_calibrated_mapping:
                hud_extra_lines.append(
                    f"Mapping: piecewise statico (5 punti) | "
                    f"{_format_fusion_status(fusion_eyes, active_eyes)}"
                )
            elif ir_calibrating:
                hud_extra_lines.append(
                    f"Mapping: calibrazione IR ({len(heatmap_session.calibrated_edges)}/4 bordi) | "
                    f"{_format_fusion_status(fusion_eyes, active_eyes)}"
                )
            elif heatmap_session.ready:
                hud_extra_lines.append(f"Mapping: 5 punti | {_format_fusion_status(fusion_eyes, active_eyes)}")
            else:
                hud_extra_lines.append(
                    f"Mapping: premi C + frecce | "
                    f"{_format_fusion_status(fusion_eyes, active_eyes)}"
                )
            if fov_screen_enabled:
                hud_extra_lines.append(
                    f"FOV front {FRONT_FOV_DEG:.0f}° | IR↔front: "
                    f"{'OK' if eye_tracker.calibrated else 'premi C'}"
                )
            if head_pose_enabled:
                hud_extra_lines.append("Compensazione testa 6DoF: ON (delta su statico)")
            else:
                hud_extra_lines.append("Compensazione testa 6DoF: OFF")
            hud_extra_lines.append(camera_panel.status_line())
            if show_corner_overlay:
                hud_extra_lines.append(corner_markers.status_line())
            if aruco_tracker is not None:
                hud_extra_lines.append(aruco_tracker.status_line())
            if pose_mapper is not None:
                hud_extra_lines.append(pose_mapper.status_line())

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
                    # Link IR gaze space to front-camera forward (FOV 60).
                    eye_tracker.calibrate_gaze_to_external(fusion_eyes)
                    saved, message = heatmap_session.set_center_calibration(calibration_gaze)
                    eye_tracker.reset_gaze_smoothing()
                    print(message)
                    if fov_screen_enabled:
                        print(
                            f"IR↔front link OK (FOV {FRONT_FOV_DEG:.0f}°). "
                            "Tieni i marker ArUco visibili per mappare sullo schermo."
                        )
                    if saved and pose_mapper is not None:
                        pose_mapper.reset()
                        pose_message = _record_pose_sample_from_history(
                            pose_mapper,
                            pose_history,
                            heatmap_session,
                            "center",
                            fusion_eyes,
                        )
                        if pose_message:
                            print(pose_message)
                    print("Guarda il MONITOR fisico. I punti verdi sono gli ancoraggi salvati.")
            elif key == ord("h"):
                eye_tracker.calibrated = False
                _, message = heatmap_session.handle_key(key, combined_gaze)
                if pose_mapper is not None:
                    pose_mapper.reset()
                pose_history.clear()
                print(message)
            elif key in (KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT):
                edge_by_key = {
                    KEY_UP: ("top", heatmap_session.calibrate_top),
                    KEY_DOWN: ("bottom", heatmap_session.calibrate_bottom),
                    KEY_LEFT: ("left", heatmap_session.calibrate_left),
                    KEY_RIGHT: ("right", heatmap_session.calibrate_right),
                }
                label, calibrate_fn = edge_by_key[key]
                saved, message = calibrate_fn(calibration_gaze)
                print(message)
                if saved and pose_mapper is not None:
                    print(
                        _record_pose_sample_from_history(
                            pose_mapper,
                            pose_history,
                            heatmap_session,
                            label,
                            fusion_eyes,
                        )
                    )
            elif key == ord("s"):
                out_path = os.path.join(ROOT_DIR, "gaze_heatmap.png")
                heatmap_session.save_png(out_path)
                print(f"Saved {out_path}")
            elif key == ord("0"):
                fusion_eyes = active_eyes
                eye_tracker.calibrated = False
                eye_tracker.reset_gaze_smoothing()
                heatmap_session.reset_calibration()
                heatmap_session.reset_heatmap()
                if pose_mapper is not None:
                    pose_mapper.reset()
                pose_history.clear()
                print("Gaze fusion: both eyes. Calibrazione resettata: premi C.")
            elif key == ord("1") and "left" in active_eyes:
                fusion_eyes = ("left",)
                eye_tracker.calibrated = False
                eye_tracker.reset_gaze_smoothing()
                heatmap_session.reset_calibration()
                heatmap_session.reset_heatmap()
                if pose_mapper is not None:
                    pose_mapper.reset()
                pose_history.clear()
                print("Gaze fusion: LEFT eye only. Calibrazione resettata: premi C.")
            elif key == ord("2") and "right" in active_eyes:
                fusion_eyes = ("right",)
                eye_tracker.calibrated = False
                eye_tracker.reset_gaze_smoothing()
                heatmap_session.reset_calibration()
                heatmap_session.reset_heatmap()
                if pose_mapper is not None:
                    pose_mapper.reset()
                pose_history.clear()
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
