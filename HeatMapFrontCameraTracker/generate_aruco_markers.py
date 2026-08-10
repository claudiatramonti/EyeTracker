"""
Fase 0 — Genera i marker ArUco da stampare.

Usage:
  cd HeatMapFrontCameraTracker
  python generate_aruco_markers.py

Stampa i PNG in aruco_markers/ e incollali sul monitor:
  ID 0 = angolo in alto a sinistra
  ID 1 = angolo in alto a destra
  ID 2 = angolo in basso a destra
  ID 3 = angolo in basso a sinistra
"""

import os

import aruco_screen

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "aruco_markers")


def main():
    paths = aruco_screen.generate_marker_sheet(OUT)
    print("Marker salvati in:", OUT)
    for path in paths:
        print(" ", path)
    print()
    print("Layout sul monitor:")
    print("  [0] -------- [1]")
    print("   |           |")
    print("   |  SCHERMO  |")
    print("   |           |")
    print("  [3] -------- [2]")
    print()
    print("Consiglio: stampa ~5 cm per lato, incolla sui 4 angoli (bezel o bordo visibile alla front camera).")


if __name__ == "__main__":
    main()
