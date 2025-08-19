#!/usr/bin/env python3
"""
URDF -> MJCF XML converter using MuJoCo's built-in URDF loader.

Usage:
  python3 urdf_to_mjcf.py <input.urdf> [-o <output.xml>]

Notes:
- Relative mesh paths are resolved by temporarily chdir to the URDF's directory.
- URDF may include <mujoco><compiler meshdir=.../> directives which MuJoCo honors.
"""

import argparse
import os
import sys
import mujoco  # pip install mujoco


def derive_output_path(input_path: str) -> str:
    base, ext = os.path.splitext(input_path)
    if ext.lower() in (".urdf", ".xml"):
        return base + "_converted.xml"
    return input_path + ".xml"


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert URDF to MJCF XML using MuJoCo")
    parser.add_argument("input", help="Path to input .urdf (or .xml)")
    parser.add_argument("-o", "--output", help="Path to output MJCF .xml")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.isfile(input_path):
        print(f"[Error] Input not found: {input_path}", file=sys.stderr)
        return 1

    output_path = os.path.abspath(args.output) if args.output else os.path.abspath(derive_output_path(input_path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load model with paths resolved relative to URDF directory
    urdf_dir = os.path.dirname(input_path)
    urdf_file = os.path.basename(input_path)
    prev_cwd = os.getcwd()
    try:
        os.chdir(urdf_dir)
        model = mujoco.MjModel.from_xml_path(urdf_file)
    except Exception as ex:
        print(f"[Error] Failed to load URDF: {input_path}\n{ex}", file=sys.stderr)
        return 2
    finally:
        os.chdir(prev_cwd)

    # Save compiled MJCF (MuJoCo 2.3.x Python API: mj_saveLastXML(filename, model) -> None)
    try:
        mujoco.mj_saveLastXML(output_path, model)
    except Exception as ex:
        print(f"[Error] Failed to save MJCF to: {output_path}\n{ex}", file=sys.stderr)
        return 3

    print(f"[OK] Saved MJCF: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

