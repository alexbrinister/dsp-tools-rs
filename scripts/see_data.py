#!/usr/bin/env python

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Plot binary output from the Rust DSP CLI.")
    parser.add_argument("filename", nargs="?", default="freq.bin",
                        help="Path to the binary file (default: freq.bin)")
    parser.add_argument("-m", "--mode", choices=["magnitude", "power", "phase", "complex"],
                        default="magnitude", help="The output format of the binary data")

    args = parser.parse_args()

    # Read the binary data
    try:
        if args.mode == "complex":
            # Interleaved f64 pairs map perfectly to numpy's complex128 memory layout
            data = np.fromfile(args.filename, dtype=np.complex128)
        else:
            data = np.fromfile(args.filename, dtype=np.float64)
    except FileNotFoundError:
        print(f"Error: Could not find file '{args.filename}'.")
        sys.exit(1)

    # Setup the plot
    plt.figure(figsize=(10, 6))

    if args.mode in ["magnitude", "power", "phase"]:
        plt.plot(data)
        plt.title(f"FFT Output: {args.mode.capitalize()}")
        plt.xlabel("Frequency Bin (k)")

        if args.mode == "phase":
            plt.ylabel("Phase (Radians)")
        else:
            plt.ylabel(args.mode.capitalize())

        plt.grid(True)

    elif args.mode == "complex":
        # Create two subplots for Real and Imaginary parts
        plt.subplot(2, 1, 1)
        plt.plot(data.real, label="Real Part", color="blue")
        plt.title("FFT Output: Complex")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(data.imag, label="Imaginary Part", color="orange")
        plt.xlabel("Frequency Bin (k)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
