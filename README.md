# DSP Tools

A command-line DSP toolkit and Rust library for signal generation, spectral analysis, windowing, and FIR filtering.

---

## Building

```bash
cargo build --release
# binary is at target/release/dsp_tools
```

---

## Data Format

All commands exchange data as a raw stream of **little-endian IEEE 754 64-bit floats** (`f64`). This means every command's stdout can be piped directly into the next command's stdin.

---

## CLI Commands

### `signal` — Signal Generator

Generates a signal and writes it to stdout.

| Flag | Short | Description |
|---|---|---|
| `--function` | `-f` | Signal type: `sine`, `cosine`, `square`, `noise-w`, `noise-g` |
| `--sample-rate` | `-s` | Sample rate in Hz |
| `--frequency` | `-w` | Signal frequency in Hz (ignored for noise) |
| `--duration` | `-d` | Duration in seconds |

```bash
# 440 Hz sine wave, 44.1 kHz sample rate, 1 second
dsp_tools signal -f sine -s 44100 -w 440 -d 1.0 > sine.bin

# 1 kHz cosine wave, 8 kHz sample rate, 0.5 seconds
dsp_tools signal -f cosine -s 8000 -w 1000 -d 0.5 > cosine.bin

# 500 Hz square wave, 44.1 kHz sample rate, 2 seconds
dsp_tools signal -f square -s 44100 -w 500 -d 2.0 > square.bin

# 1 second of white noise at 44.1 kHz (--frequency required but unused)
dsp_tools signal -f noise-w -s 44100 -w 1 -d 1.0 > white_noise.bin

# 1 second of Gaussian noise (mean=0, σ=0.33)
dsp_tools signal -f noise-g -s 44100 -w 1 -d 1.0 > gaussian_noise.bin
```

---

### `ft` — Fourier Transform

Reads a signal from stdin and writes the frequency-domain representation to stdout.

| Flag | Short | Description |
|---|---|---|
| `--transform-type` | `-t` | `dft` (O(N²)) or `fft` (Cooley-Tukey, O(N log N)) |
| `--output-format` | `-o` | `magnitude` (default), `power`, `phase`, `complex` |

Non-power-of-two inputs are automatically zero-padded to the next power of two when using `fft`.

```bash
# Magnitude spectrum of a sine wave using FFT
dsp_tools signal -f sine -s 44100 -w 440 -d 0.1 | dsp_tools ft -t fft > spectrum.bin

# Power spectrum
dsp_tools signal -f sine -s 8000 -w 440 -d 0.1 | dsp_tools ft -t fft -o power > power.bin

# Phase spectrum
dsp_tools signal -f sine -s 8000 -w 440 -d 0.1 | dsp_tools ft -t fft -o phase > phase.bin

# Complex output (interleaved re/im f64 pairs)
dsp_tools signal -f sine -s 8000 -w 440 -d 0.1 | dsp_tools ft -t fft -o complex > complex.bin

# DFT on a short signal
dsp_tools signal -f cosine -s 8000 -w 1000 -d 0.01 | dsp_tools ft -t dft > dft_out.bin
```

---

### `window` — Window Functions

Reads a signal from stdin, multiplies it by a window function, and writes the result to stdout. Apply a window before the FFT to reduce spectral leakage.

| Flag | Short | Description |
|---|---|---|
| `--window-function` | `-z` | `hann`, `hamming`, or `blackman` |

```bash
# Apply a Hann window
cat signal.bin | dsp_tools window -z hann > windowed.bin

# Apply a Hamming window
cat signal.bin | dsp_tools window -z hamming > windowed.bin

# Apply a Blackman window (steepest sidelobe rolloff)
cat signal.bin | dsp_tools window -z blackman > windowed.bin

# Window a signal before computing its FFT
dsp_tools signal -f sine -s 44100 -w 440 -d 0.1 \
  | dsp_tools window -z hann \
  | dsp_tools ft -t fft > spectrum.bin
```

---

### `filter` — FIR Filter

Reads a signal from stdin, designs a FIR filter, applies it, and writes the result to stdout.

**Common flags:**

| Flag | Short | Description |
|---|---|---|
| `--sample-rate` | `-s` | Sample rate in Hz |
| `--taps` | `-n` | Number of filter taps (must be odd; more taps = sharper roll-off) |
| `--window-function` | `-z` | Window for filter design: `hann`, `hamming`, `blackman` (default) |

#### `low-pass` — Low-Pass Filter

Passes frequencies below the cutoff and attenuates everything above.

```bash
# Remove everything above 1 kHz from a 44.1 kHz signal
dsp_tools signal -f noise-w -s 44100 -w 1 -d 1.0 \
  | dsp_tools filter -s 44100 -n 101 low-pass --cutoff 1000 > lp_filtered.bin

# Gentler roll-off with fewer taps
dsp_tools signal -f noise-w -s 8000 -w 1 -d 1.0 \
  | dsp_tools filter -s 8000 -n 31 low-pass --cutoff 500 > lp_filtered.bin
```

#### `high-pass` — High-Pass Filter

Passes frequencies above the cutoff and attenuates everything below.

```bash
# Remove DC and low-frequency content below 300 Hz
dsp_tools signal -f noise-w -s 8000 -w 1 -d 1.0 \
  | dsp_tools filter -s 8000 -n 101 high-pass --cutoff 300 > hp_filtered.bin
```

#### `band-pass` — Band-Pass Filter

Passes frequencies between two cutoffs and attenuates everything outside.

```bash
# Telephone-band filter: pass 300 Hz – 3400 Hz
dsp_tools signal -f noise-w -s 8000 -w 1 -d 1.0 \
  | dsp_tools filter -s 8000 -n 101 band-pass --cutoff-low 300 --cutoff-high 3400 > bp_filtered.bin

# Voice-band from a wideband signal
dsp_tools signal -f noise-w -s 44100 -w 1 -d 1.0 \
  | dsp_tools filter -s 44100 -n 201 band-pass --cutoff-low 80 --cutoff-high 8000 > voice_band.bin
```

#### `notch` — Notch (Band-Stop) Filter

Attenuates frequencies between two cutoffs and passes everything outside.

```bash
# Remove 60 Hz power-line interference (±5 Hz)
dsp_tools signal -f sine -s 44100 -w 440 -d 1.0 \
  | dsp_tools filter -s 44100 -n 201 notch --cutoff-low 55 --cutoff-high 65 > notched.bin
```

#### `derivative` — Derivative Filter

Applies a three-tap FIR differentiator `[0.5, 0.0, -0.5]`. Useful for edge detection or approximating the derivative of a signal.

```bash
dsp_tools signal -f sine -s 44100 -w 440 -d 1.0 \
  | dsp_tools filter -s 44100 -n 3 derivative > derivative.bin
```

#### `matched` — Matched Filter

Cross-correlates the input signal with a template to detect occurrences of that pattern. The template file must be in the same binary f64 format.

```bash
# Generate a template pulse
dsp_tools signal -f sine -s 44100 -w 1000 -d 0.01 > template.bin

# Apply matched filter to detect the pulse in a noisy signal
dsp_tools signal -f noise-w -s 44100 -w 1 -d 1.0 \
  | dsp_tools filter -s 44100 -n 101 matched --template-file template.bin > correlation.bin
```

---

## Pipeline Examples

Commands chain naturally via Unix pipes. All intermediate data is the raw f64 binary stream.

```bash
# Full analysis pipeline: generate → window → FFT → magnitude spectrum
dsp_tools signal -f sine -s 44100 -w 440 -d 0.5 \
  | dsp_tools window -z blackman \
  | dsp_tools ft -t fft \
  > spectrum.bin

# Denoise pipeline: low-pass then high-pass to isolate a frequency band
dsp_tools signal -f noise-w -s 44100 -w 1 -d 2.0 \
  | dsp_tools filter -s 44100 -n 101 low-pass --cutoff 3000 \
  | dsp_tools filter -s 44100 -n 101 high-pass --cutoff 200 \
  > band_filtered.bin

# Visualise with the bundled Python script (requires numpy + matplotlib)
dsp_tools signal -f sine -s 44100 -w 440 -d 0.1 \
  | dsp_tools window -z hann \
  | dsp_tools ft -t fft \
  > freq.bin
python scripts/see_data.py freq.bin --mode magnitude
```

---

## Library Usage

The four DSP modules are available as a library crate.

```rust
use dsp_tools::{filter, ft, signal, window};
use dsp_tools::window::WindowFunction;

// --- Signal generation ---
let sine    = signal::generate_sine(440.0, 1.0, 44100.0);
let cosine  = signal::generate_cosine(440.0, 1.0, 44100.0);
let square  = signal::generate_square(440.0, 1.0, 44100.0);
let noise_w = signal::generate_white_noise(1.0, 44100.0);
let noise_g = signal::generate_gaussian_noise(1.0, 44100.0);

// --- Window functions (applied in-place) ---
let mut samples = signal::generate_sine(440.0, 0.1, 44100.0);
window::apply_hann(&mut samples);
window::apply_hamming(&mut samples);
window::apply_blackman(&mut samples);

// --- Fourier transforms ---
let dft_out = ft::dft(&samples);   // O(N²), any input length
let fft_out = ft::fft(&samples);   // O(N log N), zero-pads to next power of two

// --- FIR filter design ---
let lp_taps = filter::generate_low_pass(101, 0.1, WindowFunction::Blackman).unwrap();
let hp_taps = filter::generate_high_pass(101, 0.3, WindowFunction::Hann).unwrap();
let bp_taps = filter::generate_band_pass(101, 0.05, 0.2, WindowFunction::Hamming).unwrap();
let notch   = filter::generate_notch(201, 0.05, 0.07, WindowFunction::Blackman).unwrap();

// Cutoff frequencies are normalised: fc = Hz / sample_rate (must be 0 < fc ≤ 0.5)

// --- Apply a FIR filter ---
let signal  = signal::generate_white_noise(1.0, 44100.0);
let filtered = filter::apply_fir(&signal, &lp_taps).unwrap();
```

---

## Scripts

| Script | Description |
|---|---|
| `scripts/make_data.py` | Generates a small hardcoded binary f64 stream for testing |
| `scripts/see_data.py` | Plots a binary f64 file using matplotlib. Supports `magnitude`, `power`, `phase`, and `complex` modes |

```bash
python scripts/see_data.py freq.bin --mode magnitude
python scripts/see_data.py complex.bin --mode complex
```
