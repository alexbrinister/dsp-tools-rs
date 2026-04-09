//! # DSP Tools
//!
//! A library for digital signal processing covering signal generation,
//! spectral analysis, window functions, and FIR filter design and application.
//!
//! ## Modules
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`signal`] | Generate synthetic signals (sine, cosine, square, noise) |
//! | [`ft`] | Discrete and Fast Fourier transforms |
//! | [`window`] | Hann, Hamming, and Blackman window functions |
//! | [`filter`] | FIR filter design and application |
//!
//! ## Quick Start
//!
//! ```rust
//! use dsp_tools::{signal, window, ft, filter};
//! use dsp_tools::window::WindowFunction;
//!
//! // 1. Generate a 440 Hz sine wave sampled at 8 kHz for 0.1 s
//! let mut samples = signal::generate_sine(440.0, 0.1, 8000.0);
//!
//! // 2. Apply a Hann window to reduce spectral leakage before the FFT
//! window::apply_hann(&mut samples);
//!
//! // 3. Compute the frequency-domain representation
//! let spectrum = ft::fft(&samples);
//!
//! // 4. Design a low-pass FIR filter (normalised cutoff = 440 / 8000 ≈ 0.055)
//! let taps = filter::generate_low_pass(31, 0.055, WindowFunction::Blackman).unwrap();
//!
//! // 5. Apply the filter to the original signal
//! let filtered = filter::apply_fir(&samples, &taps).unwrap();
//! assert_eq!(filtered.len(), samples.len());
//! ```

/// Filter generation and application.
pub mod filter;
/// Fourier transforms (DFT and FFT).
pub mod ft;
/// Signal generation (sine, cosine, square, noise).
pub mod signal;
/// Window functions (Hann, Hamming, Blackman).
pub mod window;
