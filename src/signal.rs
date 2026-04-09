//! Synthetic signal generation.
//!
//! Each function returns a `Vec<f64>` of time-domain samples. The number of
//! samples is computed as `⌈duration × sample_rate⌉`.
//!
//! Deterministic signals (sine, cosine, square) are normalised to the
//! amplitude range `[−1.0, 1.0]`. Noise signals are also scaled to
//! approximately this range.
//!
//! ## Example: generate and inspect a sine wave
//!
//! ```
//! use dsp_tools::signal::generate_sine;
//!
//! let samples = generate_sine(440.0, 1.0, 44100.0);
//!
//! // 44 100 samples for 1 second at 44.1 kHz
//! assert_eq!(samples.len(), 44100);
//!
//! // All samples are within the normalised amplitude range
//! assert!(samples.iter().all(|&s| s >= -1.0 && s <= 1.0));
//! ```

use rand::prelude::*;
use rand_distr::{Distribution, Normal};

fn get_sample_count(duration: f64, sample_rate: f64) -> usize {
    (duration * sample_rate).ceil() as usize
}

/// Generates a sine wave.
///
/// Produces `⌈duration × sample_rate⌉` samples of:
///
/// ```text
/// x[n] = sin(2π · frequency · n / sample_rate)
/// ```
///
/// The signal starts at 0 and is normalised to the range `[−1.0, 1.0]`.
///
/// # Arguments
///
/// * `frequency`   — Frequency of the sine wave in Hz.
/// * `duration`    — Length of the signal in seconds.
/// * `sample_rate` — Number of samples per second (Hz).
///
/// # Returns
///
/// A `Vec<f64>` of length `⌈duration × sample_rate⌉`.
///
/// # Examples
///
/// ```
/// use dsp_tools::signal::generate_sine;
/// use std::f64::consts::PI;
///
/// let freq = 1.0;        // 1 Hz
/// let sample_rate = 1000.0;
/// let samples = generate_sine(freq, 1.0, sample_rate);
///
/// assert_eq!(samples.len(), 1000);
///
/// // t = 0  →  sin(0) = 0
/// assert!(samples[0].abs() < 1e-10);
///
/// // t = T/4  →  sin(π/2) = 1
/// let quarter = (sample_rate / (4.0 * freq)).round() as usize;
/// assert!((samples[quarter] - 1.0).abs() < 1e-3);
///
/// // t = T/2  →  sin(π) ≈ 0
/// let half = (sample_rate / (2.0 * freq)).round() as usize;
/// assert!(samples[half].abs() < 1e-3);
/// ```
pub fn generate_sine(frequency: f64, duration: f64, sample_rate: f64) -> Vec<f64> {
    let sample_count = get_sample_count(duration, sample_rate);

    (0..sample_count)
        .map(|n| {
            let time = (n as f64) / sample_rate;
            let phase: f64 = 2.0 * std::f64::consts::PI * frequency * time;
            phase.sin()
        })
        .collect()
}

/// Generates a cosine wave.
///
/// Produces `⌈duration × sample_rate⌉` samples of:
///
/// ```text
/// x[n] = cos(2π · frequency · n / sample_rate)
/// ```
///
/// The signal starts at its maximum value of 1.0 and is normalised to
/// `[−1.0, 1.0]`.
///
/// # Arguments
///
/// * `frequency`   — Frequency of the cosine wave in Hz.
/// * `duration`    — Length of the signal in seconds.
/// * `sample_rate` — Number of samples per second (Hz).
///
/// # Returns
///
/// A `Vec<f64>` of length `⌈duration × sample_rate⌉`.
///
/// # Examples
///
/// ```
/// use dsp_tools::signal::generate_cosine;
///
/// let freq = 1.0;
/// let sample_rate = 1000.0;
/// let samples = generate_cosine(freq, 1.0, sample_rate);
///
/// assert_eq!(samples.len(), 1000);
///
/// // t = 0  →  cos(0) = 1
/// assert!((samples[0] - 1.0).abs() < 1e-10);
///
/// // t = T/4  →  cos(π/2) ≈ 0
/// let quarter = (sample_rate / (4.0 * freq)).round() as usize;
/// assert!(samples[quarter].abs() < 1e-3);
///
/// // t = T/2  →  cos(π) = -1
/// let half = (sample_rate / (2.0 * freq)).round() as usize;
/// assert!((samples[half] + 1.0).abs() < 1e-3);
/// ```
pub fn generate_cosine(frequency: f64, duration: f64, sample_rate: f64) -> Vec<f64> {
    let sample_count = get_sample_count(duration, sample_rate);

    (0..sample_count)
        .map(|n| {
            let time = (n as f64) / sample_rate;
            let phase: f64 = 2.0 * std::f64::consts::PI * frequency * time;
            phase.cos()
        })
        .collect()
}

/// Generates a square wave.
///
/// Produces `⌈duration × sample_rate⌉` samples that alternate between `+1.0`
/// and `−1.0` with the given frequency:
///
/// ```text
/// x[n] = +1.0  if  (n / sample_rate) mod T  <  T / 2
///         −1.0  otherwise
/// ```
///
/// where `T = 1 / frequency` is the period in seconds.
///
/// # Arguments
///
/// * `frequency`   — Frequency of the square wave in Hz.
/// * `duration`    — Length of the signal in seconds.
/// * `sample_rate` — Number of samples per second (Hz).
///
/// # Returns
///
/// A `Vec<f64>` of length `⌈duration × sample_rate⌉`. Every element is
/// exactly `+1.0` or `−1.0`.
///
/// # Examples
///
/// ```
/// use dsp_tools::signal::generate_square;
///
/// let samples = generate_square(1.0, 1.0, 1000.0);
///
/// assert_eq!(samples.len(), 1000);
///
/// // All values are exactly ±1
/// assert!(samples.iter().all(|&s| s == 1.0 || s == -1.0));
///
/// // First sample is in the positive half-cycle
/// assert_eq!(samples[0], 1.0);
///
/// // Just past the halfway point of the first period the signal is −1
/// assert_eq!(samples[501], -1.0);
/// ```
pub fn generate_square(frequency: f64, duration: f64, sample_rate: f64) -> Vec<f64> {
    let sample_count = get_sample_count(duration, sample_rate);
    let period = 1.0 / frequency;

    (0..sample_count)
        .map(|n| {
            let time = (n as f64) / sample_rate;
            if time % period < (period / 2.0) {
                1.0
            } else {
                -1.0
            }
        })
        .collect()
}

/// Generates white noise uniformly distributed in `[−1.0, 1.0)`.
///
/// Each sample is drawn independently from a uniform distribution over
/// `[−1.0, 1.0)`. The output has a flat power spectral density across all
/// frequencies up to the Nyquist limit.
///
/// # Arguments
///
/// * `duration`    — Length of the signal in seconds.
/// * `sample_rate` — Number of samples per second (Hz).
///
/// # Returns
///
/// A `Vec<f64>` of length `⌈duration × sample_rate⌉`. All samples are in
/// `[−1.0, 1.0)`.
///
/// # Examples
///
/// ```
/// use dsp_tools::signal::generate_white_noise;
///
/// let samples = generate_white_noise(1.0, 8000.0);
///
/// assert_eq!(samples.len(), 8000);
/// assert!(samples.iter().all(|&s| s >= -1.0 && s < 1.0));
/// ```
pub fn generate_white_noise(duration: f64, sample_rate: f64) -> Vec<f64> {
    let sample_count = get_sample_count(duration, sample_rate);
    let mut rng = rand::rng();

    (0..sample_count)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect()
}

/// Generates Gaussian (normally distributed) noise centred at zero.
///
/// Each sample is drawn from `N(0, 0.33)`. With a standard deviation of 0.33,
/// roughly 99.7 % of samples fall within `[−1.0, 1.0]` (three sigma). A small
/// fraction of samples may exceed this range.
///
/// # Arguments
///
/// * `duration`    — Length of the signal in seconds.
/// * `sample_rate` — Number of samples per second (Hz).
///
/// # Returns
///
/// A `Vec<f64>` of length `⌈duration × sample_rate⌉`. Most samples are in
/// `[−1.0, 1.0]` but values outside this range are possible.
///
/// # Examples
///
/// ```
/// use dsp_tools::signal::generate_gaussian_noise;
///
/// let samples = generate_gaussian_noise(1.0, 8000.0);
///
/// assert_eq!(samples.len(), 8000);
/// // The mean should be close to zero over a large sample
/// let mean = samples.iter().sum::<f64>() / samples.len() as f64;
/// assert!(mean.abs() < 0.05);
/// ```
pub fn generate_gaussian_noise(duration: f64, sample_rate: f64) -> Vec<f64> {
    let sample_count = get_sample_count(duration, sample_rate);
    let mut rng = rand::rng();

    // mean: 0.0, std. dev 0.33
    // center values around 0 and ensure they are all within 3 std. dev of the mean
    // ensures that most values are within -1.0..1.0
    let normal = Normal::new(0.0, 0.33).unwrap();

    (0..sample_count).map(|_| normal.sample(&mut rng)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    fn expected_sample_count(duration: f64, sample_rate: f64) -> usize {
        (duration * sample_rate).ceil() as usize
    }

    // --- generate_sine ---

    #[test]
    fn generate_sine_sample_count() {
        let out = generate_sine(440.0, 1.0, 44100.0);
        assert_eq!(out.len(), expected_sample_count(1.0, 44100.0));
    }

    #[test]
    fn generate_sine_range() {
        let out = generate_sine(440.0, 0.1, 44100.0);
        for s in &out {
            assert!(*s >= -1.0 && *s <= 1.0, "sample out of range: {s}");
        }
    }

    #[test]
    fn generate_sine_known_values() {
        // At t=0 sin(0) = 0; at t=T/4 sin(π/2) = 1.
        let freq = 1.0;
        let sample_rate = 1000.0;
        let out = generate_sine(freq, 1.0, sample_rate);

        // n=0 → t=0 → sin(0) = 0
        assert!(approx_eq(out[0], 0.0, EPSILON));

        // n = sample_rate/4 → t = 1/4 period → sin(π/2) = 1
        let quarter = (sample_rate / (4.0 * freq)).round() as usize;
        assert!(approx_eq(out[quarter], 1.0, 1e-3));

        // n = sample_rate/2 → t = 1/2 period → sin(π) ≈ 0
        let half = (sample_rate / (2.0 * freq)).round() as usize;
        assert!(approx_eq(out[half], 0.0, 1e-3));
    }

    // --- generate_cosine ---

    #[test]
    fn generate_cosine_sample_count() {
        let out = generate_cosine(440.0, 1.0, 44100.0);
        assert_eq!(out.len(), expected_sample_count(1.0, 44100.0));
    }

    #[test]
    fn generate_cosine_range() {
        let out = generate_cosine(440.0, 0.1, 44100.0);
        for s in &out {
            assert!(*s >= -1.0 && *s <= 1.0, "sample out of range: {s}");
        }
    }

    #[test]
    fn generate_cosine_known_values() {
        // At t=0 cos(0) = 1; at t=T/4 cos(π/2) = 0; at t=T/2 cos(π) = -1.
        let freq = 1.0;
        let sample_rate = 1000.0;
        let out = generate_cosine(freq, 1.0, sample_rate);

        assert!(approx_eq(out[0], 1.0, EPSILON));

        let quarter = (sample_rate / (4.0 * freq)).round() as usize;
        assert!(approx_eq(out[quarter], 0.0, 1e-3));

        let half = (sample_rate / (2.0 * freq)).round() as usize;
        assert!(approx_eq(out[half], -1.0, 1e-3));
    }

    // --- generate_square ---

    #[test]
    fn generate_square_sample_count() {
        let out = generate_square(440.0, 1.0, 44100.0);
        assert_eq!(out.len(), expected_sample_count(1.0, 44100.0));
    }

    #[test]
    fn generate_square_only_unit_values() {
        let out = generate_square(440.0, 0.1, 44100.0);
        for s in &out {
            assert!(
                approx_eq(*s, 1.0, EPSILON) || approx_eq(*s, -1.0, EPSILON),
                "expected ±1, got {s}"
            );
        }
    }

    #[test]
    fn generate_square_first_half_positive() {
        // The first sample should fall in the positive half of the first period.
        let freq = 1.0;
        let sample_rate = 1000.0;
        let out = generate_square(freq, 1.0, sample_rate);
        assert!(approx_eq(out[0], 1.0, EPSILON));
    }

    #[test]
    fn generate_square_second_half_negative() {
        // Samples just past the halfway point of the first period must be -1.
        let freq = 1.0;
        let sample_rate = 1000.0;
        let out = generate_square(freq, 1.0, sample_rate);
        // T/2 + a small offset lands in the negative half
        let mid = (sample_rate / (2.0 * freq)) as usize + 1;
        assert!(approx_eq(out[mid], -1.0, EPSILON));
    }

    // --- generate_white_noise ---

    #[test]
    fn generate_white_noise_sample_count() {
        let out = generate_white_noise(1.0, 44100.0);
        assert_eq!(out.len(), expected_sample_count(1.0, 44100.0));
    }

    #[test]
    fn generate_white_noise_range() {
        let out = generate_white_noise(1.0, 44100.0);
        for s in &out {
            assert!(*s >= -1.0 && *s < 1.0, "sample out of range: {s}");
        }
    }

    // --- generate_gaussian_noise ---

    #[test]
    fn generate_gaussian_noise_sample_count() {
        let out = generate_gaussian_noise(1.0, 44100.0);
        assert_eq!(out.len(), expected_sample_count(1.0, 44100.0));
    }
}
