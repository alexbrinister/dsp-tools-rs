//! FIR filter design and application.
//!
//! This module provides windowed-sinc FIR filter design for the four standard
//! filter types, plus a convolution function to apply any tap vector to a signal.
//!
//! ## Filter Design Functions
//!
//! | Function | Description |
//! |----------|-------------|
//! | [`generate_low_pass`](crate::filter::generate_low_pass) | Passes frequencies below a cutoff |
//! | [`generate_high_pass`](crate::filter::generate_high_pass) | Passes frequencies above a cutoff |
//! | [`generate_band_pass`](crate::filter::generate_band_pass) | Passes a band between two cutoffs |
//! | [`generate_notch`](crate::filter::generate_notch) | Rejects a band between two cutoffs |
//!
//! ## Applying a Filter
//!
//! [`apply_fir`](crate::filter::apply_fir) convolves any `&[f64]` tap vector with an input signal.
//!
//! ## Normalised Cutoff Frequencies
//!
//! All cutoff frequency parameters (`fc`, `fc1`, `fc2`) are **normalised**
//! relative to the sample rate:
//!
//! ```text
//! fc = cutoff_Hz / sample_rate_Hz
//! ```
//!
//! Valid values are in `(0.0, 0.5]`, where 0.5 represents the Nyquist
//! frequency. For example, a 1 kHz cutoff at an 8 kHz sample rate is
//! `fc = 1000.0 / 8000.0 = 0.125`.
//!
//! ## Number of Taps
//!
//! More taps produce a sharper transition band at the cost of a longer
//! impulse response (group delay = `(num_taps − 1) / 2` samples). An odd
//! number of taps is required so the filter has a symmetric centre
//! coefficient with zero group-delay distortion.
//!
//! ## Example: low-pass filter pipeline
//!
//! ```
//! use dsp_tools::signal::generate_white_noise;
//! use dsp_tools::filter::{generate_low_pass, apply_fir};
//! use dsp_tools::window::WindowFunction;
//!
//! // Generate 0.1 s of white noise at 8 kHz
//! let noise = generate_white_noise(0.1, 8000.0);
//!
//! // Design a 31-tap low-pass filter with a cutoff at 1 kHz (fc = 0.125)
//! let taps = generate_low_pass(31, 0.125, WindowFunction::Blackman).unwrap();
//! assert_eq!(taps.len(), 31);
//!
//! // Apply the filter
//! let filtered = apply_fir(&noise, &taps).unwrap();
//! assert_eq!(filtered.len(), noise.len());
//! ```

use crate::window;
use crate::window::WindowFunction;
use thiserror::Error;

/// An error returned by filter design or application functions.
///
/// # Variants
///
/// - [`FilterError::ZeroTaps`] — `num_taps` was 0, or `taps` slice was empty.
/// - [`FilterError::NyquistLimitExceeded`] — A cutoff frequency was greater
///   than 0.5 (the Nyquist limit for normalised frequencies).
/// - [`FilterError::InvalidBand`] — The lower cutoff was greater than or equal
///   to the upper cutoff in a two-cutoff filter.
///
/// # Example
///
/// ```
/// use dsp_tools::filter::{generate_low_pass, FilterError};
/// use dsp_tools::window::WindowFunction;
///
/// // Zero taps
/// assert!(matches!(
///     generate_low_pass(0, 0.1, WindowFunction::Hann),
///     Err(FilterError::ZeroTaps)
/// ));
///
/// // Cutoff exceeds Nyquist
/// assert!(matches!(
///     generate_low_pass(31, 0.6, WindowFunction::Hann),
///     Err(FilterError::NyquistLimitExceeded(_))
/// ));
/// ```
#[derive(Error, Debug)]
pub enum FilterError {
    /// Number of filter taps is zero.
    #[error("number of taps must be greater than 0")]
    ZeroTaps,

    /// The cutoff frequency exceeds the Nyquist limit.
    #[error("cutoff frequency ({0}) is past the Nyquist limit of 0.5")]
    NyquistLimitExceeded(f64),

    /// The lower cutoff frequency is greater than or equal to the upper cutoff frequency.
    #[error("lower cutoff ({0}) must be less than upper cutoff ({1})")]
    InvalidBand(f64, f64),
}

/// Applies a Finite Impulse Response (FIR) filter to an input signal.
///
/// Computes the causal discrete convolution of `input` with `taps`:
///
/// ```text
/// y[n] = Σ_{k=0}^{min(n, K−1)}  taps[k] · input[n − k]
/// ```
///
/// where `K = taps.len()`. The output has the same length as `input`.
/// The filter is applied causally, so the first `K − 1` output samples are
/// computed with fewer than `K` taps (equivalent to zero-padding the input
/// on the left).
///
/// # Arguments
///
/// * `input` — The signal to filter.
/// * `taps`  — The FIR filter coefficients (impulse response). Use the
///   `generate_*` functions in this module to design appropriate tap vectors.
///
/// # Returns
///
/// A `Vec<f64>` of the same length as `input`.
///
/// # Errors
///
/// Returns [`FilterError::ZeroTaps`] if `taps` is empty.
///
/// # Examples
///
/// An identity filter (single tap of `1.0`) leaves the signal unchanged:
///
/// ```
/// use dsp_tools::filter::apply_fir;
///
/// let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let output = apply_fir(&input, &[1.0]).unwrap();
/// assert_eq!(output, input);
/// ```
///
/// A two-tap averaging filter `[0.5, 0.5]` smooths the signal:
///
/// ```
/// use dsp_tools::filter::apply_fir;
///
/// let input  = vec![2.0, 4.0, 6.0];
/// let output = apply_fir(&input, &[0.5, 0.5]).unwrap();
///
/// // y[0] = 0.5·2                  = 1.0
/// // y[1] = 0.5·4 + 0.5·2          = 3.0
/// // y[2] = 0.5·6 + 0.5·4          = 5.0
/// assert!((output[0] - 1.0).abs() < 1e-10);
/// assert!((output[1] - 3.0).abs() < 1e-10);
/// assert!((output[2] - 5.0).abs() < 1e-10);
/// ```
///
/// Empty taps returns an error:
///
/// ```
/// use dsp_tools::filter::{apply_fir, FilterError};
///
/// assert!(matches!(apply_fir(&[1.0, 2.0], &[]), Err(FilterError::ZeroTaps)));
/// ```
pub fn apply_fir(input: &[f64], taps: &[f64]) -> Result<Vec<f64>, FilterError> {
    if taps.is_empty() {
        return Err(FilterError::ZeroTaps);
    }

    Ok((0..input.len())
        .map(|n| {
            let k_max = n.min(taps.len() - 1);
            (0..=k_max).map(|k| taps[k] * input[n - k]).sum()
        })
        .collect())
}

/// Generates tap coefficients for a windowed-sinc low-pass FIR filter.
///
/// Computes an ideal sinc kernel centred at the middle tap, then multiplies
/// it by the chosen window function to control spectral leakage. The taps are
/// normalised so they sum to 1.0, ensuring unity gain at DC (0 Hz).
///
/// # Arguments
///
/// * `num_taps`         — Number of filter coefficients. Must be odd and > 0.
/// * `fc`               — Normalised cutoff frequency in `(0.0, 0.5]`.
///   Computed as `cutoff_Hz / sample_rate_Hz`.
/// * `window_function`  — Window applied to the sinc kernel.
///
/// # Returns
///
/// A `Vec<f64>` of length `num_taps` containing the filter coefficients.
/// The taps are symmetric around the centre index and sum to 1.0.
///
/// # Errors
///
/// - [`FilterError::ZeroTaps`] if `num_taps == 0`.
/// - [`FilterError::NyquistLimitExceeded`] if `fc > 0.5`.
///
/// # Examples
///
/// Design a 31-tap low-pass filter with cutoff at 1 kHz (8 kHz sample rate):
///
/// ```
/// use dsp_tools::filter::generate_low_pass;
/// use dsp_tools::window::WindowFunction;
///
/// let taps = generate_low_pass(31, 0.125, WindowFunction::Blackman).unwrap();
///
/// assert_eq!(taps.len(), 31);
///
/// // Taps sum to 1.0 (unity DC gain)
/// let sum: f64 = taps.iter().sum();
/// assert!((sum - 1.0).abs() < 1e-10);
/// ```
///
/// ```
/// use dsp_tools::filter::{generate_low_pass, FilterError};
/// use dsp_tools::window::WindowFunction;
///
/// assert!(matches!(generate_low_pass(0,  0.1, WindowFunction::Hann), Err(FilterError::ZeroTaps)));
/// assert!(matches!(generate_low_pass(31, 0.6, WindowFunction::Hann), Err(FilterError::NyquistLimitExceeded(_))));
/// ```
pub fn generate_low_pass(
    num_taps: usize,
    fc: f64,
    window_function: WindowFunction,
) -> Result<Vec<f64>, FilterError> {
    if num_taps == 0 {
        return Err(FilterError::ZeroTaps);
    }

    if !(0.0..=0.5).contains(&fc) {
        return Err(FilterError::NyquistLimitExceeded(fc));
    }

    let center_point = (num_taps - 1) / 2;
    let center_point_f64 = center_point as f64;

    let mut output: Vec<f64> = (0..num_taps)
        .map(|i| {
            let i_f64 = i as f64;

            if i_f64 == center_point_f64 {
                2.0 * fc
            } else {
                let adjusted_index: f64 = std::f64::consts::PI * (i_f64 - center_point_f64);
                let angle: f64 = 2.0 * fc * adjusted_index;
                angle.sin() / adjusted_index
            }
        })
        .collect();

    match window_function {
        WindowFunction::Hann => window::apply_hann(&mut output),
        WindowFunction::Hamming => window::apply_hamming(&mut output),
        WindowFunction::Blackman => window::apply_blackman(&mut output),
    };

    let sum: f64 = output.iter().sum();
    output.iter_mut().for_each(|tap| *tap /= sum);

    Ok(output)
}

/// Generates tap coefficients for a windowed-sinc high-pass FIR filter.
///
/// Derives the high-pass filter from the complementary low-pass design via
/// *spectral inversion*: each low-pass tap is negated and the centre tap has
/// 1.0 added to it:
///
/// ```text
/// hp[n] = −lp[n]          for n ≠ centre
/// hp[centre] = 1 − lp[centre]
/// ```
///
/// This subtracts the low-pass response from an all-pass filter, leaving
/// only the high-frequency content.
///
/// # Arguments
///
/// * `num_taps`         — Number of filter coefficients. Must be odd and > 0.
/// * `fc`               — Normalised cutoff frequency in `(0.0, 0.5]`.
/// * `window_function`  — Window applied to the underlying sinc kernel.
///
/// # Returns
///
/// A `Vec<f64>` of length `num_taps`.
///
/// # Errors
///
/// Propagates all errors from [`generate_low_pass`]:
/// - [`FilterError::ZeroTaps`] if `num_taps == 0`.
/// - [`FilterError::NyquistLimitExceeded`] if `fc > 0.5`.
///
/// # Examples
///
/// Design a 31-tap high-pass filter with cutoff at 2 kHz (8 kHz sample rate):
///
/// ```
/// use dsp_tools::filter::generate_high_pass;
/// use dsp_tools::window::WindowFunction;
///
/// let taps = generate_high_pass(31, 0.25, WindowFunction::Hann).unwrap();
/// assert_eq!(taps.len(), 31);
/// ```
///
/// ```
/// use dsp_tools::filter::{generate_high_pass, FilterError};
/// use dsp_tools::window::WindowFunction;
///
/// assert!(matches!(generate_high_pass(0,  0.1, WindowFunction::Hann), Err(FilterError::ZeroTaps)));
/// assert!(matches!(generate_high_pass(31, 0.6, WindowFunction::Hann), Err(FilterError::NyquistLimitExceeded(_))));
/// ```
pub fn generate_high_pass(
    num_taps: usize,
    fc: f64,
    window_function: WindowFunction,
) -> Result<Vec<f64>, FilterError> {
    let mut low_pass: Vec<f64> = generate_low_pass(num_taps, fc, window_function)?;
    let center_point: usize = (low_pass.len() - 1) / 2;

    low_pass.iter_mut().for_each(|tap| *tap *= -1.0);
    low_pass[center_point] += 1.0;
    Ok(low_pass)
}

/// Generates tap coefficients for a windowed-sinc band-pass FIR filter.
///
/// Constructs the band-pass filter as the **difference** of two low-pass
/// designs:
///
/// ```text
/// bp[n] = lp_fc2[n] − lp_fc1[n]
/// ```
///
/// This passes the band `(fc1, fc2)` and attenuates everything outside it.
///
/// # Arguments
///
/// * `num_taps`         — Number of filter coefficients. Must be odd and > 0.
/// * `fc1`              — Normalised lower cutoff frequency in `(0.0, 0.5]`.
/// * `fc2`              — Normalised upper cutoff frequency in `(0.0, 0.5]`.
/// * `window_function`  — Window applied to the underlying sinc kernels.
///
/// # Returns
///
/// A `Vec<f64>` of length `num_taps`.
///
/// # Errors
///
/// - [`FilterError::ZeroTaps`] if `num_taps == 0`.
/// - [`FilterError::NyquistLimitExceeded`] if either cutoff exceeds 0.5.
/// - [`FilterError::InvalidBand`] if `fc1 >= fc2`.
///
/// # Examples
///
/// Telephone-band filter: pass 300–3400 Hz at an 8 kHz sample rate:
///
/// ```
/// use dsp_tools::filter::generate_band_pass;
/// use dsp_tools::window::WindowFunction;
///
/// // fc1 = 300/8000 = 0.0375,  fc2 = 3400/8000 = 0.425
/// let taps = generate_band_pass(101, 0.0375, 0.425, WindowFunction::Blackman).unwrap();
/// assert_eq!(taps.len(), 101);
/// ```
///
/// ```
/// use dsp_tools::filter::{generate_band_pass, FilterError};
/// use dsp_tools::window::WindowFunction;
///
/// // fc1 >= fc2 is invalid
/// assert!(matches!(
///     generate_band_pass(31, 0.3, 0.1, WindowFunction::Hann),
///     Err(FilterError::InvalidBand(_, _))
/// ));
///
/// // Cutoff beyond Nyquist
/// assert!(matches!(
///     generate_band_pass(31, 0.1, 0.8, WindowFunction::Hann),
///     Err(FilterError::NyquistLimitExceeded(_))
/// ));
/// ```
pub fn generate_band_pass(
    num_taps: usize,
    fc1: f64,
    fc2: f64,
    window_function: WindowFunction,
) -> Result<Vec<f64>, FilterError> {
    if !(0.0..=0.5).contains(&fc1) {
        return Err(FilterError::NyquistLimitExceeded(fc1));
    }

    if !(0.0..=0.5).contains(&fc2) {
        return Err(FilterError::NyquistLimitExceeded(fc2));
    }

    if fc1 >= fc2 {
        return Err(FilterError::InvalidBand(fc1, fc2));
    }

    let low_pass_fc1: Vec<f64> = generate_low_pass(num_taps, fc1, window_function.clone())?;
    let low_pass_fc2: Vec<f64> = generate_low_pass(num_taps, fc2, window_function.clone())?;

    Ok(low_pass_fc2
        .iter()
        .zip(low_pass_fc1.iter())
        .map(|(high, low)| high - low)
        .collect())
}

/// Generates tap coefficients for a windowed-sinc notch (band-stop) FIR filter.
///
/// Constructs the notch filter as the **sum** of a low-pass and a high-pass
/// design:
///
/// ```text
/// notch[n] = lp_fc1[n] + hp_fc2[n]
/// ```
///
/// This passes everything outside `(fc1, fc2)` and attenuates the band
/// between the two cutoffs.
///
/// # Arguments
///
/// * `num_taps`         — Number of filter coefficients. Must be odd and > 0.
/// * `fc1`              — Normalised lower cutoff of the rejected band in `(0.0, 0.5]`.
/// * `fc2`              — Normalised upper cutoff of the rejected band in `(0.0, 0.5]`.
/// * `window_function`  — Window applied to the underlying sinc kernels.
///
/// # Returns
///
/// A `Vec<f64>` of length `num_taps`.
///
/// # Errors
///
/// - [`FilterError::ZeroTaps`] if `num_taps == 0`.
/// - [`FilterError::NyquistLimitExceeded`] if either cutoff exceeds 0.5.
/// - [`FilterError::InvalidBand`] if `fc1 >= fc2`.
///
/// # Examples
///
/// Remove 60 Hz mains hum (±5 Hz) from a 44.1 kHz signal:
///
/// ```
/// use dsp_tools::filter::generate_notch;
/// use dsp_tools::window::WindowFunction;
///
/// // fc1 = 55/44100 ≈ 0.00125,  fc2 = 65/44100 ≈ 0.00147
/// let taps = generate_notch(201, 55.0 / 44100.0, 65.0 / 44100.0, WindowFunction::Blackman).unwrap();
/// assert_eq!(taps.len(), 201);
/// ```
///
/// ```
/// use dsp_tools::filter::{generate_notch, FilterError};
/// use dsp_tools::window::WindowFunction;
///
/// // fc1 >= fc2 is invalid
/// assert!(matches!(
///     generate_notch(31, 0.3, 0.1, WindowFunction::Hann),
///     Err(FilterError::InvalidBand(_, _))
/// ));
/// ```
pub fn generate_notch(
    num_taps: usize,
    fc1: f64,
    fc2: f64,
    window_function: WindowFunction,
) -> Result<Vec<f64>, FilterError> {
    if !(0.0..=0.5).contains(&fc1) {
        return Err(FilterError::NyquistLimitExceeded(fc1));
    }

    if !(0.0..=0.5).contains(&fc2) {
        return Err(FilterError::NyquistLimitExceeded(fc2));
    }

    if fc1 >= fc2 {
        return Err(FilterError::InvalidBand(fc1, fc2));
    }

    let low_pass: Vec<f64> = generate_low_pass(num_taps, fc1, window_function.clone())?;
    let high_pass: Vec<f64> = generate_high_pass(num_taps, fc2, window_function.clone())?;

    Ok(high_pass
        .iter()
        .zip(low_pass.iter())
        .map(|(high, low)| high + low)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    // --- apply_fir ---

    #[test]
    fn apply_fir_zero_taps_error() {
        let result = apply_fir(&[1.0, 2.0, 3.0], &[]);
        assert!(matches!(result, Err(FilterError::ZeroTaps)));
    }

    #[test]
    fn apply_fir_identity() {
        // A single tap of [1.0] is an identity filter.
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let output = apply_fir(&input, &[1.0]).unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn apply_fir_known_response() {
        // A two-tap averaging filter [0.5, 0.5] on [2.0, 4.0, 6.0]:
        //   n=0: 0.5*2.0                = 1.0
        //   n=1: 0.5*4.0 + 0.5*2.0     = 3.0
        //   n=2: 0.5*6.0 + 0.5*4.0     = 5.0
        let input = vec![2.0, 4.0, 6.0];
        let taps = vec![0.5, 0.5];
        let output = apply_fir(&input, &taps).unwrap();
        assert_eq!(output.len(), input.len());
        assert!(approx_eq(output[0], 1.0, EPSILON));
        assert!(approx_eq(output[1], 3.0, EPSILON));
        assert!(approx_eq(output[2], 5.0, EPSILON));
    }

    #[test]
    fn apply_fir_output_length() {
        let input = vec![1.0; 10];
        let taps = vec![0.5, 0.25, 0.25];
        let output = apply_fir(&input, &taps).unwrap();
        assert_eq!(output.len(), input.len());
    }

    // --- generate_low_pass ---

    #[test]
    fn generate_low_pass_zero_taps_error() {
        let result = generate_low_pass(0, 0.1, WindowFunction::Hann);
        assert!(matches!(result, Err(FilterError::ZeroTaps)));
    }

    #[test]
    fn generate_low_pass_nyquist_error() {
        let result = generate_low_pass(31, 0.6, WindowFunction::Hann);
        assert!(matches!(result, Err(FilterError::NyquistLimitExceeded(_))));
    }

    #[test]
    fn generate_low_pass_tap_count() {
        let num_taps = 31;
        let taps = generate_low_pass(num_taps, 0.1, WindowFunction::Hann).unwrap();
        assert_eq!(taps.len(), num_taps);
    }

    #[test]
    fn generate_low_pass_dc_gain() {
        // After normalization the taps must sum to 1.0 (unity DC gain).
        let taps = generate_low_pass(31, 0.1, WindowFunction::Hann).unwrap();
        let sum: f64 = taps.iter().sum();
        assert!(approx_eq(sum, 1.0, 1e-10));
    }

    #[test]
    fn generate_low_pass_dc_gain_hamming() {
        let taps = generate_low_pass(31, 0.2, WindowFunction::Hamming).unwrap();
        let sum: f64 = taps.iter().sum();
        assert!(approx_eq(sum, 1.0, 1e-10));
    }

    #[test]
    fn generate_low_pass_dc_gain_blackman() {
        let taps = generate_low_pass(31, 0.15, WindowFunction::Blackman).unwrap();
        let sum: f64 = taps.iter().sum();
        assert!(approx_eq(sum, 1.0, 1e-10));
    }

    // --- generate_high_pass ---

    #[test]
    fn generate_high_pass_zero_taps_error() {
        let result = generate_high_pass(0, 0.1, WindowFunction::Hann);
        assert!(matches!(result, Err(FilterError::ZeroTaps)));
    }

    #[test]
    fn generate_high_pass_nyquist_error() {
        let result = generate_high_pass(31, 0.6, WindowFunction::Hann);
        assert!(matches!(result, Err(FilterError::NyquistLimitExceeded(_))));
    }

    #[test]
    fn generate_high_pass_tap_count() {
        let num_taps = 31;
        let taps = generate_high_pass(num_taps, 0.3, WindowFunction::Hann).unwrap();
        assert_eq!(taps.len(), num_taps);
    }

    // --- generate_band_pass ---

    #[test]
    fn generate_band_pass_invalid_band() {
        // fc1 == fc2 must be rejected
        let result = generate_band_pass(31, 0.2, 0.2, WindowFunction::Hann);
        assert!(matches!(result, Err(FilterError::InvalidBand(_, _))));
    }

    #[test]
    fn generate_band_pass_fc1_greater_than_fc2() {
        let result = generate_band_pass(31, 0.3, 0.1, WindowFunction::Hann);
        assert!(matches!(result, Err(FilterError::InvalidBand(_, _))));
    }

    #[test]
    fn generate_band_pass_nyquist_error_fc1() {
        let result = generate_band_pass(31, 0.6, 0.8, WindowFunction::Hann);
        assert!(matches!(result, Err(FilterError::NyquistLimitExceeded(_))));
    }

    #[test]
    fn generate_band_pass_nyquist_error_fc2() {
        let result = generate_band_pass(31, 0.1, 0.8, WindowFunction::Hann);
        assert!(matches!(result, Err(FilterError::NyquistLimitExceeded(_))));
    }

    #[test]
    fn generate_band_pass_tap_count() {
        let num_taps = 31;
        let taps = generate_band_pass(num_taps, 0.1, 0.3, WindowFunction::Hann).unwrap();
        assert_eq!(taps.len(), num_taps);
    }

    // --- generate_notch ---

    #[test]
    fn generate_notch_invalid_band() {
        let result = generate_notch(31, 0.2, 0.2, WindowFunction::Hann);
        assert!(matches!(result, Err(FilterError::InvalidBand(_, _))));
    }

    #[test]
    fn generate_notch_fc1_greater_than_fc2() {
        let result = generate_notch(31, 0.4, 0.1, WindowFunction::Hann);
        assert!(matches!(result, Err(FilterError::InvalidBand(_, _))));
    }

    #[test]
    fn generate_notch_nyquist_error_fc1() {
        let result = generate_notch(31, 0.6, 0.8, WindowFunction::Hann);
        assert!(matches!(result, Err(FilterError::NyquistLimitExceeded(_))));
    }

    #[test]
    fn generate_notch_nyquist_error_fc2() {
        let result = generate_notch(31, 0.1, 0.8, WindowFunction::Hann);
        assert!(matches!(result, Err(FilterError::NyquistLimitExceeded(_))));
    }

    #[test]
    fn generate_notch_tap_count() {
        let num_taps = 31;
        let taps = generate_notch(num_taps, 0.1, 0.3, WindowFunction::Hann).unwrap();
        assert_eq!(taps.len(), num_taps);
    }
}
