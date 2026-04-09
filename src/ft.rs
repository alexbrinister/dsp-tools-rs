//! Fourier transforms for frequency-domain analysis.
//!
//! Two implementations are provided:
//!
//! - [`dft`](crate::ft::dft) — Naïve O(N²) Discrete Fourier Transform. Works on any input
//!   length. Suitable for short signals or correctness verification.
//! - [`fft`](crate::ft::fft) — O(N log N) Cooley-Tukey radix-2 FFT. Automatically zero-pads
//!   non-power-of-two inputs to the next power of two before processing.
//!
//! Both functions return `Vec<Complex<f64>>` where element `k` represents the
//! complex amplitude at frequency `k · fs / N` Hz (with `fs` the sample rate
//! and `N` the transform length).
//!
//! ## Choosing DFT vs FFT
//!
//! | | [`dft`](crate::ft::dft) | [`fft`](crate::ft::fft) |
//! |-|-------------------------|-------------------------|
//! | Complexity | O(N²) | O(N log N) |
//! | Input length | Any | Any (zero-padded to power of two) |
//! | Use case | Short signals, reference | Production use |
//!
//! ## Interpreting the Output
//!
//! The output vector has length `N` (or the padded length for [`fft`](crate::ft::fft)).
//! Bins `0` through `N/2` correspond to frequencies `0` through `fs/2`
//! (the Nyquist frequency). Bins `N/2 + 1` through `N - 1` are the negative
//! frequencies and mirror the positive side for real-valued input.
//!
//! Common post-processing operations on each [`num_complex::Complex`] bin:
//!
//! | Quantity | Expression |
//! |----------|------------|
//! | Magnitude | `bin.norm()` |
//! | Power | `bin.norm_sqr()` |
//! | Phase | `bin.arg()` |

use num_complex::Complex;

/// Computes the Discrete Fourier Transform (DFT) of a real-valued signal.
///
/// The DFT is defined as:
///
/// ```text
/// X[k] = Σ_{n=0}^{N-1}  x[n] · e^{−j 2π k n / N}
/// ```
///
/// where `N = input.len()` and `k` is the frequency bin index. This is a
/// direct O(N²) implementation. For large signals, prefer [`fft`].
///
/// # Arguments
///
/// * `input` — Real-valued time-domain samples.
///
/// # Returns
///
/// A [`Vec<Complex<f64>>`] of length `N`. The magnitude of bin `k` indicates
/// how strongly frequency `k · fs / N` is present in the signal.
///
/// # Examples
///
/// A constant (DC) signal concentrates all energy at bin 0:
///
/// ```
/// use dsp_tools::ft::dft;
///
/// let input = vec![1.0f64; 8];
/// let output = dft(&input);
///
/// // Bin 0 equals the sum of all samples
/// assert!((output[0].re - 8.0).abs() < 1e-10);
/// assert!(output[0].im.abs() < 1e-10);
///
/// // All other bins are zero for a DC signal
/// for bin in &output[1..] {
///     assert!(bin.norm() < 1e-10);
/// }
/// ```
///
/// A unit cosine at bin `k=1` over `N=8` samples produces magnitude `N/2`
/// at bins 1 and `N-1`, and zero everywhere else:
///
/// ```
/// use dsp_tools::ft::dft;
/// use std::f64::consts::PI;
///
/// let n = 8usize;
/// let input: Vec<f64> = (0..n)
///     .map(|i| (2.0 * PI * i as f64 / n as f64).cos())
///     .collect();
/// let output = dft(&input);
///
/// assert!((output[1].norm() - 4.0).abs() < 1e-10);
/// assert!((output[n - 1].norm() - 4.0).abs() < 1e-10);
/// for k in 2..(n - 1) {
///     assert!(output[k].norm() < 1e-10);
/// }
/// ```
pub fn dft(input: &[f64]) -> Vec<Complex<f64>> {
    let length: usize = input.len();
    let len64 = length as f64;

    (0..length)
        .map(|k| {
            let k64 = k as f64;

            (0..length)
                .map(|n| {
                    let n64 = n as f64;
                    let angle: f64 = -2.0 * std::f64::consts::PI * k64 * n64 / len64;
                    input[n] * Complex::new(angle.cos(), angle.sin())
                })
                .sum()
        })
        .collect()
}

fn bit_reverse_permutation(data: &mut [Complex<f64>]) {
    let n = data.len();
    let bits = n.trailing_zeros();

    for i in 0..n {
        let rev = i.reverse_bits() >> (usize::BITS - bits);

        if i < rev {
            data.swap(i, rev);
        }
    }
}

/// Computes the Fast Fourier Transform (FFT) of a real-valued signal.
///
/// Uses the Cooley-Tukey radix-2 decimation-in-time algorithm. The input is
/// converted to complex form, then an in-place butterfly computation produces
/// the frequency-domain output. The mathematical result is identical to
/// [`dft`]:
///
/// ```text
/// X[k] = Σ_{n=0}^{N-1}  x[n] · e^{−j 2π k n / N}
/// ```
///
/// ## Zero-padding
///
/// If `input.len()` is not a power of two, the internal buffer is extended
/// with zeros to `input.len().next_power_of_two()` before the transform runs.
/// The returned vector has this padded length, not the original input length.
/// Zero-padding increases frequency resolution in the output but does not add
/// information — it interpolates between the "true" DFT bins.
///
/// # Arguments
///
/// * `input` — Real-valued time-domain samples. Any length is accepted.
///
/// # Returns
///
/// A [`Vec<Complex<f64>>`] whose length is `input.len()` rounded up to the
/// next power of two (unchanged if already a power of two).
///
/// # Examples
///
/// Basic FFT of a DC signal:
///
/// ```
/// use dsp_tools::ft::fft;
///
/// let input = vec![1.0f64; 8];
/// let output = fft(&input);
///
/// assert_eq!(output.len(), 8);
/// assert!((output[0].re - 8.0).abs() < 1e-10);
/// for bin in &output[1..] {
///     assert!(bin.norm() < 1e-10);
/// }
/// ```
///
/// Non-power-of-two input is zero-padded automatically:
///
/// ```
/// use dsp_tools::ft::fft;
///
/// let input = vec![1.0f64; 5]; // 5 is not a power of two → padded to 8
/// let output = fft(&input);
///
/// assert_eq!(output.len(), 8);
/// // DC bin equals the sum of the original (non-zero-padded) samples
/// assert!((output[0].re - 5.0).abs() < 1e-10);
/// ```
///
/// FFT and DFT produce equivalent results for power-of-two inputs:
///
/// ```
/// use dsp_tools::ft::{dft, fft};
/// use std::f64::consts::PI;
///
/// let input: Vec<f64> = (0..16)
///     .map(|i| (2.0 * PI * 3.0 * i as f64 / 16.0).sin())
///     .collect();
///
/// let dft_out = dft(&input);
/// let fft_out = fft(&input);
///
/// for (d, f) in dft_out.iter().zip(fft_out.iter()) {
///     assert!((d.re - f.re).abs() < 1e-9);
///     assert!((d.im - f.im).abs() < 1e-9);
/// }
/// ```
pub fn fft(input: &[f64]) -> Vec<Complex<f64>> {
    let n = input.len();

    // prepare input data
    let mut output: Vec<Complex<f64>> = input
        .iter()
        .map(|&element| Complex::new(element, 0.0))
        .collect();

    if n > 0 && !n.is_power_of_two() {
        let new_byte_count = n.next_power_of_two();
        output.resize(new_byte_count, Complex::new(0.0, 0.0));
    }

    // use the (possibly padded) output length so all butterfly passes run
    let padded_n = output.len();

    // run bit reverse permutation on the complex input data to line
    // up even and odd samples correctly
    bit_reverse_permutation(&mut output);

    let mut step: usize = 2;

    // loop 1: sub-fft, from 2 to N, inclusive
    while step <= padded_n {
        let step_f64 = step as f64;
        let half_step = step >> 1;

        // loop 2: perform butterfly on both halves of the sub-fft
        // create a slice, starting at index i of size `step`, here representing the region of data the sub-fft
        // operates on
        for j in 0..half_step {
            let j_f64 = j as f64;
            let angle: f64 = -2.0 * std::f64::consts::PI * j_f64 / step_f64;
            let wk: Complex<f64> = Complex::new(angle.cos(), angle.sin());

            // loop 3: process each sub-fft, moving over the array in `step` increments
            for subfft_data in output.chunks_mut(step) {
                let odd_term = wk * subfft_data[j + half_step];

                // critical... save the old even fft before we mutate it
                // to use in the odd calculation
                let old_even = subfft_data[j];

                subfft_data[j] += odd_term;
                subfft_data[j + half_step] = old_even - odd_term;
            }
        }

        step <<= 1;
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    // --- DFT tests ---

    #[test]
    fn dft_output_length() {
        let input = vec![1.0, 0.0, -1.0, 0.0, 1.0];
        let output = dft(&input);
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn dft_zero_input() {
        let input = vec![0.0f64; 8];
        let output = dft(&input);
        for bin in &output {
            assert!(approx_eq(bin.re, 0.0, EPSILON));
            assert!(approx_eq(bin.im, 0.0, EPSILON));
        }
    }

    #[test]
    fn dft_dc_input() {
        // Constant input of 1.0: bin 0 should equal N, all others 0
        let n = 8;
        let input = vec![1.0f64; n];
        let output = dft(&input);
        assert!(approx_eq(output[0].re, n as f64, EPSILON));
        assert!(approx_eq(output[0].im, 0.0, EPSILON));
        for bin in &output[1..] {
            assert!(approx_eq(bin.re, 0.0, EPSILON));
            assert!(approx_eq(bin.im, 0.0, EPSILON));
        }
    }

    #[test]
    fn dft_single_freq() {
        // A pure cosine at frequency k=1 over N=8 samples should produce
        // magnitude N/2 at bins 1 and N-1, and ~0 elsewhere.
        let n = 8usize;
        let input: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos())
            .collect();
        let output = dft(&input);
        let half = (n / 2) as f64;
        assert!(approx_eq(output[1].norm(), half, 1e-10));
        assert!(approx_eq(output[n - 1].norm(), half, 1e-10));
        // All other bins should be near zero
        for k in 2..(n - 1) {
            assert!(approx_eq(output[k].norm(), 0.0, 1e-10));
        }
    }

    // --- FFT tests ---

    #[test]
    fn fft_zero_input() {
        let input = vec![0.0f64; 8];
        let output = fft(&input);
        for bin in &output {
            assert!(approx_eq(bin.re, 0.0, EPSILON));
            assert!(approx_eq(bin.im, 0.0, EPSILON));
        }
    }

    #[test]
    fn fft_dc_input() {
        let n = 8;
        let input = vec![1.0f64; n];
        let output = fft(&input);
        assert!(approx_eq(output[0].re, n as f64, EPSILON));
        assert!(approx_eq(output[0].im, 0.0, EPSILON));
        for bin in &output[1..] {
            assert!(approx_eq(bin.norm(), 0.0, EPSILON));
        }
    }

    #[test]
    fn fft_matches_dft() {
        // For any power-of-2 input the FFT must produce the same result as the DFT.
        let input: Vec<f64> = (0..16)
            .map(|i| (2.0 * std::f64::consts::PI * 3.0 * i as f64 / 16.0).sin())
            .collect();
        let dft_out = dft(&input);
        let fft_out = fft(&input);
        assert_eq!(dft_out.len(), fft_out.len());
        for (d, f) in dft_out.iter().zip(fft_out.iter()) {
            assert!(
                approx_eq(d.re, f.re, 1e-9),
                "re mismatch: {} vs {}",
                d.re,
                f.re
            );
            assert!(
                approx_eq(d.im, f.im, 1e-9),
                "im mismatch: {} vs {}",
                d.im,
                f.im
            );
        }
    }

    #[test]
    fn fft_single_freq() {
        let n = 8usize;
        let input: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos())
            .collect();
        let output = fft(&input);
        let half = (n / 2) as f64;
        assert!(approx_eq(output[1].norm(), half, 1e-10));
        assert!(approx_eq(output[n - 1].norm(), half, 1e-10));
        for k in 2..(n - 1) {
            assert!(approx_eq(output[k].norm(), 0.0, 1e-10));
        }
    }

    // --- FFT zero-padding tests ---

    #[test]
    fn fft_non_pow2_output_length() {
        // A 5-sample input must be padded to the next power of two (8).
        let input = vec![1.0f64; 5];
        let output = fft(&input);
        assert_eq!(output.len(), 8);
    }

    #[test]
    fn fft_non_pow2_zero_input() {
        // Zero-padding a zero signal must still produce zero output.
        let input = vec![0.0f64; 5];
        let output = fft(&input);
        assert_eq!(output.len(), 8);
        for bin in &output {
            assert!(approx_eq(bin.re, 0.0, EPSILON));
            assert!(approx_eq(bin.im, 0.0, EPSILON));
        }
    }

    #[test]
    fn fft_non_pow2_dc_bin() {
        // DC bin (k=0) must equal the sum of the original (non-padded) samples.
        // For 5 ones padded to 8 the sum is still 5.0.
        let input = vec![1.0f64; 5];
        let output = fft(&input);
        assert!(approx_eq(output[0].re, 5.0, EPSILON));
        assert!(approx_eq(output[0].im, 0.0, EPSILON));
    }

    #[test]
    fn fft_non_pow2_matches_dft_of_padded() {
        // The FFT of a non-power-of-two input must equal the DFT of the
        // same input explicitly zero-padded to the next power of two.
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let padded: Vec<f64> = {
            let mut v = input.clone();
            v.resize(8, 0.0);
            v
        };
        let fft_out = fft(&input);
        let dft_out = dft(&padded);
        assert_eq!(fft_out.len(), dft_out.len());
        for (f, d) in fft_out.iter().zip(dft_out.iter()) {
            assert!(approx_eq(f.re, d.re, 1e-9), "re mismatch: {} vs {}", f.re, d.re);
            assert!(approx_eq(f.im, d.im, 1e-9), "im mismatch: {} vs {}", f.im, d.im);
        }
    }

    #[test]
    fn fft_pow2_input_length_unchanged() {
        // Power-of-two inputs must not be resized.
        let input = vec![1.0f64; 8];
        let output = fft(&input);
        assert_eq!(output.len(), 8);
    }
}
