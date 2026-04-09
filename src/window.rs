//! Window functions for reducing spectral leakage in Fourier analysis.
//!
//! When a finite-length signal is transformed with the DFT or FFT, the sharp
//! edges at the start and end of the buffer introduce high-frequency artefacts
//! known as *spectral leakage*. Multiplying the buffer element-wise by a
//! smooth window that tapers to (or near) zero at both ends significantly
//! reduces this effect, at the cost of slightly widening the main spectral
//! peak.
//!
//! All functions operate **in-place** on a `&mut [f64]` slice and do nothing
//! for slices of length 0 or 1.
//!
//! ## Choosing a Window
//!
//! | Window | Endpoint value | Sidelobe attenuation | Main-lobe width |
//! |--------|---------------|----------------------|-----------------|
//! | [`apply_hann`](crate::window::apply_hann) | 0.0 | ~31 dB | Moderate |
//! | [`apply_hamming`](crate::window::apply_hamming) | 0.08 | ~41 dB | Moderate |
//! | [`apply_blackman`](crate::window::apply_blackman) | 0.0 | ~57 dB | Wide |
//!
//! A wider main lobe makes closely spaced frequencies harder to resolve;
//! higher sidelobe attenuation makes weak signals near a strong peak easier
//! to detect. Blackman is the best default when the dynamic range between
//! frequency components is large.
//!
//! ## Example: window a signal before FFT
//!
//! ```
//! use dsp_tools::signal::generate_sine;
//! use dsp_tools::window::apply_blackman;
//! use dsp_tools::ft::fft;
//!
//! let mut samples = generate_sine(440.0, 0.1, 8000.0);
//! apply_blackman(&mut samples);
//! let spectrum = fft(&samples);
//! assert!(!spectrum.is_empty());
//! ```

/// The window function variant to use for in-place application or FIR filter
/// design.
///
/// Passed to [`apply_hann`], [`apply_hamming`], [`apply_blackman`], or the
/// filter-design functions in [`crate::filter`].
///
/// # Example
///
/// ```
/// use dsp_tools::window::{WindowFunction, apply_hann, apply_hamming, apply_blackman};
///
/// let mut data = vec![1.0f64; 64];
///
/// fn apply(wf: WindowFunction, data: &mut Vec<f64>) {
///     match wf {
///         WindowFunction::Hann     => apply_hann(data),
///         WindowFunction::Hamming  => apply_hamming(data),
///         WindowFunction::Blackman => apply_blackman(data),
///     }
/// }
///
/// apply(WindowFunction::Blackman, &mut data);
/// // Endpoints are zero for the Blackman window
/// assert!(data[0].abs() < 1e-10);
/// assert!(data[63].abs() < 1e-10);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub enum WindowFunction {
    /// The Hann window.
    Hann,
    /// The Hamming window.
    Hamming,
    /// The Blackman window.
    Blackman,
}

/// Applies the Hann window to `data` in-place.
///
/// Multiplies each sample by the Hann weight:
///
/// ```text
/// w(n) = 0.5 · (1 − cos(2π n / (N − 1)))
/// ```
///
/// Both endpoints equal **0.0** and the centre of an odd-length buffer
/// reaches a peak of **1.0**. The Hann window is a good general-purpose
/// choice with moderate sidelobe attenuation (~31 dB).
///
/// Slices of length 0 or 1 are returned unchanged.
///
/// # Arguments
///
/// * `data` — Signal samples to window in-place.
///
/// # Examples
///
/// ```
/// use dsp_tools::window::apply_hann;
///
/// let mut data = vec![1.0f64; 8];
/// apply_hann(&mut data);
///
/// // Length is unchanged
/// assert_eq!(data.len(), 8);
///
/// // Both endpoints are zero
/// assert!(data[0].abs() < 1e-10);
/// assert!(data[7].abs() < 1e-10);
/// ```
///
/// Applying to all-ones gives the raw window coefficients. For a 5-sample
/// window `phase_step = 2π/4`, so `w(1) = 0.5·(1 − cos(π/2)) = 0.5`:
///
/// ```
/// use dsp_tools::window::apply_hann;
///
/// let mut data = vec![1.0f64; 5];
/// apply_hann(&mut data);
/// assert!((data[1] - 0.5).abs() < 1e-10);
/// ```
///
/// A single-sample buffer is not modified:
///
/// ```
/// use dsp_tools::window::apply_hann;
///
/// let mut data = vec![42.0f64];
/// apply_hann(&mut data);
/// assert_eq!(data[0], 42.0);
/// ```
pub fn apply_hann(data: &mut [f64]) {
    if data.len() <= 1 {
        return;
    }

    let length_f64 = data.len() as f64;
    let phase_step = (2.0 * std::f64::consts::PI) / (length_f64 - 1.0);

    data.iter_mut().enumerate().for_each(|(n, sample)| {
        let n_f64 = n as f64;
        let angle = phase_step * n_f64;
        let window_multiplier = 0.5 * (1.0 - angle.cos());
        *sample *= window_multiplier;
    });
}

/// Applies the Hamming window to `data` in-place.
///
/// Multiplies each sample by the Hamming weight:
///
/// ```text
/// w(n) = 0.54 − 0.46 · cos(2π n / (N − 1))
/// ```
///
/// Unlike the Hann window, the Hamming window does **not** reach zero at its
/// endpoints: `w(0) = w(N−1) = 0.08`. This slight discontinuity is traded
/// for higher sidelobe attenuation (~41 dB), making it better at revealing
/// weak spectral components close to a strong one.
///
/// The centre of an odd-length buffer reaches a peak of **1.0**.
/// Slices of length 0 or 1 are returned unchanged.
///
/// # Arguments
///
/// * `data` — Signal samples to window in-place.
///
/// # Examples
///
/// ```
/// use dsp_tools::window::apply_hamming;
///
/// let mut data = vec![1.0f64; 8];
/// apply_hamming(&mut data);
///
/// assert_eq!(data.len(), 8);
///
/// // Endpoints equal 0.54 − 0.46 = 0.08
/// assert!((data[0] - 0.08).abs() < 1e-10);
/// assert!((data[7] - 0.08).abs() < 1e-10);
/// ```
///
/// A single-sample buffer is not modified:
///
/// ```
/// use dsp_tools::window::apply_hamming;
///
/// let mut data = vec![7.0f64];
/// apply_hamming(&mut data);
/// assert_eq!(data[0], 7.0);
/// ```
pub fn apply_hamming(data: &mut [f64]) {
    if data.len() <= 1 {
        return;
    }

    let length_f64 = data.len() as f64;
    let phase_step = (2.0 * std::f64::consts::PI) / (length_f64 - 1.0);

    data.iter_mut().enumerate().for_each(|(n, sample)| {
        let n_f64 = n as f64;
        let angle = phase_step * n_f64;
        let window_multiplier = 0.54 - 0.46 * angle.cos();
        *sample *= window_multiplier;
    });
}

/// Applies the Blackman window to `data` in-place.
///
/// Multiplies each sample by the Blackman weight:
///
/// ```text
/// w(n) = 0.42 − 0.5 · cos(2π n / (N − 1)) + 0.08 · cos(4π n / (N − 1))
/// ```
///
/// The additional cosine term gives the Blackman window the **highest
/// sidelobe attenuation** (~57 dB) of the three windows in this module.
/// This makes it the best choice when a strong signal must not obscure a
/// weaker nearby frequency, at the cost of a slightly wider main lobe.
///
/// Both endpoints equal **0.0** and the centre of an odd-length buffer
/// reaches a peak of **1.0**. Slices of length 0 or 1 are returned unchanged.
///
/// # Arguments
///
/// * `data` — Signal samples to window in-place.
///
/// # Examples
///
/// ```
/// use dsp_tools::window::apply_blackman;
///
/// let mut data = vec![1.0f64; 8];
/// apply_blackman(&mut data);
///
/// assert_eq!(data.len(), 8);
///
/// // Endpoints are zero: 0.42 − 0.5 + 0.08 = 0
/// assert!(data[0].abs() < 1e-10);
/// assert!(data[7].abs() < 1e-10);
/// ```
///
/// The centre of an odd-length all-ones buffer reaches 1.0:
///
/// ```
/// use dsp_tools::window::apply_blackman;
///
/// let mut data = vec![1.0f64; 9];
/// apply_blackman(&mut data);
/// assert!((data[4] - 1.0).abs() < 1e-10);
/// ```
///
/// A single-sample buffer is not modified:
///
/// ```
/// use dsp_tools::window::apply_blackman;
///
/// let mut data = vec![3.0f64];
/// apply_blackman(&mut data);
/// assert_eq!(data[0], 3.0);
/// ```
pub fn apply_blackman(data: &mut [f64]) {
    if data.len() <= 1 {
        return;
    }

    let length_f64 = data.len() as f64;
    let phase_step = (2.0 * std::f64::consts::PI) / (length_f64 - 1.0);

    data.iter_mut().enumerate().for_each(|(n, sample)| {
        let n_f64 = n as f64;
        let angle = phase_step * n_f64;
        let window_multiplier = 0.42 - 0.5 * angle.cos() + 0.08 * (2.0 * angle).cos();
        *sample *= window_multiplier;
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    // --- apply_hann ---

    #[test]
    fn apply_hann_length_preserved() {
        let mut data = vec![1.0f64; 16];
        apply_hann(&mut data);
        assert_eq!(data.len(), 16);
    }

    #[test]
    fn apply_hann_single_sample_unchanged() {
        let mut data = vec![42.0f64];
        apply_hann(&mut data);
        assert!(approx_eq(data[0], 42.0, EPSILON));
    }

    #[test]
    fn apply_hann_endpoints_zero() {
        // Hann window: w(0) = 0, w(N-1) = 0
        let n = 16;
        let mut data = vec![1.0f64; n];
        apply_hann(&mut data);
        assert!(approx_eq(data[0], 0.0, EPSILON));
        assert!(approx_eq(data[n - 1], 0.0, EPSILON));
    }

    #[test]
    fn apply_hann_center_peak() {
        // For an odd-length window the center sample has maximum weight (1.0).
        let n = 9;
        let mut data = vec![1.0f64; n];
        apply_hann(&mut data);
        let center = n / 2;
        assert!(approx_eq(data[center], 1.0, 1e-10));
    }

    #[test]
    fn apply_hann_known_values() {
        // Apply to all-ones; spot-check n=1 of a 5-sample window.
        // phase_step = 2π/4; w(1) = 0.5*(1 - cos(2π/4)) = 0.5*(1-0) = 0.5
        let mut data = vec![1.0f64; 5];
        apply_hann(&mut data);
        assert!(approx_eq(data[1], 0.5, 1e-10));
    }

    // --- apply_hamming ---

    #[test]
    fn apply_hamming_length_preserved() {
        let mut data = vec![1.0f64; 16];
        apply_hamming(&mut data);
        assert_eq!(data.len(), 16);
    }

    #[test]
    fn apply_hamming_single_sample_unchanged() {
        let mut data = vec![42.0f64];
        apply_hamming(&mut data);
        assert!(approx_eq(data[0], 42.0, EPSILON));
    }

    #[test]
    fn apply_hamming_endpoints() {
        // Hamming window: w(0) = w(N-1) = 0.54 - 0.46 = 0.08
        let n = 16;
        let mut data = vec![1.0f64; n];
        apply_hamming(&mut data);
        assert!(approx_eq(data[0], 0.08, 1e-10));
        assert!(approx_eq(data[n - 1], 0.08, 1e-10));
    }

    #[test]
    fn apply_hamming_center_peak() {
        // Center of an odd-length window: w = 0.54 + 0.46 = 1.0
        let n = 9;
        let mut data = vec![1.0f64; n];
        apply_hamming(&mut data);
        let center = n / 2;
        assert!(approx_eq(data[center], 1.0, 1e-10));
    }

    // --- apply_blackman ---

    #[test]
    fn apply_blackman_length_preserved() {
        let mut data = vec![1.0f64; 16];
        apply_blackman(&mut data);
        assert_eq!(data.len(), 16);
    }

    #[test]
    fn apply_blackman_single_sample_unchanged() {
        let mut data = vec![42.0f64];
        apply_blackman(&mut data);
        assert!(approx_eq(data[0], 42.0, EPSILON));
    }

    #[test]
    fn apply_blackman_endpoints_zero() {
        // Blackman window: w(0) = 0.42 - 0.5 + 0.08 = 0.0
        let n = 16;
        let mut data = vec![1.0f64; n];
        apply_blackman(&mut data);
        assert!(approx_eq(data[0], 0.0, 1e-10));
        assert!(approx_eq(data[n - 1], 0.0, 1e-10));
    }

    #[test]
    fn apply_blackman_center_peak() {
        // Center of an odd-length window: w = 0.42 + 0.5 + 0.08 = 1.0
        let n = 9;
        let mut data = vec![1.0f64; n];
        apply_blackman(&mut data);
        let center = n / 2;
        assert!(approx_eq(data[center], 1.0, 1e-10));
    }
}
