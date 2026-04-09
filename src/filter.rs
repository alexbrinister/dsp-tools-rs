use crate::window;
use crate::window::WindowFunction;
use thiserror::Error;

/// An error that can occur during filter generation or application.
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

/// Generates the coefficients for a low-pass FIR filter.
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

/// Generates the coefficients for a high-pass FIR filter.
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

/// Generates the coefficients for a band-pass FIR filter.
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

/// Generates the coefficients for a notch (band-stop) FIR filter.
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
