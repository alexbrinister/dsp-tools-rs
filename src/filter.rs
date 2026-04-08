use crate::window;
use crate::window::WindowFunction;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum FilterError {
    #[error("number of taps must be greater than 0")]
    ZeroTaps,

    #[error("cutoff frequency ({0}) is past the Nyquist limit of 0.5")]
    NyquistLimitExceeded(f64),

    #[error("lower cutoff ({0}) must be less than upper cutoff ({1})")]
    InvalidBand(f64, f64),
}

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
