use crate::WindowFunction;
use crate::window;

pub fn apply_fir(input: &[f64], taps: &[f64]) -> Vec<f64> {
    (0..input.len())
        .map(|n| {
            let k_max = n.min(taps.len() - 1);
            (0..=k_max).map(|k| taps[k] * input[n - k]).sum()
        })
        .collect()
}

pub fn generate_low_pass(num_taps: usize, fc: f64, window_function: WindowFunction) -> Vec<f64> {
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

    output
}

pub fn generate_high_pass(num_taps: usize, fc: f64, window_function: WindowFunction) -> Vec<f64> {
    let mut low_pass: Vec<f64> = generate_low_pass(num_taps, fc, window_function);
    let center_point: usize = (low_pass.len() - 1) / 2;

    low_pass.iter_mut().for_each(|tap| *tap *= -1.0);
    low_pass[center_point] += 1.0;
    low_pass
}

pub fn generate_band_pass(
    num_taps: usize,
    fc1: f64,
    fc2: f64,
    window_function: WindowFunction,
) -> Vec<f64> {
    let low_pass_fc1: Vec<f64> = generate_low_pass(num_taps, fc1, window_function.clone());
    let low_pass_fc2: Vec<f64> = generate_low_pass(num_taps, fc2, window_function.clone());
    low_pass_fc2
        .iter()
        .zip(low_pass_fc1.iter())
        .map(|(high, low)| high - low)
        .collect()
}

pub fn generate_notch(
    num_taps: usize,
    fc1: f64,
    fc2: f64,
    window_function: WindowFunction,
) -> Vec<f64> {
    let low_pass: Vec<f64> = generate_low_pass(num_taps, fc1, window_function.clone());
    let high_pass: Vec<f64> = generate_high_pass(num_taps, fc2, window_function.clone());
    high_pass
        .iter()
        .zip(low_pass.iter())
        .map(|(high, low)| high + low)
        .collect()
}
