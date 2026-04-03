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
    let adjusted_num_taps = if num_taps.is_multiple_of(2) {
        num_taps + 1
    } else {
        num_taps
    };

    let center_point = (adjusted_num_taps - 1) / 2;
    let center_point_f64 = center_point as f64;

    let mut output: Vec<f64> = (0..adjusted_num_taps)
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
