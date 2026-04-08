#[derive(Clone, Debug, PartialEq)]
pub enum WindowFunction {
    Hann,
    Hamming,
    Blackman,
}

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
