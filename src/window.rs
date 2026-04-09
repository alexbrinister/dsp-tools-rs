/// Supported window functions.
#[derive(Clone, Debug, PartialEq)]
pub enum WindowFunction {
    /// The Hann window.
    Hann,
    /// The Hamming window.
    Hamming,
    /// The Blackman window.
    Blackman,
}

/// Applies the Hann window function to the provided data in-place.
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

/// Applies the Hamming window function to the provided data in-place.
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

/// Applies the Blackman window function to the provided data in-place.
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
