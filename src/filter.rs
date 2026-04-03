pub fn apply_fir(input: &[f64], taps: &[f64]) -> Vec<f64> {
    (0..input.len())
        .map(|n| {
            let k_max = n.min(taps.len() - 1);
            (0..=k_max).map(|k| taps[k] * input[n - k]).sum()
        })
        .collect()
}
