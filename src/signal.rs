fn get_sample_count(duration: f64, sample_rate: f64) -> u64 {
    (duration * sample_rate).ceil() as u64
}

pub fn generate_sine(frequency: f64, duration: f64, sample_rate: f64) -> Vec<f64> {
    let sample_count = get_sample_count(duration, sample_rate) as usize;

    (0..sample_count)
        .map(|n| {
            let time = (n as f64) / sample_rate;
            let phase: f64 = 2_f64 * std::f64::consts::PI * frequency * time;
            phase.sin()
        })
        .collect()
}
