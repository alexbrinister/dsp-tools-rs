use rand::prelude::*;
use rand_distr::{Distribution, Normal};

fn get_sample_count(duration: f64, sample_rate: f64) -> usize {
    (duration * sample_rate).ceil() as usize
}

pub fn generate_sine(frequency: f64, duration: f64, sample_rate: f64) -> Vec<f64> {
    let sample_count = get_sample_count(duration, sample_rate);

    (0..sample_count)
        .map(|n| {
            let time = (n as f64) / sample_rate;
            let phase: f64 = 2.0 * std::f64::consts::PI * frequency * time;
            phase.sin()
        })
        .collect()
}

pub fn generate_cosine(frequency: f64, duration: f64, sample_rate: f64) -> Vec<f64> {
    let sample_count = get_sample_count(duration, sample_rate);

    (0..sample_count)
        .map(|n| {
            let time = (n as f64) / sample_rate;
            let phase: f64 = 2.0 * std::f64::consts::PI * frequency * time;
            phase.cos()
        })
        .collect()
}

pub fn generate_square(frequency: f64, duration: f64, sample_rate: f64) -> Vec<f64> {
    let sample_count = get_sample_count(duration, sample_rate);
    let period = 1.0 / frequency;

    (0..sample_count)
        .map(|n| {
            let time = (n as f64) / sample_rate;
            if time % period < (period / 2.0) {
                1.0
            } else {
                -1.0
            }
        })
        .collect()
}

pub fn generate_white_noise(duration: f64, sample_rate: f64) -> Vec<f64> {
    let sample_count = get_sample_count(duration, sample_rate);
    let mut rng = rand::rng();

    (0..sample_count)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect()
}

pub fn generate_gaussian_noise(duration: f64, sample_rate: f64) -> Vec<f64> {
    let sample_count = get_sample_count(duration, sample_rate);
    let mut rng = rand::rng();

    // mean: 0.0, std. dev 0.33
    // center values around 0 and ensure they are all within 3 std. dev of the mean
    // ensures that most values are within -1.0..1.0
    let normal = Normal::new(0.0, 0.33).unwrap();

    (0..sample_count).map(|_| normal.sample(&mut rng)).collect()
}
