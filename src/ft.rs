use num_complex::Complex;

pub fn dft(input: &[f64]) -> Vec<Complex<f64>> {
    let length: usize = input.len();
    let len64 = length as f64;

    (0..length)
        .map(|k| {
            let k64 = k as f64;

            (0..length)
                .map(|n| {
                    let n64 = n as f64;
                    let angle: f64 = -2.0 * std::f64::consts::PI * k64 * n64 / len64;
                    input[n] * Complex::new(angle.cos(), angle.sin())
                })
                .sum()
        })
        .collect()
}
