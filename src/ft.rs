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

pub fn fft(input: &[f64]) -> Vec<Complex<f64>> {
    let n: usize = input.len();

    // Guard against non-power-of-two inputs
    assert!(
        n.is_power_of_two(),
        "FFT input length must be a power of 2! Received: {}",
        n
    );

    // base case
    if n <= 1 {
        return vec![Complex::new(input[0], 0.0)];
    }

    // new vecs for odd and even
    let evens: Vec<f64> = input.iter().step_by(2).copied().collect();
    let odds: Vec<f64> = input[1..].iter().step_by(2).copied().collect();

    // compute even and odd ffts
    let even_fft = fft(&evens); // even
    let odd_fft = fft(&odds); // odd

    let mut output: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n];

    let n64 = n as f64;
    for k in 0..n / 2 {
        let k64 = k as f64;
        let angle: f64 = -2.0 * std::f64::consts::PI * k64 / n64;
        let wk: Complex<f64> = Complex::new(angle.cos(), angle.sin());
        let odd_term = wk * odd_fft[k];
        output[k] = even_fft[k] + odd_term;
        output[k + n / 2] = even_fft[k] - odd_term;
    }

    output
}
