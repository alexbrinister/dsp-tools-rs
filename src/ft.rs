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

fn bit_reverse_permutation(data: &mut [Complex<f64>]) {
    let n = data.len();
    let bits = n.trailing_zeros();

    for i in 0..n {
        let rev = i.reverse_bits() >> (usize::BITS - bits);

        if i < rev {
            data.swap(i, rev);
        }
    }
}

pub fn fft(input: &[f64]) -> Vec<Complex<f64>> {
    let n = input.len();

    // prepare input data
    let mut output: Vec<Complex<f64>> = input
        .iter()
        .map(|&element| Complex::new(element, 0.0))
        .collect();

    // run bit reverse permutation on the complex input data to line
    // up even and odd samples correctly
    bit_reverse_permutation(&mut output);

    let mut step: usize = 2;

    // loop 1: sub-fft, from 2 to N, inclusive
    while step <= n {
        let step_f64 = step as f64;
        let half_step = step >> 1;

        // loop 2: perform butterfly on both halves of the sub-fft
        // create a slice, starting at index i of size `step`, here representing the region of data the sub-fft
        // operates on
        for j in 0..half_step {
            let j_f64 = j as f64;
            let angle: f64 = -2.0 * std::f64::consts::PI * j_f64 / step_f64;
            let wk: Complex<f64> = Complex::new(angle.cos(), angle.sin());

            // loop 3: process each sub-fft, moving over the array in `step` increments
            for subfft_data in output.chunks_mut(step) {
                let odd_term = wk * subfft_data[j + half_step];

                // critical... save the old even fft before we mutate it
                // to use in the odd calculation
                let old_even = subfft_data[j];

                subfft_data[j] += odd_term;
                subfft_data[j + half_step] = old_even - odd_term;
            }
        }

        step <<= 1;
    }

    output
}
