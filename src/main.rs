use anyhow::{Context, anyhow};
use byteorder::{LittleEndian, ReadBytesExt};
use clap::{Parser, Subcommand, ValueEnum};
use num_complex::Complex;
use std::fmt;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::Path;

use dsp_tools::{filter, ft, signal, window};

/// The main CLI application struct.
#[derive(Parser)]
#[command(name = "DSP CLI")]
#[command(version = "0.1.0")]
#[command(about = "a DSP toolkit CLI", long_about = None)]
struct Cli {
    /// The command to run.
    #[command(subcommand)]
    command: Command,
}

/// Available commands in the DSP toolkit.
#[derive(Subcommand)]
enum Command {
    /// Generates a signal.
    Signal {
        /// The type of signal to generate.
        #[arg(short, long)]
        function: SignalFunction,

        /// The sample rate of the signal in Hz.
        #[arg(short, long)]
        sample_rate: f64,

        /// The frequency of the signal in Hz (for sine, cosine, square).
        #[arg(short = 'w', long)]
        frequency: f64,

        /// The duration of the signal in seconds.
        #[arg(short, long)]
        duration: f64,
    },

    /// Performs a Fourier Transform on the input signal.
    Ft {
        /// The type of Fourier transform to apply (DFT or FFT).
        #[arg(short, long)]
        transform_type: TransformType,

        /// The output format for the complex transform results.
        #[arg(short, long, default_value_t = OutputFormat::Magnitude)]
        output_format: OutputFormat,
    },

    /// Applies a window function to the input signal.
    Window {
        /// The type of window function to apply.
        #[arg(short = 'z', long)]
        window_function: CliWindowFunction,
    },

    /// Applies a digital filter to the input signal.
    Filter {
        /// The sample rate of the signal in Hz.
        #[arg(short, long, value_parser = parse_finite_gt0_f64)]
        sample_rate: f64,

        /// The number of taps for the FIR filter (must be odd).
        #[arg(short = 'n', long, value_parser = parse_gt0_odd)]
        taps: usize,

        /// The window function to use for filter design.
        #[arg(short = 'z', long, default_value_t = CliWindowFunction::Blackman)]
        window_function: CliWindowFunction,

        /// The type of filter to apply.
        #[command(subcommand)]
        filter_type: FilterCommand,
    },
}

/// Arguments for filters that require a single cutoff frequency.
#[derive(clap::Args, Debug, Clone)]
pub struct SingleCutoff {
    /// The cutoff frequency in Hz.
    #[arg(short, long, value_parser = parse_finite_gt0_f64)]
    pub cutoff: f64,
}

/// Arguments for filters that require a low and high cutoff frequency.
#[derive(clap::Args, Debug, Clone)]
pub struct DualCutoff {
    /// The lower cutoff frequency in Hz.
    #[arg(short = 'l', long, value_parser = parse_finite_gt0_f64)]
    pub cutoff_low: f64,

    /// The upper cutoff frequency in Hz.
    #[arg(short = 'a', long, value_parser = parse_finite_gt0_f64)]
    pub cutoff_high: f64,
}

impl DualCutoff {
    fn validate(&self) -> Result<(), anyhow::Error> {
        if self.cutoff_low >= self.cutoff_high {
            return Err(anyhow!(
                "low cutoff frequency must be strictly less than high cutoff frequency."
                    .to_string()
            ));
        }

        Ok(())
    }
}

/// Available filter types.
#[derive(Subcommand)]
enum FilterCommand {
    /// Low-pass filter.
    LowPass(SingleCutoff),

    /// High-pass filter.
    HighPass(SingleCutoff),

    /// Band-pass filter.
    BandPass(DualCutoff),

    /// Notch (band-stop) filter.
    Notch(DualCutoff),

    /// Derivative filter.
    Derivative,

    /// Matched filter using a template file.
    Matched {
        /// Path to the template file.
        #[arg(long)]
        template_file: String,
    },
}

/// The type of signal to generate.
#[derive(ValueEnum, Clone, Debug)]
enum SignalFunction {
    /// Square wave.
    Square,
    /// Sine wave.
    Sine,
    /// Cosine wave.
    Cosine,
    /// Gaussian noise.
    NoiseG,
    /// White noise.
    NoiseW,
}

/// The type of Fourier Transform to perform.
#[derive(ValueEnum, Clone, Debug)]
enum TransformType {
    /// Discrete Fourier Transform.
    Dft,
    /// Fast Fourier Transform.
    Fft,
}

/// Output format for Fourier Transform results.
#[derive(ValueEnum, Clone, Debug)]
enum OutputFormat {
    /// Magnitude of complex numbers.
    Magnitude,
    /// Power (squared magnitude) of complex numbers.
    Power,
    /// Phase of complex numbers.
    Phase,
    /// Real and imaginary parts interleaved.
    Complex,
}

/// Window functions available in the CLI.
#[derive(ValueEnum, Clone, Debug)]
pub enum CliWindowFunction {
    /// Hann window.
    Hann,
    /// Hamming window.
    Hamming,
    /// Blackman window.
    Blackman,
}

impl fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            OutputFormat::Magnitude => "magnitude",
            OutputFormat::Power => "power",
            OutputFormat::Phase => "phase",
            OutputFormat::Complex => "complex",
        };

        write!(f, "{}", name)
    }
}

impl fmt::Display for CliWindowFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            CliWindowFunction::Hann => "hann",
            CliWindowFunction::Hamming => "hamming",
            CliWindowFunction::Blackman => "blackman",
        };

        write!(f, "{}", name)
    }
}

impl From<CliWindowFunction> for window::WindowFunction {
    fn from(cli_enum: CliWindowFunction) -> Self {
        match cli_enum {
            CliWindowFunction::Hann => window::WindowFunction::Hann,
            CliWindowFunction::Hamming => window::WindowFunction::Hamming,
            CliWindowFunction::Blackman => window::WindowFunction::Blackman,
        }
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {}", e);
        std::process::exit(2);
    }
}

fn run() -> Result<(), anyhow::Error> {
    let cli = Cli::parse();

    match &cli.command {
        Command::Signal {
            function,
            sample_rate,
            frequency,
            duration,
        } => {
            let out = match function {
                SignalFunction::Sine => signal::generate_sine(*frequency, *duration, *sample_rate),

                SignalFunction::Cosine => {
                    signal::generate_cosine(*frequency, *duration, *sample_rate)
                }

                SignalFunction::Square => {
                    signal::generate_square(*frequency, *duration, *sample_rate)
                }

                SignalFunction::NoiseW => signal::generate_white_noise(*duration, *sample_rate),

                SignalFunction::NoiseG => signal::generate_gaussian_noise(*duration, *sample_rate),
            };

            write_to_stdout(&out)?;
        }

        Command::Ft {
            transform_type,
            output_format,
        } => {
            let data = read_from_stdin()?;

            let out = match transform_type {
                TransformType::Dft => ft::dft(&data),
                TransformType::Fft => ft::fft(&data),
            };

            let formatted_out = format_output(&out, output_format.clone());
            write_to_stdout(&formatted_out)?;
        }

        Command::Window { window_function } => {
            let mut data = read_from_stdin()?;

            match window_function {
                CliWindowFunction::Hann => window::apply_hann(&mut data),
                CliWindowFunction::Hamming => window::apply_hamming(&mut data),
                CliWindowFunction::Blackman => window::apply_blackman(&mut data),
            };

            write_to_stdout(&data)?;
        }

        Command::Filter {
            sample_rate,
            taps,
            window_function,
            filter_type,
        } => {
            let input = read_from_stdin()?;

            let computed_taps = match filter_type {
                FilterCommand::LowPass(args) => {
                    let fc = try_get_frequency_ratio(args.cutoff, *sample_rate)?;
                    filter::generate_low_pass(*taps, fc, window_function.clone().into())?
                }

                FilterCommand::HighPass(args) => {
                    let fc = try_get_frequency_ratio(args.cutoff, *sample_rate)?;
                    filter::generate_high_pass(*taps, fc, window_function.clone().into())?
                }

                FilterCommand::BandPass(args) => {
                    args.validate()?;
                    let fc1 = try_get_frequency_ratio(args.cutoff_low, *sample_rate)?;
                    let fc2 = try_get_frequency_ratio(args.cutoff_high, *sample_rate)?;
                    filter::generate_band_pass(*taps, fc1, fc2, window_function.clone().into())?
                }

                FilterCommand::Notch(args) => {
                    args.validate()?;
                    let fc1 = try_get_frequency_ratio(args.cutoff_low, *sample_rate)?;
                    let fc2 = try_get_frequency_ratio(args.cutoff_high, *sample_rate)?;
                    filter::generate_notch(*taps, fc1, fc2, window_function.clone().into())?
                }

                FilterCommand::Derivative => vec![0.5, 0.0, -0.5],

                FilterCommand::Matched { template_file } => {
                    let template_path = Path::new(&template_file);

                    let file = File::open(template_path).context("cannot access template file")?;

                    let mut template_taps = read_f64_stream(file)?;
                    template_taps.reverse();
                    template_taps
                }
            };

            let output = filter::apply_fir(&input, &computed_taps)?;
            write_to_stdout(&output)?;
        }
    }

    Ok(())
}

fn parse_finite_gt0_f64(arg: &str) -> Result<f64, anyhow::Error> {
    let val: f64 = arg
        .parse()
        .context("value must be a valid number".to_string())?;
    if val.is_finite() && val > 0.0 {
        return Ok(val);
    }

    Err(anyhow!("value out of range 0.0 < x < NaN".to_string()))
}

fn parse_gt0_odd(arg: &str) -> Result<usize, anyhow::Error> {
    let val: usize = arg
        .parse()
        .context("value must be a valid integer".to_string())?;
    if val % 2 == 0 {
        return Err(anyhow!("value must be odd".to_string()));
    }

    Ok(val)
}

fn try_get_frequency_ratio(cutoff: f64, sample_rate: f64) -> Result<f64, anyhow::Error> {
    let fc = cutoff / sample_rate;
    if fc > 0.5 {
        return Err(anyhow!(
            "error: cutoff frequency {0} is past Nyquist limit",
            cutoff
        ));
    }

    Ok(fc)
}

fn write_to_stream<W: Write>(mut writer: W, data: &[f64]) -> Result<(), anyhow::Error> {
    for val in data {
        if let Err(e) = writer.write_all(&val.to_le_bytes()) {
            if e.kind() == std::io::ErrorKind::BrokenPipe {
                return Ok(());
            }

            return Err(anyhow!("Failed to write wave to stream"));
        }
    }

    writer.flush().context("Failed to flush stream")
}

fn write_to_stdout(data: &[f64]) -> Result<(), anyhow::Error> {
    write_to_stream(std::io::stdout().lock(), data)
}

fn read_from_stdin() -> Result<Vec<f64>, anyhow::Error> {
    let stdin = std::io::stdin().lock();
    read_f64_stream(stdin)
}

fn read_f64_stream<R: Read>(reader: R) -> Result<Vec<f64>, anyhow::Error> {
    // take in no more than 65536 values
    const UPPER_INPUT_BOUND: usize = 1 << 16;
    let mut data = Vec::with_capacity(UPPER_INPUT_BOUND);

    let mut buffered_reader = BufReader::new(reader);
    let mut count: usize = 0;

    while count < UPPER_INPUT_BOUND {
        match buffered_reader.read_f64::<LittleEndian>() {
            // if we get a good result, add it to the output vector
            Ok(val) => {
                data.push(val);
                count += 1;
            }

            // if we get EOF, then leave the loop
            // if any other error, we panic
            Err(error) => match error.kind() {
                std::io::ErrorKind::UnexpectedEof => break,
                _ => return Err(anyhow!("I/O error reading stream")),
            },
        }
    }

    Ok(data)
}

fn format_output(complex_data: &[Complex<f64>], output_format: OutputFormat) -> Vec<f64> {
    match output_format {
        OutputFormat::Magnitude => complex_data.iter().map(|c| c.norm()).collect(),
        OutputFormat::Power => complex_data.iter().map(|c| c.norm_sqr()).collect(),
        OutputFormat::Phase => complex_data.iter().map(|c| c.arg()).collect(),
        OutputFormat::Complex => complex_data.iter().flat_map(|c| [c.re, c.im]).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use byteorder::WriteBytesExt;
    use std::io::Cursor;

    const EPSILON: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    // --- parse_finite_gt0_f64 ---

    #[test]
    fn parse_finite_gt0_f64_valid_positive() {
        let result = parse_finite_gt0_f64("1.5").unwrap();
        assert!(approx_eq(result, 1.5, EPSILON));
    }

    #[test]
    fn parse_finite_gt0_f64_zero() {
        assert!(parse_finite_gt0_f64("0.0").is_err());
    }

    #[test]
    fn parse_finite_gt0_f64_negative() {
        assert!(parse_finite_gt0_f64("-1.0").is_err());
    }

    #[test]
    fn parse_finite_gt0_f64_nan() {
        assert!(parse_finite_gt0_f64("NaN").is_err());
    }

    #[test]
    fn parse_finite_gt0_f64_infinity() {
        assert!(parse_finite_gt0_f64("inf").is_err());
    }

    #[test]
    fn parse_finite_gt0_f64_non_numeric() {
        assert!(parse_finite_gt0_f64("abc").is_err());
    }

    // --- parse_gt0_odd ---

    #[test]
    fn parse_gt0_odd_valid_odd() {
        assert_eq!(parse_gt0_odd("31").unwrap(), 31);
    }

    #[test]
    fn parse_gt0_odd_one() {
        assert_eq!(parse_gt0_odd("1").unwrap(), 1);
    }

    #[test]
    fn parse_gt0_odd_even() {
        assert!(parse_gt0_odd("32").is_err());
    }

    #[test]
    fn parse_gt0_odd_zero() {
        assert!(parse_gt0_odd("0").is_err());
    }

    #[test]
    fn parse_gt0_odd_non_numeric() {
        assert!(parse_gt0_odd("abc").is_err());
    }

    // --- try_get_frequency_ratio ---

    #[test]
    fn try_get_frequency_ratio_valid() {
        let fc = try_get_frequency_ratio(1000.0, 8000.0).unwrap();
        assert!(approx_eq(fc, 0.125, EPSILON));
    }

    #[test]
    fn try_get_frequency_ratio_nyquist_exact() {
        // fc == 0.5 is on the boundary and must succeed (check is fc > 0.5)
        let fc = try_get_frequency_ratio(4000.0, 8000.0).unwrap();
        assert!(approx_eq(fc, 0.5, EPSILON));
    }

    #[test]
    fn try_get_frequency_ratio_exceeds_nyquist() {
        assert!(try_get_frequency_ratio(5000.0, 8000.0).is_err());
    }

    // --- format_output ---

    #[test]
    fn format_output_empty_input() {
        let empty: Vec<Complex<f64>> = vec![];
        for fmt in [
            OutputFormat::Magnitude,
            OutputFormat::Power,
            OutputFormat::Phase,
            OutputFormat::Complex,
        ] {
            assert!(format_output(&empty, fmt).is_empty());
        }
    }

    #[test]
    fn format_output_magnitude() {
        let input = vec![Complex::new(3.0, 4.0)];
        let out = format_output(&input, OutputFormat::Magnitude);
        assert_eq!(out.len(), 1);
        assert!(approx_eq(out[0], 5.0, EPSILON)); // sqrt(9+16) = 5
    }

    #[test]
    fn format_output_power() {
        let input = vec![Complex::new(3.0, 4.0)];
        let out = format_output(&input, OutputFormat::Power);
        assert_eq!(out.len(), 1);
        assert!(approx_eq(out[0], 25.0, EPSILON)); // 9+16 = 25
    }

    #[test]
    fn format_output_phase() {
        // A purely imaginary number has phase π/2
        let input = vec![Complex::new(0.0, 1.0)];
        let out = format_output(&input, OutputFormat::Phase);
        assert_eq!(out.len(), 1);
        assert!(approx_eq(out[0], std::f64::consts::FRAC_PI_2, EPSILON));
    }

    #[test]
    fn format_output_complex_interleaved() {
        let input = vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let out = format_output(&input, OutputFormat::Complex);
        assert_eq!(out.len(), 4); // 2× input length
        assert!(approx_eq(out[0], 1.0, EPSILON));
        assert!(approx_eq(out[1], 2.0, EPSILON));
        assert!(approx_eq(out[2], 3.0, EPSILON));
        assert!(approx_eq(out[3], 4.0, EPSILON));
    }

    // --- read_f64_stream ---

    fn make_le_f64_bytes(values: &[f64]) -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::with_capacity(values.len() * 8);
        for &v in values {
            buf.write_f64::<LittleEndian>(v).unwrap();
        }
        buf
    }

    #[test]
    fn read_f64_stream_empty_reader() {
        let cursor = Cursor::new(vec![]);
        let result = read_f64_stream(cursor).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn read_f64_stream_known_values() {
        let values = [1.0f64, -2.5, 0.0, 1234.5678];
        let bytes = make_le_f64_bytes(&values);
        let result = read_f64_stream(Cursor::new(bytes)).unwrap();
        assert_eq!(result.len(), values.len());
        for (a, b) in result.iter().zip(values.iter()) {
            assert!(approx_eq(*a, *b, EPSILON));
        }
    }

    #[test]
    fn read_f64_stream_respects_upper_bound() {
        // Stream has 65537 values but the function must stop at 65536.
        const UPPER_BOUND: usize = 1 << 16;
        let values: Vec<f64> = (0..UPPER_BOUND + 1).map(|i| i as f64).collect();
        let bytes = make_le_f64_bytes(&values);
        let result = read_f64_stream(Cursor::new(bytes)).unwrap();
        assert_eq!(result.len(), UPPER_BOUND);
    }

    // --- DualCutoff::validate ---

    #[test]
    fn dual_cutoff_validate_valid() {
        let dc = DualCutoff {
            cutoff_low: 0.1,
            cutoff_high: 0.3,
        };
        assert!(dc.validate().is_ok());
    }

    #[test]
    fn dual_cutoff_validate_equal() {
        let dc = DualCutoff {
            cutoff_low: 0.2,
            cutoff_high: 0.2,
        };
        assert!(dc.validate().is_err());
    }

    #[test]
    fn dual_cutoff_validate_reversed() {
        let dc = DualCutoff {
            cutoff_low: 0.4,
            cutoff_high: 0.1,
        };
        assert!(dc.validate().is_err());
    }

    // --- write_to_stream ---

    fn read_le_f64s(bytes: &[u8]) -> Vec<f64> {
        read_f64_stream(Cursor::new(bytes.to_vec())).unwrap()
    }

    #[test]
    fn write_to_stream_empty() {
        let mut buf: Vec<u8> = Vec::new();
        write_to_stream(&mut buf, &[]).unwrap();
        assert!(buf.is_empty());
    }

    #[test]
    fn write_to_stream_single_value() {
        let mut buf: Vec<u8> = Vec::new();
        write_to_stream(&mut buf, &[42.0f64]).unwrap();
        assert_eq!(buf.len(), 8); // one f64 = 8 bytes
        let readback = read_le_f64s(&buf);
        assert!(approx_eq(readback[0], 42.0, EPSILON));
    }

    #[test]
    fn write_to_stream_known_values() {
        let values = [1.0f64, -2.5, 0.0, 1234.5678];
        let mut buf: Vec<u8> = Vec::new();
        write_to_stream(&mut buf, &values).unwrap();
        assert_eq!(buf.len(), values.len() * 8);
        let readback = read_le_f64s(&buf);
        for (a, b) in readback.iter().zip(values.iter()) {
            assert!(approx_eq(*a, *b, EPSILON));
        }
    }

    #[test]
    fn write_to_stream_preserves_order() {
        let values: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let mut buf: Vec<u8> = Vec::new();
        write_to_stream(&mut buf, &values).unwrap();
        let readback = read_le_f64s(&buf);
        assert_eq!(readback, values);
    }

    #[test]
    fn write_to_stream_roundtrip_with_read_f64_stream() {
        // Data written by write_to_stream must be recovered exactly by read_f64_stream.
        let values = [0.1f64, -99.9, f64::MAX, f64::MIN_POSITIVE];
        let mut buf: Vec<u8> = Vec::new();
        write_to_stream(&mut buf, &values).unwrap();
        let readback = read_f64_stream(Cursor::new(buf)).unwrap();
        for (a, b) in readback.iter().zip(values.iter()) {
            assert!(approx_eq(*a, *b, EPSILON));
        }
    }
}
