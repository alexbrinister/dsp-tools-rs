use byteorder::{LittleEndian, ReadBytesExt};
use clap::{Parser, Subcommand, ValueEnum};
use num_complex::Complex;
use std::fmt;
use std::io::Write;

mod filter;
mod ft;
mod signal;
mod window;

#[derive(Parser)]
#[command(name = "DSP CLI")]
#[command(version = "0.1.0")]
#[command(about = "a DSP toolkit CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Signal {
        #[arg(short, long)]
        function: SignalFunction,

        #[arg(short, long)]
        sample_rate: f64,

        #[arg(short = 'w', long)]
        frequency: f64,

        #[arg(short, long)]
        duration: f64,
    },

    Ft {
        #[arg(short, long)]
        transform_type: TransformType,

        #[arg(short, long, default_value_t = OutputFormat::Magnitude)]
        output_format: OutputFormat,
    },

    Window {
        #[arg(short = 'z', long)]
        window_function: WindowFunction,
    },

    Filter {
        #[arg(short, long)]
        filter_type: FilterType,

        #[arg(short = 'l', long)]
        cutoff_low: f64,

        #[arg(short = 'a', long, default_value_t = 0.0)]
        cutoff_high: f64,

        #[arg(short, long)]
        sample_rate: f64,

        #[arg(short = 'n', long)]
        taps: usize,

        #[arg(short = 'z', long, default_value_t = WindowFunction::Blackman)]
        window_function: WindowFunction,
    },
}

#[derive(ValueEnum, Clone, Debug)]
enum SignalFunction {
    Square,
    Sine,
    Cosine,
    NoiseG,
    NoiseW,
}

#[derive(ValueEnum, Clone, Debug)]
enum TransformType {
    Dft,
    Fft,
}

#[derive(ValueEnum, Clone, Debug)]
enum OutputFormat {
    Magnitude,
    Power,
    Phase,
    Complex,
}

#[derive(ValueEnum, Clone, Debug)]
enum WindowFunction {
    Hann,
    Hamming,
    Blackman,
}

#[derive(ValueEnum, Clone, Debug)]
enum FilterType {
    LowPass,
    HighPass,
    BandPass,
    Notch,
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

impl fmt::Display for WindowFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            WindowFunction::Hann => "hann",
            WindowFunction::Hamming => "hamming",
            WindowFunction::Blackman => "blackman",
        };

        write!(f, "{}", name)
    }
}

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Command::Signal {
            function,
            sample_rate,
            frequency,
            duration,
        } => {
            eprintln!("signal command invoked!");
            eprintln!("args:");
            eprintln!("{:>4}function: {:?}", "", function);
            eprintln!("{:>4}sample rate: {:?}", "", sample_rate);
            eprintln!("{:>4}frequency: {:?}", "", frequency);
            eprintln!("{:>4}duration: {:?}", "", duration);

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

            write_to_stdout(&out);
        }

        Command::Ft {
            transform_type,
            output_format,
        } => {
            eprintln!("ft command invoked!");
            eprintln!("args:");
            eprintln!("{:>4}transform type: {:?}", "", transform_type);
            eprintln!("{:>4}output format: {:?}", "", output_format);

            let mut data = read_from_stdin();

            let out = match transform_type {
                TransformType::Dft => ft::dft(&data),
                TransformType::Fft => {
                    // Guard against non-power-of-two inputs by zero-padding the data
                    pad_to_pow2(&mut data);
                    ft::fft(&data)
                }
            };

            let formatted_out = format_output(&out, output_format.clone());
            write_to_stdout(&formatted_out);
        }

        Command::Window { window_function } => {
            eprintln!("window command invoked!");
            eprintln!("args:");
            eprintln!("{:>4}window function: {}", "", window_function);

            let mut data = read_from_stdin();

            match window_function {
                WindowFunction::Hann => window::apply_hann(&mut data),
                WindowFunction::Hamming => window::apply_hamming(&mut data),
                WindowFunction::Blackman => window::apply_blackman(&mut data),
            };

            write_to_stdout(&data);
        }

        Command::Filter {
            filter_type,
            cutoff_low,
            cutoff_high,
            sample_rate,
            taps,
            window_function,
        } => {
            if *sample_rate <= 0.0 {
                eprintln!("error: sample rate must be greater than 0");
                std::process::exit(2);
            }

            if *cutoff_low <= 0.0 {
                eprintln!("error: low cutoff frequency must be greater than 0");
                std::process::exit(2);
            }

            if taps.is_multiple_of(2) {
                eprintln!("error: # of taps must be odd");
                std::process::exit(2);
            }

            if matches!(*filter_type, FilterType::BandPass | FilterType::Notch) {
                if *cutoff_high <= 0.0 {
                    eprintln!(
                        "error: must specify a high cutoff frequency > 0.0 for band-pass and notch filters."
                    );
                    std::process::exit(2);
                }
                if *cutoff_low >= *cutoff_high {
                    eprintln!(
                        "error: low cutoff frequency must be strictly less than high cutoff frequency."
                    );
                    std::process::exit(2);
                }
            }

            eprintln!("filter command invoked!");
            eprintln!("args:");
            eprintln!("{:>4}cutoff frequency (low): {:?}", "", cutoff_low);

            if *cutoff_high > 0.0 {
                eprintln!("{:>4}cutoff frequency (high): {:?}", "", cutoff_high);
            }

            eprintln!("{:>4}sample rate: {:?}", "", sample_rate);
            eprintln!("{:>4}window function: {}", "", window_function);

            let fc1 = cutoff_low / sample_rate;
            if fc1 > 0.5 {
                eprintln!("error: low cutoff frequency is past Nyquist limit");
                std::process::exit(2);
            }

            let fc2 = cutoff_high / sample_rate;
            if fc2 > 0.5 && matches!(*filter_type, FilterType::BandPass | FilterType::Notch) {
                eprintln!("error: high cutoff frequency is past Nyquist limit");
                std::process::exit(2);
            }

            let input = read_from_stdin();

            let num_taps = match filter_type {
                FilterType::LowPass => {
                    filter::generate_low_pass(*taps, fc1, window_function.clone())
                }
                FilterType::HighPass => {
                    filter::generate_high_pass(*taps, fc1, window_function.clone())
                }
                FilterType::BandPass => {
                    filter::generate_band_pass(*taps, fc1, fc2, window_function.clone())
                }
                FilterType::Notch => {
                    filter::generate_notch(*taps, fc1, fc2, window_function.clone())
                }
            };

            let output = filter::apply_fir(&input, &num_taps);
            write_to_stdout(&output);
        }
    }
}

fn write_to_stdout(data: &[f64]) {
    let mut stdout = std::io::stdout().lock();

    for val in data {
        stdout
            .write_all(&val.to_le_bytes())
            .expect("Failed to write wave to stdout");
    }

    stdout.flush().expect("Failed to flush stdout");
}

fn read_from_stdin() -> Vec<f64> {
    // take in no more than 65536 values
    const UPPER_INPUT_BOUND: usize = 1 << 16;
    let mut data = Vec::with_capacity(UPPER_INPUT_BOUND);
    let mut stdin = std::io::stdin().lock();
    let mut count: usize = 0;

    while count < UPPER_INPUT_BOUND {
        let result = stdin.read_f64::<LittleEndian>();

        match result {
            // if we get EOF, then leave the loop
            // if any other error, we panic
            Err(error) => match error.kind() {
                std::io::ErrorKind::UnexpectedEof => break,
                _ => panic!("Couldn't read from stdin"),
            },

            // if we get a good result, add it to the output vector
            Ok(result) => {
                data.push(result);
                count += 1;
            }
        }
    }

    data
}

fn format_output(complex_data: &[Complex<f64>], output_format: OutputFormat) -> Vec<f64> {
    match output_format {
        OutputFormat::Magnitude => complex_data.iter().map(|c| c.norm()).collect(),
        OutputFormat::Power => complex_data.iter().map(|c| c.norm_sqr()).collect(),
        OutputFormat::Phase => complex_data.iter().map(|c| c.arg()).collect(),
        OutputFormat::Complex => complex_data.iter().flat_map(|c| [c.re, c.im]).collect(),
    }
}

fn pad_to_pow2(data: &mut Vec<f64>) {
    let n: usize = data.len();

    // not a power of two
    // find new size (next power of two) and resize data in-place
    if n > 0 && !n.is_power_of_two() {
        let new_byte_count = n.next_power_of_two();
        data.resize(new_byte_count, 0.0);
    }
}
