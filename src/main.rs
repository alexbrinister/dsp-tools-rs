use anyhow::{Context, anyhow};
use byteorder::{LittleEndian, ReadBytesExt};
use clap::{Parser, Subcommand, ValueEnum};
use num_complex::Complex;
use std::fmt;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::Path;

use dsp_tools::{filter, ft, signal, window};

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
        window_function: CliWindowFunction,
    },

    Filter {
        #[arg(short, long, value_parser = parse_finite_gt0_f64)]
        sample_rate: f64,

        #[arg(short = 'n', long, value_parser = parse_gt0_odd)]
        taps: usize,

        #[arg(short = 'z', long, default_value_t = CliWindowFunction::Blackman)]
        window_function: CliWindowFunction,

        #[command(subcommand)]
        filter_type: FilterCommand,
    },
}

#[derive(clap::Args, Debug, Clone)]
pub struct SingleCutoff {
    #[arg(short, long, value_parser = parse_finite_gt0_f64)]
    pub cutoff: f64,
}

#[derive(clap::Args, Debug, Clone)]
pub struct DualCutoff {
    #[arg(short = 'l', long, value_parser = parse_finite_gt0_f64)]
    pub cutoff_low: f64,

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

#[derive(Subcommand)]
enum FilterCommand {
    LowPass(SingleCutoff),

    HighPass(SingleCutoff),

    BandPass(DualCutoff),

    Notch(DualCutoff),

    Derivative,

    Matched {
        #[arg(long)]
        template_file: String,
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
pub enum CliWindowFunction {
    Hann,
    Hamming,
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

            write_to_stdout(&out)?;
        }

        Command::Ft {
            transform_type,
            output_format,
        } => {
            eprintln!("ft command invoked!");
            eprintln!("args:");
            eprintln!("{:>4}transform type: {:?}", "", transform_type);
            eprintln!("{:>4}output format: {:?}", "", output_format);

            let mut data = read_from_stdin()?;

            let out = match transform_type {
                TransformType::Dft => ft::dft(&data),
                TransformType::Fft => {
                    // Guard against non-power-of-two inputs by zero-padding the data
                    pad_to_pow2(&mut data);
                    ft::fft(&data)
                }
            };

            let formatted_out = format_output(&out, output_format.clone());
            write_to_stdout(&formatted_out)?;
        }

        Command::Window { window_function } => {
            eprintln!("window command invoked!");
            eprintln!("args:");
            eprintln!("{:>4}window function: {}", "", window_function);

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

fn write_to_stdout(data: &[f64]) -> Result<(), anyhow::Error> {
    let mut stdout = std::io::stdout().lock();

    for val in data {
        if let Err(e) = stdout.write_all(&val.to_le_bytes()) {
            if e.kind() == std::io::ErrorKind::BrokenPipe {
                return Ok(());
            }

            return Err(anyhow!("Failed to write wave to stdout"));
        }
    }

    stdout.flush().context("Failed to flush stdout")
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

fn pad_to_pow2(data: &mut Vec<f64>) {
    let n: usize = data.len();

    // not a power of two
    // find new size (next power of two) and resize data in-place
    if n > 0 && !n.is_power_of_two() {
        let new_byte_count = n.next_power_of_two();
        data.resize(new_byte_count, 0.0);
    }
}
