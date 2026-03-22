use clap::{Parser, Subcommand, ValueEnum};
use std::fmt;
use std::io::Write;

mod signal;

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

            match function {
                SignalFunction::Sine => {
                    let out = signal::generate_sine(*frequency, *duration, *sample_rate);
                    write_to_stdout(&out);
                }

                SignalFunction::Square => {
                    let out = signal::generate_square(*frequency, *duration, *sample_rate);
                    write_to_stdout(&out);
                }

                _ => {
                    println!("under construction!");
                }
            }
        }

        Command::Ft {
            transform_type,
            output_format,
        } => {
            println!("ft command invoked!");
            println!("args:");
            println!("{:>4}transform type: {:?}", "", transform_type);
            println!("{:>4}output format: {:?}", "", output_format);
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
