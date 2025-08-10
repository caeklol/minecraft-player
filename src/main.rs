extern crate openblas_src;
#[macro_use]
extern crate ndarray;

use std::{collections::HashMap, path::PathBuf, time::Instant};

use anyhow::{Error, anyhow};
use clap::Parser;
use inquire::Select;
use minecraft_player::{algebra::{self}, assets::{self, AudioResourceLocation, FetchBehavior}, audio::{self, Sound}, mojang::{self, AssetIndex, Version}};
use ndarray::Axis;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

#[derive(clap::Args, Debug)]
#[group(required = false, multiple = false)]
struct BehaviorGroup {
    #[arg(short, long, help = "refetch all assets and replace all local files")]
    refetch: bool,
    
    #[arg(short, long, help = "do not check against asset index and force use of local files")]
    local: bool
}

#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    #[arg(short, long, help = "version from which to fetch assets from")]
    target_version: Option<String>,

    #[clap(flatten)]
    behavior: BehaviorGroup,

    #[arg(short, long, help = "assets directory (default: ./data)", default_value = "./data")]
    assets: PathBuf,

    #[arg(short, long, help = "input audio file")]
    input: PathBuf,

    #[arg(short, long, help = "output mcfunctions")]
    output: PathBuf,
}


async fn find_version(target_version: &Option<String>) -> Result<Version, Error> {
    println!("fetching version manifest...");
    let manifest = mojang::fetch_version_manifest().await?;
    println!("fetched version manifest");

    match target_version {
        Some(version_str) => {
            let possible_versions = manifest.versions.iter().filter(|v| v.id.contains(version_str)).collect::<Vec<&Version>>();
            let exact_version = manifest.versions.iter().find(|v| v.id == *version_str);

            if let Some(exact_version) = exact_version {
                return Ok(exact_version.clone())
            }

            if possible_versions.is_empty() {
                println!("could not find a matching version to `{}`", version_str);
            } else if possible_versions.len() > 1 {
                println!("multiple matching versions to `{}`", version_str);
                return Ok(Select::new("what version will you use?", possible_versions).prompt().unwrap().clone())
            } else {
                return Ok(possible_versions[0].clone())
            }
        },
        None => {},
    };

    return Ok(Select::new("what version will you use?", manifest.versions).prompt().unwrap())
}

async fn fetch_predictable_sounds(
    version: &Option<String>,
    assets: &PathBuf,
    behavior: &FetchBehavior
) -> Result<Vec<(String, Sound)>, Error> {
    let version = find_version(version).await?;
    
    let asset_index = match behavior {
        FetchBehavior::FetchIfMissing | FetchBehavior::Refetch => {
            println!("fetching asset index...");
            mojang::fetch_asset_index(&version).await?
        },
        FetchBehavior::CacheOnly => AssetIndex {
            objects: HashMap::new()
        },
    };

    println!("fetching sound definitions...");
    let definitions = assets::fetch_sound_definitions(&assets, &version, &behavior, &asset_index).await?;

    println!("fetching sounds...");
    let sounds = assets::fetch_sounds(&assets, &version, &behavior, &asset_index).await?;

    let mut result = HashMap::new();

    let sound_path = PathBuf::from("minecraft/sounds");

    for (identifier, def) in definitions {
        if def.sounds.len() == 1 {
            if let Some(sound) = def.sounds.get(0) {
                let resource = match sound {
                    AudioResourceLocation::Partial(s) => Some((PathBuf::from(s), 1.0, 1.0)),
                    AudioResourceLocation::Full(resource_location) => {
                        match &resource_location.resource_type {
                            Some(resource_type) if resource_type != "sound" => None,
                            _ => Some((
                                    resource_location.name.clone(),
                                    resource_location.pitch.unwrap_or(1.0),
                                    resource_location.volume.unwrap_or(1.0)
                            )),
                        }
                    },
                };

                if let Some((sound_name, pitch, volume)) = resource {
                    let sound_path = sound_path.join(&sound_name).with_extension("ogg");
                    let sound = sounds.iter().find(|(path, _)| *path == &sound_path);
                    if let Some(sound) = sound {
                        let mut sound = sound.1.clone();
                        result.insert(identifier, sound.adjust_pitch(pitch).adjust_volume(volume).resample(48000).clone());
                    }
                }
            }
        }
    }

    Ok(result.into_iter().collect::<Vec<(String, Sound)>>()) 
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let args = Args::parse();

    let behavior = match (args.behavior.refetch, args.behavior.local) {
        (true, false) => FetchBehavior::Refetch,
        (false, true) => FetchBehavior::CacheOnly,
        (false, false) => FetchBehavior::FetchIfMissing,
        _ => unimplemented!("impossible")
    };

    let predictable_sounds = fetch_predictable_sounds(&args.target_version, &args.assets, &behavior).await?;

    println!("found {} predictable sounds", predictable_sounds.len());

    let processor = audio::Processor::new();

    let sounds = audio::permute_with_pitch(predictable_sounds, 256)
        .into_par_iter()
        .map(|(id, mut sound)| (id, sound.mel(&processor).clone()))
        .collect::<Vec<((String, f32), Sound)>>();

    let sound_ids = sounds.iter().map(|s| s.0.clone()).collect::<Vec<(String, f32)>>();

    let sound_bins = sounds.iter().map(|s| s.1.samples.clone()).collect::<Vec<Vec<f32>>>();

    let mut sound_bins = algebra::matrix_from_vecs(sound_bins)?
        .reversed_axes();

    drop(sounds);

    println!("reading target file");
    let mut reader = hound::WavReader::open(&args.input)?;

    if reader.spec().channels > 1 {
        eprintln!("!! ERROR: stereo audio is not supported! please convert your input file into mono:");
        let input_filename: &str = args.input.file_stem().unwrap().to_str().unwrap();
        println!("help: if you have ffmpeg installed:");
        println!("help: ffmpeg -i {}.wav -ac 1 {}.mono.wav", input_filename, input_filename);
        return Err(anyhow!("input was stereo"));
    }

    let samples = reader.samples::<i16>()
        .map(|r| r.expect("found empty sample"))
        .collect::<Vec<i16>>()
        .iter()
        .map(|i| *i as f32)
        .collect::<Vec<f32>>();

    let sample_rate: usize = reader.spec().sample_rate.try_into().unwrap();

    // 20 minecraft ticks per second, (1s/20t) = 0.05s/t = 50ms/t
    let samples_per_tick = audio::time_as_samples!(50, sample_rate); 
    println!("sample rate of {}Hz, splitting input into {} sized chunks", sample_rate, samples_per_tick);

    let chunks = samples.chunks_exact(samples_per_tick.try_into().unwrap()).collect::<Vec<&[f32]>>()
        .into_par_iter()
        .map(|samples| Sound {
            samples: samples.to_vec(),
            sample_rate
        })
        .map(|mut sound| sound.mel(&processor).clone())
        .map(|sound| sound.samples)
        .collect::<Vec<Vec<f32>>>();

    drop(samples);

    let start = Instant::now();
    let mut chunks = algebra::matrix_from_vecs(chunks)?
        .reversed_axes();

    println!("chunks: {:?}", &chunks.dim());
    println!("bins: {:?}", &sound_bins.dim());

    algebra::normalize_to_global(&mut chunks);
    algebra::normalize_to_global(&mut sound_bins);

    println!("running NNLS...");
    let mut approximation = algebra::pgd_nnls(&chunks, &sound_bins, 128, 1e-6);

    algebra::normalize_to_global(&mut approximation);
    algebra::apply_epsilon(&mut approximation, 1e-5);

    drop(chunks);

    println!("done! elapsed: {}ms", start.elapsed().as_millis());

    println!("saving to datapack...");

    let mut writer = hound::WavWriter::create("output.wav", hound::WavSpec {
        channels: 1,
        sample_rate: 48000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    }).unwrap();

    for (index, amplitudes) in approximation.axis_iter(Axis(1)).enumerate() {
        let mut amplitudes: Vec<(usize, (&f32, &(String, f32)))> = amplitudes.iter().zip(&sound_ids).enumerate().collect();
        amplitudes.sort_by(|a, b| b.1.0.partial_cmp(a.1.0).unwrap());

        let amplitudes = &amplitudes[0..64];
        let mut output = String::new();
        output.push_str("stopsound @a[tag=!nomusic] record\n");
        let mut output_sample = vec![0.0; 2400];

        for (i, (amplitude, (name, pitch))) in amplitudes {
            output.push_str(&format!("playsound {} record @a 0 -60 0 {:.5} {:.5} \n", name, amplitude, pitch));

            let mut sound = Sound {
                samples: sound_bins.column(*i).to_vec(),
                sample_rate: 48000
            };

            sound.adjust_volume(**amplitude);

            for (j, sample) in sound.samples.iter().enumerate() {
                output_sample[j] += sample;
            }
        }

        for sample in output_sample {
            writer.write_sample(sample).unwrap();
        }

        output.push_str(&format!("schedule function audio:_/{} 1t append\n", index + 1));
        tokio::fs::write(args.output.join("function/_/").join(index.to_string()).with_extension("mcfunction"), output).await?;
    }

    writer.finalize().unwrap();

    return Ok(());
}
