use std::{collections::HashMap, path::PathBuf};

use anyhow::{Error, anyhow};
use clap::Parser;
use inquire::Select;
use minecraft_player::{assets::{self, AudioResourceLocation, FetchBehavior, SoundAsset}, audio::{self, adjust_pitch}, mojang::{self, AssetIndex, Version}};

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
) -> Result<HashMap<String, Vec<i16>>, Error> {
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
                    AudioResourceLocation::Partial(s) => Some((PathBuf::from(s), 1.0)),
                    AudioResourceLocation::Full(resource_location) => {
                        if let Some(resource_type) = &resource_location.resource_type {
                            if resource_type == "sound" {
                                Some((resource_location.name.clone(), resource_location.pitch.unwrap_or(1.0)))
                            } else {
                                None
                            }
                        } else {
                            Some((resource_location.name.clone(), resource_location.pitch.unwrap_or(1.0)))
                        }
                    },
                };

                if let Some((sound_name, pitch)) = resource {
                    let sound = sounds.iter().find(|(path, _)| path == &sound_path.join(&sound_name));
                    if let Some(sound) = sound {
                        let sound = &sound.1;
                        let pitch_normalized = audio::adjust_pitch(&sound.samples, pitch);
                        let samples_per_tick = (sound.sample_rate * 50) / 1000;
                        if pitch_normalized.len() >= samples_per_tick {
                            result.insert(identifier, pitch_normalized[0..samples_per_tick].to_vec());
                        }
                    }
                }
            }
        }
    }

    Ok(result) 
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

    let sounds = fetch_predictable_sounds(&args.target_version, &args.assets, &behavior).await?;

    println!("reading target file");
    let mut reader = hound::WavReader::open(&args.input)?;

    if reader.spec().channels > 1 {
        eprintln!("!! ERROR: stereo audio is not supported! please convert your input file into mono:");
        let input_filename: &str = args.input.file_stem().unwrap().to_str().unwrap();
        println!("help: if you have ffmpeg installed:");
        println!("help: ffmpeg -i {}.wav -ac 1 {}.mono.wav", input_filename, input_filename);
        return Err(anyhow!("input was stereo"));
    }

    let samples = reader.samples::<i16>().map(|r| r.expect("found empty sample")).collect::<Vec<i16>>();

    let sample_rate = reader.spec().sample_rate;

    // 20 minecraft ticks per second, (1s/20t) = 0.05s/t = 50ms/t
    let samples_per_tick = audio::time_as_samples!(50, sample_rate); 
    println!("sample rate of {}Hz, splitting input into {} sized chunks", sample_rate, samples_per_tick);

    // your computer DESERVES to panic if `u32` > `usize` and you are running this. that thing should have left a long time ago. picked
    // up its little feet and ran
    let chunks = samples.chunks_exact(samples_per_tick.try_into().unwrap()).collect::<Vec<&[i16]>>();

    return Ok(());
}
