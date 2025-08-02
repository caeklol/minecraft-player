use std::{collections::HashMap, fs, io, path::{Path, PathBuf}, sync::{atomic::{AtomicUsize, Ordering}, Arc}};

use futures::{stream, StreamExt};
use bytes::Bytes;
use clap::Parser;
use futures::stream::FuturesUnordered;
use inquire::Select;
use minecraft_player::assets::{self, Object, Version};
use anyhow::{anyhow, Error};

#[derive(Parser, Debug)]
enum FetchBehavior {
    CacheOnly,
    Refetch,
    FetchIfMissing
}

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

    #[arg(short, long, help = "cache directory (default: ./data)", default_value = "./data")]
    cache: PathBuf,

    #[arg(short, long, help = "input audio file")]
    input: PathBuf,
}

fn visit_dirs(dir: &Path) -> io::Result<Vec<PathBuf>> {
    let mut files = Vec::new();

    if dir.is_dir() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                files.extend(visit_dirs(&path)?);
            } else {
                files.push(path);
            }
        }
    }
    
    Ok(files)
}

async fn find_version(target_version: &Option<String>) -> Result<Version, Error> {
    println!("fetching version manifest...");
    let manifest = assets::fetch_version_manifest().await?;
    println!("fetched version manifest");

    return Ok(match target_version {
        Some(version_str) => {
            let possible_versions = manifest.versions.iter().filter(|v| v.id.contains(version_str)).collect::<Vec<&Version>>();
            let exact_version = manifest.versions.iter().find(|v| v.id == *version_str);

            if possible_versions.is_empty() {
                println!("could not find a matching version to `{}`", version_str);
                Select::new("what version will you use?", manifest.versions).prompt().unwrap()
            } else if possible_versions.len() == 1 {
                possible_versions[0].clone()
            } else if exact_version.is_some() {
                exact_version.unwrap().clone()
            } else {
                println!("multiple matching versions to `{}`", version_str);
                Select::new("what version will you use?", possible_versions).prompt().unwrap().clone()
            }
       },
        None => Select::new("what version will you use?", manifest.versions).prompt().unwrap(),
    });

}

async fn load_and_update_sounds(args: &Args) -> Result<HashMap<PathBuf, Bytes>, Error> {
        let behavior = match (args.behavior.refetch, args.behavior.local) {
            (true, false) => FetchBehavior::Refetch,
            (false, true) => FetchBehavior::CacheOnly,
            (false, false) => FetchBehavior::FetchIfMissing,
            _ => unimplemented!("impossible")
        };

    let version = find_version(&args.target_version).await?;
    
    let asset_index = match behavior {
        FetchBehavior::FetchIfMissing | FetchBehavior::Refetch => {
            println!("fetching asset index...");
            assets::fetch_asset_index(&version).await?.objects
        },
        FetchBehavior::CacheOnly => HashMap::new(),
    };

    let mut sound_assets: HashMap<PathBuf, Bytes> = HashMap::new();

    let cache_path = args.cache.join(PathBuf::from(version.id.clone()));
    let local_paths = visit_dirs(&cache_path).unwrap_or(vec![]);

    let remote_objects = match behavior {
        FetchBehavior::Refetch => {
            asset_index
                .iter()
                .filter(|(key, _)| key.ends_with(".ogg"))
                .map(|(key, val)| (PathBuf::from(key), val))
                .collect::<HashMap<PathBuf, &Object>>()
        },
        FetchBehavior::FetchIfMissing => {
            println!("reading local sound assets...");
            let futures = local_paths
                .iter()
                .map(|path| async move {
                    (path, tokio::fs::read(path).await)
                })
                .collect::<FuturesUnordered<_>>();

            let byte_results = futures.collect::<Vec<(&PathBuf, Result<Vec<u8>, std::io::Error>)>>().await;
            for (sound_path, bytes_res) in byte_results {
                let sound_path = sound_path.strip_prefix(&cache_path);
                match bytes_res {
                    Ok(bytes) => {
                        sound_assets.insert(sound_path.expect("failed to strip cache_path off local").to_path_buf(), bytes.into());
                    },
                    Err(e) => {
                        eprintln!("failed to read `{:?}`, '{}'", sound_path, e);
                    },
                }
            }

            let mut remote_total = 0;
            let sound_objects = asset_index
                .iter()
                .filter(|(key, _)| {
                    if key.ends_with(".ogg") {
                        remote_total += 1;
                    }
                    key.ends_with(".ogg")
                })
                .map(|(key, val)| (PathBuf::from(key), val))
                .filter(|(key, _)| !local_paths.contains(&cache_path.join(key)))
                .collect::<HashMap<PathBuf, &Object>>();

            println!("found remote {} assets and {} local assets. fetching {} assets", remote_total, local_paths.len(), sound_objects.len());

            sound_objects
        },
        FetchBehavior::CacheOnly => HashMap::new(),
    };
    
    if !remote_objects.is_empty() {
        println!("fetching remote assets...");

        let total_requests = Arc::new(AtomicUsize::new(0));
        let errored_requests = Arc::new(AtomicUsize::new(0));

        let request_results: Vec<(PathBuf, Result<Bytes, Error>)> = stream::iter(remote_objects)
            .map(|(key, val)| {
            let total_requests = total_requests.clone();
            let errored_requests = errored_requests.clone();
            async move {
                let res = (key, assets::fetch_asset(&val.hash).await);

                let total = total_requests.load(Ordering::Relaxed);
                total_requests.store(total+1, Ordering::Relaxed); 
                let errored = errored_requests.load(Ordering::Relaxed);
                if res.1.is_err() { 
                    errored_requests.store(errored+1, Ordering::Relaxed);
                }

                let errored = errored_requests.load(Ordering::Relaxed);


                print!("total: {} | err: {}\r", total+1, errored);

                res
            }
            })
            .buffer_unordered(512)
            .collect()
            .await;

        print!("\n");

        for (sound_path, bytes_res) in request_results {
            match bytes_res {
                Ok(bytes) => {
                    sound_assets.insert((*sound_path).to_path_buf(), bytes.clone());
                    let sound_path = cache_path.join(sound_path);
                    tokio::fs::create_dir_all(sound_path.parent().unwrap()).await.expect("failed to create sound directory");
                    tokio::fs::write(sound_path, bytes).await.expect("failed to write to file");
                },
                Err(e) => {
                    eprintln!("failed to fetch `{:?}`, '{:?}'", sound_path, e);
                },
            }
        }
    }

    return Ok(sound_assets);
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let args = Args::parse();

    let sounds = load_and_update_sounds(&args).await?;
    println!("sound assets are up to date!");

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
    // chunk_size = sample_rate (samples/sec) * duration (ms) / 1000 (ms/sec)
    //            = sample_rate * 50ms / 1000ms/sec
    //            = (sample_rate * 50) / 1000 samples
    let chunk_size = (sample_rate * 50) / 1000; 
    println!("sample rate of {}Hz, splitting input into {} sized chunks", sample_rate, chunk_size);

    // your computer DESERVES to panic if `u32` > `usize` and you are running this. that thing should have left a long time ago. picked
    // up its little feet and ran
    let chunks = samples.chunks_exact(chunk_size.try_into().unwrap()).collect::<Vec<&[i16]>>();
    
    for chunk in chunks {

    }

    return Ok(());
}
