use std::{collections::HashMap, fs, io, path::{Path, PathBuf}};

use futures::{stream, StreamExt};
use bytes::Bytes;
use clap::Parser;
use futures::stream::FuturesUnordered;
use inquire::Select;
use minecraft_player::assets::{self, Object, Version};
use anyhow::Error;

#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    #[arg(short, long, help = "version from which to fetch assets from")]
    target_version: Option<String>,

    #[arg(short, long, help = "refetch all assets and replace all locally cached files")]
    refetch: bool,

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

async fn find_version(target_version: Option<String>) -> Result<Version, Error> {
    println!("fetching version manifest...");
    let manifest = assets::fetch_version_manifest().await?;
    println!("fetched version manifest");

    return Ok(match target_version {
        Some(version_str) => {
            let possible_versions = manifest.versions.iter().filter(|v| v.id.contains(&version_str)).collect::<Vec<&Version>>();
            let exact_version = manifest.versions.iter().find(|v| v.id == version_str);

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

async fn load_and_update_sounds(args: Args) -> Result<HashMap<PathBuf, Bytes>, Error> {
    let version = find_version(args.target_version).await?;
    
    println!("fetching asset index...");
    let asset_index = assets::fetch_asset_index(&version).await?;
    println!("fetched asset index");

    let mut sound_assets: HashMap<PathBuf, Bytes> = HashMap::new();

    let cache_path = args.cache.join(PathBuf::from(version.id.clone()));
    let local_paths = visit_dirs(&cache_path).unwrap_or(vec![]);

    let remote_objects = match args.refetch {
        true => {
            asset_index.objects
                .iter()
                .filter(|(key, _)| key.ends_with(".ogg"))
                .map(|(key, val)| (PathBuf::from(key), val))
                .collect::<HashMap<PathBuf, &Object>>()
        },
        false => {
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
            let sound_objects = asset_index.objects
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
    };
    
    if !remote_objects.is_empty() {
        println!("fetching remote assets...");

        let request_results: Vec<(PathBuf, Result<Bytes, reqwest::Error>)> = stream::iter(remote_objects)
            .map(|(key, val)| async move {
                (key, assets::fetch_asset(&val.hash).await)
            })
            .buffer_unordered(512)
            .collect()
            .await;

        for (sound_path, bytes_res) in request_results {
            match bytes_res {
                Ok(bytes) => {
                    sound_assets.insert((*sound_path).to_path_buf(), bytes.clone());
                    let sound_path = cache_path.join(sound_path);
                    tokio::fs::create_dir_all(sound_path.parent().unwrap()).await.expect("failed to create sound directory");
                    tokio::fs::write(sound_path, bytes).await.expect("failed to write to file");
                },
                Err(e) => {
                    eprintln!("failed to fetch `{:?}`, '{:#?}'", sound_path, e);
                },
            }
        }
    }

    println!("sound assets are up to date!");

    return Ok(sound_assets);
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let args = Args::parse();

    let sounds = load_and_update_sounds(args).await?;

    println!("all sounds loaded into memory!");
    return Ok(());
}
