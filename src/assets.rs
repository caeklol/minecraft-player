use std::{collections::HashMap, fs::exists, io::Cursor, path::{Path, PathBuf}, sync::{atomic::{AtomicUsize, Ordering}, Arc}};

use anyhow::{anyhow, Error};
use bytes::Bytes;
use clap::Parser;
use futures::stream::{self, FuturesUnordered};
use lewton::inside_ogg::OggStreamReader;
use futures::StreamExt;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use tokio::fs;

use crate::mojang::{self, AssetIndex, Object, Version};

#[derive(Parser, Debug)]
pub enum FetchBehavior {
    CacheOnly,
    Refetch,
    FetchIfMissing
}

#[derive(Clone)]
pub struct Sound {
    first_tick: Vec<i16>,
    sample_rate: usize
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ResourceLocation {
    pub name: PathBuf,
    pub volume: Option<f32>,
    pub pitch: Option<f32>,
    pub weight: Option<usize>,
    pub resource_type: Option<String>
}

#[derive(Deserialize, Debug)]
#[serde(untagged)]
pub enum AudioResourceLocation {
    Partial(String),
    Full(ResourceLocation)
}

#[derive(Deserialize, Debug)]
pub struct SoundDefinition {
    pub sounds: Vec<AudioResourceLocation>,
    pub subtitle: Option<String>
}

pub async fn fetch_sound_definitions(assets: &PathBuf, version: &Version, behavior: &FetchBehavior, asset_index: &AssetIndex) -> Result<HashMap<String, SoundDefinition>, Error> {
    let assets_path = assets.join(PathBuf::from(version.id.clone()));
    let sound_definitions_path = &assets_path.join("sound_definitons.json");

    match behavior {
        FetchBehavior::CacheOnly => {
            if fs::try_exists(sound_definitions_path).await? {
                return Ok(serde_json::from_str(&fs::read_to_string(sound_definitions_path).await?)?)
            } else {
                eprintln!("cache-only mode specified without a sound definitions (`sound_definitions.json`) file");
                println!("help: run with refetch or normal fetch behavior");
                return Err(anyhow!("missing sound_definitions"))
            }
        }
        FetchBehavior::FetchIfMissing => {
            if fs::try_exists(sound_definitions_path).await? {
                return Ok(serde_json::from_str(&fs::read_to_string(sound_definitions_path).await?)?)
            }
        },
        FetchBehavior::Refetch => {}
    };

    let sound_definition_asset = asset_index.objects.iter().find(|(k, _)| k.ends_with("sounds.json")).expect("could not find `sounds.json` in asset index");
    let defs_bytes = mojang::fetch_asset(&sound_definition_asset.1.hash).await?;
    let defs_json = str::from_utf8(&defs_bytes)?;
    let defs = serde_json::from_str(&defs_json)?;
    tokio::fs::create_dir_all(assets_path).await.expect("failed to create version directory");
    tokio::fs::write(sound_definitions_path, defs_json).await.expect("failed to write to file");
    return Ok(defs);
}

pub async fn fetch_sounds(cache: &PathBuf, version: &Version, behavior: &FetchBehavior, asset_index: &AssetIndex) -> Result<HashMap<PathBuf, Sound>, Error> {
    let mut sound_assets_bytes: HashMap<PathBuf, Bytes> = HashMap::new();

    let cache_path = cache.join(PathBuf::from(version.id.clone()));
    let local_paths: Vec<PathBuf> = visit_dirs(&cache_path)
        .unwrap_or(vec![])
        .into_iter()
        .filter(|path| path.extension().is_some_and(|ext| ext == "ogg"))
        .collect();

    let remote_objects = match behavior {
        FetchBehavior::Refetch => {
            asset_index.objects
                .iter()
                .filter(|(key, _)| key.ends_with(".ogg"))
                .map(|(key, val)| (PathBuf::from(key), val))
                .collect::<HashMap<PathBuf, &Object>>()
        },
        FetchBehavior::FetchIfMissing => {
            println!("reading local sound assets...");
            let byte_results = stream::iter(&local_paths)
                .map(|path| async move {
                    (path, fs::read(path).await)
                })
                .buffer_unordered(512)
                .collect::<Vec<(&PathBuf, Result<Vec<u8>, std::io::Error>)>>()
                .await;

            for (sound_path, bytes_res) in byte_results {
                let sound_path = sound_path.strip_prefix(&cache_path).unwrap();
                match bytes_res {
                    Ok(bytes) => {
                        sound_assets_bytes.insert(sound_path.to_path_buf(), bytes.into());
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
                    let res = (key, mojang::fetch_asset(&val.hash).await);

                    let total = total_requests.load(Ordering::Relaxed);
                    total_requests.store(total+1, Ordering::Relaxed); 
                    let errored = errored_requests.load(Ordering::Relaxed);
                    if res.1.is_err() { 
                        errored_requests.store(errored+1, Ordering::Relaxed);
                    }

                    let errored = errored_requests.load(Ordering::Relaxed);

                    print!("total: {}, errored: {}\r", total, errored);

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
                    sound_assets_bytes.insert((*sound_path).to_path_buf(), bytes.clone());
                    let sound_path = cache_path.join(sound_path);
                    fs::create_dir_all(sound_path.parent().unwrap()).await.expect("failed to create parent sound directory");
                    fs::write(sound_path, bytes).await.expect("failed to write to file");
                },
                Err(e) => {
                    eprintln!("failed to fetch `{:?}`, '{:?}'", sound_path, e);
                },
            }
        }
    }

    return Ok(sound_assets_bytes
        .into_par_iter()
        .map(|(path, bytes)| -> Result<Option<(PathBuf, Sound)>, Error> {
            let cursor = Cursor::new(bytes);

            let mut ogg_reader = OggStreamReader::new(cursor)
                .map_err(|e| anyhow!("failed to decode {}, {}", path.to_string_lossy(), e))?;

            let sample_rate: usize = ogg_reader.ident_hdr.audio_sample_rate.try_into().unwrap();
            let samples_per_tick = (sample_rate * 50) / 1000; 

            let mut samples = Vec::new();

            let stereo = ogg_reader.ident_hdr.audio_channels == 2;
            
            while let Some(channels) = ogg_reader.read_dec_packet()
                .map_err(|e| anyhow!("failed to read packet for {}, {}", path.to_string_lossy(), e))? {
                if samples.len() >= samples_per_tick {
                    break
                }

                if stereo {
                    let left_channel = &channels[0];
                    let right_channel = &channels[1];

                    let mut averaged: Vec<i16> = Vec::new();
                    for index in 0..left_channel.len() {
                        let avg = (left_channel[index] as i32 + right_channel[index] as i32) / 2i32;
                        averaged.push(avg.try_into().unwrap());
                    }

                    samples.extend(averaged);
                } else {
                    samples.extend(channels[0].clone());
                }
            }

            if samples.len() < samples_per_tick {
                return Ok(None);
            }

            let samples = &samples[0..samples_per_tick];

            return Ok(Some((path.to_path_buf(), Sound {
                first_tick: samples.to_vec(),
                sample_rate
            })));
        })
        .collect::<Result<Vec<Option<(PathBuf, Sound)>>, Error>>()?
        .iter()
        .filter(|t| t.is_some())
        .map(|t| t.clone().unwrap())
        .collect::<HashMap<PathBuf, Sound>>()
    );
}
fn visit_dirs(dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut files = Vec::new();

    if dir.is_dir() {
        for entry in std::fs::read_dir(dir)? {
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
