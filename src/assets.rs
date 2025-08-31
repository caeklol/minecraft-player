use std::{collections::HashMap, io::Cursor, path::{Path, PathBuf}, sync::{atomic::{AtomicUsize, Ordering}, Arc}};

use anyhow::{anyhow, Error};
use bytes::Bytes;
use clap::Parser;
use futures::stream::{self};
use lewton::inside_ogg::OggStreamReader;
use futures::StreamExt;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use tokio::fs;
use tracing::{event, instrument, span, Level};

use crate::{audio::Sound, mojang::{self, AssetIndex, Object, Version}};

#[derive(Parser, Debug)]
pub enum FetchBehavior {
    CacheOnly,
    Refetch,
    FetchIfMissing
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

pub async fn fetch_sound_definitions(assets: &PathBuf, version: &Version, behavior: &FetchBehavior, asset_index: &AssetIndex) -> Result<HashMap<String, SoundDefinition>, Error> {
    let _span = span!(Level::INFO, "fetch_sound_definitions", tag = "assets").entered();

    let assets_path = assets.join(PathBuf::from(version.id.clone()));
    let sound_definitions_path = &assets_path.join("sound_definitons.json");

    match behavior {
        FetchBehavior::CacheOnly => {
            if fs::try_exists(sound_definitions_path).await? {
                return Ok(serde_json::from_str(&fs::read_to_string(sound_definitions_path).await?)?)
            } else {
                event!(Level::ERROR, "cache-only mode specified without a sound definitions (`sound_definitions.json`) file");
                event!(Level::ERROR, help = true, "run with refetch or normal fetch behavior");
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

/// converts all stereo sounds to mono
pub async fn fetch_sounds(assets: &PathBuf, version: &Version, behavior: &FetchBehavior, asset_index: &AssetIndex) -> Result<HashMap<PathBuf, Sound>, Error> {
    let _span = span!(Level::INFO, "fetch_sounds", tag = "assets").entered();

    event!(Level::INFO, "eggs in the morning with toast");

    let mut sound_assets_bytes: HashMap<PathBuf, Bytes> = HashMap::new();

    let cache_path = assets.join(PathBuf::from(version.id.clone()));
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
            event!(Level::INFO, "reading local sound assets");
            let byte_results = stream::iter(&local_paths)
                .map(|path| async move {
                    (path, fs::read(path).await)
                })
                .buffer_unordered(512)
                .collect::<HashMap<&PathBuf, Result<Vec<u8>, std::io::Error>>>()
                .await;

            for (sound_path, bytes_res) in byte_results {
                let sound_path = sound_path.strip_prefix(&cache_path).unwrap();
                match bytes_res {
                    Ok(bytes) => {
                        sound_assets_bytes.insert(sound_path.to_path_buf(), bytes.into());
                    },
                    Err(e) => {
                        event!(Level::WARN, "failed to read `{:?}`, '{}'", sound_path, e);
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

            event!(Level::INFO, "found remote {} assets and {} local assets. fetching {} assets", remote_total, local_paths.len(), sound_objects.len());

            sound_objects
        },
        FetchBehavior::CacheOnly => {
            event!(Level::INFO, "reading local sound assets");
            let byte_results = stream::iter(&local_paths)
                .map(|path| async move {
                    (path, fs::read(path).await)
                })
                .buffer_unordered(512)
                .collect::<HashMap<&PathBuf, Result<Vec<u8>, std::io::Error>>>()
                .await;

            for (sound_path, bytes_res) in byte_results {
                let sound_path = sound_path.strip_prefix(&cache_path).unwrap();
                match bytes_res {
                    Ok(bytes) => {
                        sound_assets_bytes.insert(sound_path.to_path_buf(), bytes.into());
                    },
                    Err(e) => {
                        event!(Level::WARN, "failed to read `{:?}`, '{}'", sound_path, e);
                    },
                }
            }

            HashMap::new()
        },
    };
    
    if !remote_objects.is_empty() {
        event!(Level::INFO, "fetching remote assets");

        let total_requests = Arc::new(AtomicUsize::new(0));
        let errored_requests = Arc::new(AtomicUsize::new(0));

        let request_results: HashMap<PathBuf, Result<Bytes, Error>> = stream::iter(remote_objects)
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

                    event!(Level::DEBUG, "total: {}, errored: {}\r", total, errored);

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
                    event!(Level::WARN, "failed to fetch `{:?}`, '{:?}'", sound_path, e);
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
            
            while let Some(channels) = ogg_reader.read_dec_packet_generic::<Vec<Vec<f32>>>()
                .map_err(|e| anyhow!("failed to read packet for {}, {}", path.to_string_lossy(), e))? {
                    
                if samples.len() >= (samples_per_tick * 5) { // max pitch is 2, and pitch is only
                                                             // ever applied twice, so only ever
                                                             // need 4 samples. 5 for leeway
                    break
                }

                if stereo {
                    let left_channel = &channels[0];
                    let right_channel = &channels[1];

                    let mut averaged = Vec::new();
                    for index in 0..left_channel.len() {
                        let avg = (left_channel[index] + right_channel[index] ) / 2.0;
                        averaged.push(avg);
                    }

                    samples.extend(averaged);
                } else {
                    samples.extend(channels[0].clone());
                }
            }

            return Ok(Some((path.to_path_buf(), Sound {
                samples: samples.to_vec(),
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
