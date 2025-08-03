use std::{collections::HashMap, fmt::Display, hash::Hash};
use bytes::Bytes;

use anyhow::{Error, anyhow};
use serde::{Deserialize, Deserializer};
use serde_json::Value;
use sha1_smol::Sha1;

static VERSION_MANIFEST_URL: &str = "https://piston-meta.mojang.com/mc/game/version_manifest_v2.json";
static ASSET_URL: &str = "https://resources.download.minecraft.net";

#[derive(Deserialize, Clone, Debug)]
pub struct LatestVersion {
    pub release: String,
    pub snapshot: String
}

#[derive(Deserialize, Clone, Debug)]
pub struct Version {
    pub id: String,
    pub url: String
}

impl Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.id)
    }
}

#[derive(Deserialize, Clone, Debug)]
pub struct VersionManifest {
    pub latest: LatestVersion,
    pub versions: Vec<Version>
}

pub async fn fetch_version_manifest() -> Result<VersionManifest, Error> {
    Ok(reqwest::get(VERSION_MANIFEST_URL)
        .await?
        .json::<VersionManifest>()
        .await?
    )
}

#[derive(Debug)]
struct VersionPackage {
    asset_index_url: String,
}

impl<'de> Deserialize<'de> for VersionPackage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;

        let url = value
            .get("assetIndex")
            .and_then(|ai| ai.get("url"))
            .and_then(|url| url.as_str())
            .ok_or_else(|| serde::de::Error::custom("Missing 'assetIndex.url' field"))?
            .to_string();

        Ok(VersionPackage { asset_index_url: url })
    }
}

#[derive(Deserialize, Clone, Debug)]
pub struct Object {
    pub hash: String,
    pub size: usize
}

#[derive(Deserialize, Clone, Debug)]
pub struct AssetIndex {
    pub objects: HashMap<String, Object>
}

pub async fn fetch_asset_index(version: &Version) -> Result<AssetIndex, Error> {
    let package = reqwest::get(&version.url)
        .await?
        .json::<VersionPackage>()
        .await?;

    Ok(reqwest::get(&package.asset_index_url)
        .await?
        .json::<AssetIndex>()
        .await?
    )
}

pub async fn fetch_asset(hash: &str) -> Result<Bytes, Error> {
    let mut hasher = Sha1::new();
    let response_bytes = reqwest::get(format!("{}/{}/{}", ASSET_URL, &hash[0..2], hash))
        .await?
        .bytes()
        .await?;

    hasher.update(&response_bytes);

    if hasher.digest().to_string() != hash {
        return Err(anyhow!("repsonse hash did not match asset hash"));
    }

    return Ok(response_bytes);
}


