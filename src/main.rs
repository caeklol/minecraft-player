use std::path::PathBuf;

use clap::Parser;
use inquire::Select;
use minecraft_player::assets::{self, Version};

#[derive(Debug, clap::Args)]
struct OperatingArgs {

}

#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    #[arg(short, long, help = "version from which to fetch assets from")]
    target_version: Option<String>,

    #[arg(short, long, help = "input audio file")]
    input: PathBuf,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    println!("fetching version manifest...");

    let manifest = match assets::fetch_version_manifest().await {
        Ok(m) => m,
        Err(e) => {
            eprintln!("failed to fetch version manifest: {}", e);
            return
        },
    };

    let version = match args.target_version {
        Some(version_str) => {
            let possible_versions = manifest.versions.iter().filter(|v| v.id.contains(&version_str)).collect::<Vec<&Version>>();
            let exact_version = manifest.versions.iter().find(|v| v.id == version_str);

            if possible_versions.len() == 0 {
                println!("could not find a matching version to `{}`", version_str);
                &Select::new("what version will you use?", manifest.versions).prompt().unwrap()
            } else if possible_versions.len() == 1 {
                possible_versions[0]
            } else if exact_version.is_some() {
                exact_version.unwrap()
            } else {
                println!("multiple matching versions to `{}`", version_str);
                Select::new("what version will you use?", possible_versions).prompt().unwrap()
            }
       },
        None => &Select::new("what version will you use?", manifest.versions).prompt().unwrap(),
    };

    let asset_index = assets::fetch_asset_index(version).await.unwrap();
}
    // out  =hashmap
    // assets = filter(assets)
    // for (key, val) in assets
    // out.key = fetch(https://resources.download.minecraft.net/val.hash.first(2)/val.hash)
