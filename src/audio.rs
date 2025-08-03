#[macro_export]
macro_rules! time_as_samples {
    ($sample_rate:expr, $time:expr) => {
        ($sample_rate * $time) / 1000
    };
}
use std::cmp::min;

pub use time_as_samples;
use anyhow::{Error, anyhow};

fn lerp(start: i16, end: i16, t: f32) -> i16 {
    (start as f32 * (1.0 - t) + end as f32 * t) as i16
}

/// rescales audio samples by a given pitch
/// fills gaps linearly
pub fn adjust_pitch(samples: &Vec<i16>, pitch: f32) -> Vec<i16> {
    let new_length = (samples.len() as f32 / pitch) as usize;

    let mut scaled = Vec::with_capacity(new_length);

    for i in 0..new_length {
        let original_index = i as f32 * pitch;

        let lower_index = original_index.floor() as usize;
        let upper_index = original_index.ceil() as usize;

        if lower_index != upper_index {
            let t = original_index - lower_index as f32;
            let interpolated_value = lerp(samples[lower_index], samples[upper_index], t);
            scaled.push(interpolated_value);
        } else {
            scaled.push(samples[lower_index]);
        }
    }

    scaled
}
