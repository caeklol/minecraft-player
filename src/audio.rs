#[macro_export]
macro_rules! time_as_samples {
    ($sample_rate:expr, $time:expr) => {
        ($sample_rate * $time) / 1000
    };
}
use std::{cmp::min, collections::HashMap, sync::Arc};

use ndarray::Array2;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rustfft::{num_complex::Complex, Fft, FftPlanner};
pub use time_as_samples;

use crate::algebra;

#[derive(Clone)]
pub struct Sound {
    pub samples: Vec<f32>,
    pub sample_rate: usize
}


fn lerp(start: f32, end: f32, t: f32) -> f32 {
    start * (1.0 - t) + end * t
}

pub fn permute_with_pitch(samples: Vec<(String, Sound)>, resolution: usize) -> Vec<((String, f32), Sound)> {
    let pitches = algebra::interpolated_range(0.5, 2.0, resolution);
    let zipped = samples.into_iter().flat_map(|(st, s)| {
        pitches
            .iter()
            .map(|p| ((st.clone(), *p), s.clone()))
            .collect::<Vec<((String, f32), Sound)>>()
    }).collect::<Vec<((String, f32), Sound)>>();

    return zipped
        .into_par_iter()
        .filter_map(|((id, pitch), sound)| Some(((id, pitch), first_tick(&adjust_pitch(&sound, pitch))?)))
        .collect::<Vec<((String, f32), Sound)>>();
}

pub fn first_tick(sound: &Sound) -> Option<Sound> {
    let samples_per_tick = (sound.sample_rate * 50) / 1000;
    if sound.samples.len() < samples_per_tick {
        None
    } else {
        return Some(Sound {
            samples: (sound.samples[0..samples_per_tick]).to_vec(),
            sample_rate: sound.sample_rate
        });
    }
}

/// rescales audio samples by a given pitch by time dilation
/// fills gaps linearly
pub fn adjust_pitch(sound: &Sound, pitch: f32) -> Sound {
    if pitch == 1.0 {
        return sound.clone();
    }

    let samples = &sound.samples;
    let new_length = (samples.len() as f32 / pitch) as usize;

    let mut scaled = Vec::with_capacity(new_length);

    for i in 0..new_length {
        let original_index = i as f32 * pitch;

        let lower_index = original_index.floor() as usize;
        let upper_index = original_index.ceil() as usize;

        let upper_index = if upper_index >= samples.len() { samples.len() - 1 } else { upper_index };

        if lower_index != upper_index {
            let t = original_index - lower_index as f32;
            let interpolated_value = lerp(samples[lower_index], samples[upper_index], t);
            scaled.push(interpolated_value);
        } else {
            scaled.push(samples[lower_index]);
        }
    }

    return Sound { 
        sample_rate: sound.sample_rate,
        samples: scaled
    };
}

/// handles up and downsampling
/// linear interpolation
/// samples to 48khz
/// assumes 1 tick of audio
pub fn resample(sound: &Sound) -> Sound {
    let input_len = sound.samples.len();
    let output_len = 2400;

    if input_len == 0 || output_len == 0 {
        return Sound {
            samples: Vec::new(),
            sample_rate: sound.sample_rate,
        };
    }

    if input_len == output_len {
        return Sound {
            samples: sound.samples.clone(),
            sample_rate: sound.sample_rate,
        };
    }

    let mut resampled = Vec::with_capacity(output_len);
    let step = (input_len - 1) as f32 / (output_len - 1) as f32;

    for i in 0..output_len {
        let pos = i as f32 * step;
        let index = pos.floor() as usize;
        let frac = pos - index as f32;

        let s1 = sound.samples.get(index).copied().unwrap_or(0.0);
        let s2 = sound.samples.get(index + 1).copied().unwrap_or(s1);

        resampled.push(lerp(s1, s2, frac));
    }

    Sound {
        samples: resampled,
        sample_rate: 48000,
    }
}
