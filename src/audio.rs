#[macro_export]
macro_rules! time_as_samples {
    ($sample_rate:expr, $time:expr) => {
        ($sample_rate * $time) / 1000
    };
}
use std::{cmp::min, collections::HashMap, sync::Arc};

use rustfft::{num_complex::Complex, Fft, FftPlanner};
pub use time_as_samples;

#[derive(Clone)]
pub struct Sound {
    pub samples: Vec<f32>,
    pub sample_rate: usize
}

pub struct Frequency {
    pub freq: f32,
    pub volume: f32,
    pub position: f32
}

fn lerp(start: f32, end: f32, t: f32) -> f32 {
    start * (1.0 - t) + end * t
}

/// rescales audio samples by a given pitch
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
        samples: samples.clone()
    };
}

// heavily inspired and simplified version of `audioviz` processor, specifically optimized for
// audio samples of single ticks of minecraft sounds
pub struct Processor {
    fft_cache: HashMap<usize, Arc<dyn Fft<f32>>>
}

impl Processor {
    pub fn new() -> Self {
        let mut fft_planner = FftPlanner::new();
        let mut fft_cache = HashMap::new();
        fft_cache.insert(2205, fft_planner.plan_fft_forward(2205)); // # samples for 1 tick at 44.1kHz
        fft_cache.insert(2400, fft_planner.plan_fft_forward(2400)); // # samples for 1 tick at 48kHz

        Self {
            fft_cache
        } 
    }

    pub fn fft(&self, sound: Sound) -> Vec<Frequency> {
        let length = sound.samples.len();
        let mut buffer = Vec::new();

        for sample in sound.samples {
            buffer.push(Complex { re: sample, im: 0.0 });
        }

        let fft = match self.fft_cache.get(&length) {
            Some(fft) => fft,
            None => {
                &FftPlanner::new().plan_fft_forward(length)
            },
        };

        fft.process(&mut buffer);

        let normalized: Vec<f32> = buffer.iter().map(|x| x.norm_sqr()).collect();
        let len = normalized.len() / 2 + 1;
        let mut raw = normalized[..len].to_vec();

        // log volume normalization
        for i in 0..raw.len() {
            let percentage = (i + 1) as f32 / raw.len() as f32;
            raw[i] *= 1.0 / 2_f32.log(percentage + 1.0);
            raw[i] *= 0.2;
        }

        let mut output = Vec::new();

        for (i, val) in raw.iter().enumerate() {
            let percentage: f32 = (i + 1) as f32 / raw.len() as f32;
            output.push(Frequency {
                volume: *val,
                position: percentage,
                freq: percentage * (sound.sample_rate as f32 / 2.0),
            });
        }

        Vec::new()
    }
}
