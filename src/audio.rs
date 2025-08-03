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

pub struct FftResult {
    distribution: Vec<Frequency>
}

impl FftResult {
    pub fn as_volume(&self) -> Vec<f32> {
        self.distribution
            .iter()
            .map(|f| f.volume)
            .collect::<Vec<f32>>()
    }
}

// everything here is heavily inspired from `audioviz`, but this is specifically 
// optimized for audio samples of 1 tick minecraft sounds
#[derive(Clone)]
pub struct Frequency {
    pub freq: f32,
    pub volume: f32,
    pub position: f32
}

impl Frequency {
    pub fn empty() -> Self {
        Frequency {
            volume: 0.0,
            freq: 0.0,
            position: 0.0,
        }
    }
}

pub struct Processor {
    fft_cache: HashMap<usize, Arc<dyn Fft<f32>>>
}

impl Processor {
    pub fn new() -> Self {
        let mut fft_planner = FftPlanner::new();
        let mut fft_cache = HashMap::new();
        fft_cache.insert(time_as_samples!(44100, 50), fft_planner.plan_fft_forward(time_as_samples!(44100, 50))); // # samples for 1 tick at 44.1kHz
        fft_cache.insert(time_as_samples!(48000, 50), fft_planner.plan_fft_forward(time_as_samples!(48000, 50))); // # samples for 1 tick at 48kHz

        Self {
            fft_cache
        } 
    }

    pub fn fft(&self, sound: Sound, resolution: usize) -> FftResult {
        let length = sound.samples.len();
        let mut buffer = Vec::new();

        let mut samples = sound.samples;

        // audioviz::spectrum::processor::Processor.apodize()
        let window = apodize::hamming_iter(length).collect::<Vec<f64>>();
        for i in 0..length {
            samples[i] *= window[i] as f32;
        }

        for sample in samples {
            buffer.push(Complex { re: sample, im: 0.0 });
        }

        let fft = match self.fft_cache.get(&length) {
            Some(fft) => fft,
            None => {
                println!("cache miss, {} sample size, {} sample rate", length, sound.sample_rate);
                &FftPlanner::new().plan_fft_forward(length)
            },
        };

        // audioviz::spectrum::processor::Processor.fft()
        fft.process(&mut buffer);

        let normalized: Vec<f32> = buffer.iter().map(|x| x.norm()).collect();
        let len = normalized.len() / 2 + 1;
        let mut raw = normalized[..len].to_vec();

        // audioviz::spectrum::processor::Processor.normalize(Log)
        for i in 0..raw.len() {
            let percentage = (i + 1) as f32 / raw.len() as f32;
            raw[i] *= 1.0 / 2_f32.log(percentage + 1.0);
            raw[i] *= 0.2;
        }

        let mut bins = Vec::new();

        // audioviz::spectrum::processor::Processor.raw_to_freq_buffer()
        for (i, val) in raw.iter().enumerate() {
            let percentage: f32 = (i + 1) as f32 / raw.len() as f32;
            bins.push(Frequency {
                volume: *val,
                position: percentage,
                freq: percentage * (sound.sample_rate as f32 / 2.0),
            });
        }

        // audioviz::spectrum::processor::Processor.interpolate(Gaps)
        let mut o_buf: Vec<Frequency> = vec![Frequency::empty(); resolution];
        for bin in bins {
            let abs_pos = (o_buf.len() as f32 * bin.position) as usize;
            if abs_pos < o_buf.len() {
                if bin.volume > o_buf[abs_pos].volume {
                    o_buf[abs_pos] = bin.clone();
                }
            }
        }

        FftResult { distribution: o_buf }
    }
}
