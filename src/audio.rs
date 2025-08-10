#[macro_export]
macro_rules! time_as_samples {
    ($sample_rate:expr, $time:expr) => {
        ($sample_rate * $time) / 1000
    };
}
use std::{cmp::min, collections::HashMap, sync::Arc};

use ndarray::Array2;
use num_traits::Pow;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rustfft::{num_complex::{Complex, Complex32, Complex64}, Fft, FftPlanner};
pub use time_as_samples;

use crate::algebra;

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
        .map(|((id, pitch), mut sound)| ((id, pitch), sound.adjust_pitch(pitch).first_tick().clone()))
        .collect::<Vec<((String, f32), Sound)>>();
}

#[derive(Clone)]
pub struct Sound {
    pub samples: Vec<f32>,
    pub sample_rate: usize
}

impl Sound {
    /// pads silence with zeroes
    pub fn first_tick(&mut self) -> &mut Self {
        let samples_per_tick = f32::ceil((self.sample_rate as f32 * 50.0) / 1000.0) as usize;

        if self.samples.len() < samples_per_tick {
            self.samples.resize(samples_per_tick, 0.0);
        } else {
            self.samples = (self.samples[0..samples_per_tick]).to_vec();
        }

        return self;
    }

    /// handles up and downsampling
    /// linear interpolation
    pub fn resample(&mut self, new_rate: usize) -> &mut Self {
        let input_len = self.samples.len();
        let output_len = (input_len * new_rate) / self.sample_rate;

        if input_len == 0 || output_len == 0 {
            panic!("resample failed, input or output len was 0");
        }

        if input_len == output_len {
            return self;
        }

        let mut resampled = Vec::with_capacity(output_len);
        let step = (input_len - 1) as f32 / (output_len - 1) as f32;

        for i in 0..output_len {
            let pos = i as f32 * step;
            let index = pos.floor() as usize;
            let frac = pos - index as f32;

            let s1 = self.samples.get(index).copied().unwrap_or(0.0);
            let s2 = self.samples.get(index + 1).copied().unwrap_or(s1);

            resampled.push(lerp(s1, s2, frac));
        }

        self.samples = resampled;
        self.sample_rate = new_rate;

        return self;
    }

    /// rescales audio samples by a given pitch by time dilation
    /// fills gaps linearly
    pub fn adjust_pitch(&mut self, pitch: f32) -> &mut Self {
        if pitch == 1.0 {
            return self;
        }

        let new_length = (self.samples.len() as f32 / pitch) as usize;

        let mut scaled = Vec::with_capacity(new_length);

        for i in 0..new_length {
            let original_index = i as f32 * pitch;

            let lower_index = original_index.floor() as usize;
            let upper_index = original_index.ceil() as usize;

            let upper_index = if upper_index >= self.samples.len() { self.samples.len() - 1 } else { upper_index };

            if lower_index != upper_index {
                let t = original_index - lower_index as f32;
                let interpolated_value = lerp(self.samples[lower_index], self.samples[upper_index], t);
                scaled.push(interpolated_value);
            } else {
                scaled.push(self.samples[lower_index]);
            }
        }

        self.samples = scaled;

        return self;
    }

    pub fn adjust_volume(&mut self, volume: f32) -> &mut Self {
        if volume == 1.0 {
            return self;
        }

        for sample in &mut self.samples {
            *sample = *sample * volume;
        }

        return self;
    }

    /// applies filtering to boost mids and decrease lows
    /// this makes it so important parts (voice, etc) are prioritized in
    /// reconstruction rather than bass (drums, etc) which our ears are
    /// more sensitive to
    pub fn mel(&mut self, processor: &Processor) -> &mut Self {
        let mut spectrum = processor.fft(self.clone());

        for bin in spectrum.iter_mut() {
            let mel_freq = (2595.0 * (1.0 + (bin.freq as f32 / 700.0)).log10()) / 24000.0;
            let high_pass = (bin.freq as f32) / ((bin.freq as f32).pow(2.0) + (100 as f32).pow(2.0)) + 0.4;
            bin.complex *= (mel_freq * 2.0) * (high_pass.min(1.0));
        }

        self.samples = processor.ifft(spectrum);

        return self;
    }
}

// todo: handroll FFT and IFFT
#[derive(Clone)]
pub struct FftBin {
    pub freq: f32,
    pub complex: Complex32
}

impl FftBin {
    pub fn empty() -> Self {
        FftBin {
            freq: 0.0,
            complex: Complex32::new(0.0, 0.0)
        }
    }
}

pub struct Processor {
    fft_cache: HashMap<usize, Arc<dyn Fft<f32>>>,
    ifft_cache: HashMap<usize, Arc<dyn Fft<f32>>>
}

impl Processor {
    pub fn new() -> Self {
        let mut fft_planner = FftPlanner::new();
        let mut fft_cache = HashMap::new();
        let mut ifft_cache = HashMap::new();
        fft_cache.insert(time_as_samples!(44100, 50), fft_planner.plan_fft_forward(time_as_samples!(44100, 50))); // # samples for 1 tick at 44.1kHz
        fft_cache.insert(time_as_samples!(48000, 50), fft_planner.plan_fft_forward(time_as_samples!(48000, 50))); // # samples for 1 tick at 48kHz

        ifft_cache.insert(time_as_samples!(44100, 50), fft_planner.plan_fft_inverse(time_as_samples!(44100, 50))); // # samples for 1 tick at 44.1kHz
        ifft_cache.insert(time_as_samples!(48000, 50), fft_planner.plan_fft_inverse(time_as_samples!(48000, 50))); // # samples for 1 tick at 48kHz

        Self {
            fft_cache,
            ifft_cache
        } 
    }

    pub fn fft(&self, sound: Sound) -> Vec<FftBin> {
        let length = sound.samples.len();
        let mut buffer = Vec::with_capacity(length);

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

        fft.process(&mut buffer);

        let mut bins: Vec<FftBin> = Vec::with_capacity(length);

        for (index, bin) in buffer.iter().enumerate() {
            bins.push(FftBin {
                freq: index as f32 * sound.sample_rate as f32 / buffer.len() as f32,
                complex: *bin
            });
        }

        bins
    }


    pub fn ifft(&self, spectrum: Vec<FftBin>) -> Vec<f32> {
        let mut buffer = spectrum.iter().map(|f| f.complex).collect::<Vec<Complex32>>();
        let length = buffer.len();

        let ifft = match self.ifft_cache.get(&length) {
            Some(ifft) => ifft,
            None => {
                println!("cache miss, inverse, {} sample size", length);
                &FftPlanner::new().plan_fft_inverse(length)
            },
        };

        ifft.process(&mut buffer);
        buffer.iter().map(|c| c.re).collect::<Vec<f32>>()
    }
}
