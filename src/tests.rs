fn gen_frequency(hz: f32, sample_rate: usize, duration_ms: usize) -> crate::audio::Sound {
    use crate::audio;
    use std::f32::consts::PI;

    let samples_per_tick = (sample_rate * duration_ms) / 1000;
    let samples: Vec<f32> = (0..samples_per_tick)
        .map(|index| {
            let t = index as f32 / sample_rate as f32;
            (2.0 * PI * hz * t).sin()
        })
        .collect();

    audio::Sound { samples, sample_rate }
}

#[test]
fn test_generator() {
    assert_eq!(gen_frequency(300.0, 44100, 50).samples.len(), 2205);
    assert_eq!(gen_frequency(300.0, 48000, 50).samples.len(), 2400);
}

#[test]
fn test_resample() {
    use crate::audio;
    let tone = gen_frequency(300.0, 44100, 50);
    let resampled_sound = audio::resample(&tone);
    assert_eq!(resampled_sound.samples.len(), 2400);
}

#[test]
fn test_firsttick() {
    use crate::audio;
    let tone = gen_frequency(300.0, 44100, 50);
    let first_tick = audio::first_tick(&tone).expect("first tick returned None on exactly 1 tick of audio");
    assert_eq!(first_tick.samples.len(), tone.samples.len(), "first_tick changed sample length");
    assert_eq!(first_tick.sample_rate, tone.sample_rate, "first_tick changed sample rate");
}

#[test]
fn test_pitch() {
    use crate::audio;
    let tone = gen_frequency(300.0, 48000, 50);
    let original = tone.samples.len();

    let resampled_sound = audio::adjust_pitch(&tone, 0.5);
    assert_eq!(resampled_sound.samples.len(), original * 2);
}
