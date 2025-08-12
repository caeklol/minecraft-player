use crate::algebra;

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
    let mut tone = gen_frequency(300.0, 44100, 50);
    tone.resample(48000);
    assert_eq!(tone.samples.len(), 2400);
}

#[test]
fn test_firsttick() {
    let tone = gen_frequency(300.0, 44100, 50);
    let mut binding = tone.clone();
    let first_tick = binding.first_tick();
    assert_eq!(first_tick.samples.len(), tone.samples.len(), "first_tick changed sample length");
    assert_eq!(first_tick.sample_rate, tone.sample_rate, "first_tick changed sample rate");
}

#[test]
fn test_pitch() {
    let mut tone = gen_frequency(300.0, 48000, 50);
    let original = tone.samples.len();

    let resampled_sound = tone.adjust_pitch(0.5);
    assert_eq!(resampled_sound.samples.len(), original * 2);
}

#[test]
fn test_layout() {
    let matrix = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0]
    ];
    let matrix_ndarray = algebra::matrix_from_vecs(matrix.clone()).unwrap();
    let flattened: Vec<f32> = matrix.into_iter().flatten().collect();
    let ndarray_vec: Vec<f32> = matrix_ndarray.iter().cloned().collect();
    assert!(flattened.iter().partial_cmp(&ndarray_vec).expect("failed to compare").is_eq());
}
