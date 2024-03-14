use std::{
    mem::transmute,
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

use halo2curves::{bn256, group::Curve, CurveExt};
use icicle_cuda_runtime::stream::CudaStream;
use ingonyama_grumpkin_msm::{bn256_msm, default_config, read_arecibo_data};
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;

pub fn gen_points(npoints: usize) -> Vec<bn256::G1Affine> {
    let ret = vec![bn256::G1Affine::default(); npoints];

    let mut rnd = vec![0u8; 32 * npoints];
    ChaCha20Rng::from_entropy().fill_bytes(&mut rnd);

    let n_workers = rayon::current_num_threads();
    let work = AtomicUsize::new(0);
    rayon::scope(|s| {
        for _ in 0..n_workers {
            s.spawn(|_| {
                let hash = bn256::G1::hash_to_curve("foobar");

                let mut stride = 1024;
                let mut tmp = vec![bn256::G1::default(); stride];

                loop {
                    let work = work.fetch_add(stride, Ordering::Relaxed);
                    if work >= npoints {
                        break;
                    }
                    if work + stride > npoints {
                        stride = npoints - work;
                        unsafe { tmp.set_len(stride) };
                    }
                    for (i, point) in tmp.iter_mut().enumerate().take(stride) {
                        let off = (work + i) * 32;
                        *point = hash(&rnd[off..off + 32]);
                    }
                    #[allow(mutable_transmutes)]
                    bn256::G1::batch_normalize(&tmp, unsafe {
                        transmute::<&[bn256::G1Affine], &mut [bn256::G1Affine]>(
                            &ret[work..work + stride],
                        )
                    });
                }
            })
        }
    });

    ret
}

fn main() {
    let section = "witness_0x02c29fabf43b87a73513f6ecbfb348c146809c1609c21b48333a8096700d63ad";
    let label_i = format!("len_8131411_{}", 0);
    // let section = "cross_term_0x02c29fabf43b87a73513f6ecbfb348c146809c1609c21b48333a8096700d63ad";
    // let label_i = format!("len_9873811_{}", 0);
    let scalars: Vec<bn256::Fr> = read_arecibo_data(section.to_string(), label_i.clone());
    let size = scalars.len();
    let points = gen_points(size);

    let stream = CudaStream::create().expect("Failed to create CUDA stream");
    let cfg = default_config(&stream);

    println!("start msm");
    let start = Instant::now();
    let res = bn256_msm(&points, &scalars, &cfg);
    println!("{:?} {:?}", start.elapsed(), res);
}
