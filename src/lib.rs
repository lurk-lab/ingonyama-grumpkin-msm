use halo2curves::grumpkin;
use halo2curves::{bn256, ff::PrimeField};
use icicle_bn254::curve as icicle_bn254;
use icicle_core::{msm, traits::FieldImpl};
use icicle_cuda_runtime::memory::{DeviceVec, HostSlice};
use icicle_cuda_runtime::stream::CudaStream;
use icicle_grumpkin::curve as icicle_grumpkin;

use serde::de::DeserializeOwned;
use std::convert::TryInto;
use std::fs::File;
use std::io::BufReader;

/// Path to the directory where Arecibo data will be stored.
pub static ARECIBO_DATA: &str = ".arecibo_data";

/// Reads and deserializes data from a specified section and label.
pub fn read_arecibo_data<T: DeserializeOwned>(section: String, label: String) -> Vec<T> {
    let root_dir = home::home_dir().unwrap().join(ARECIBO_DATA);
    let section_path = root_dir.join(section);
    assert!(section_path.exists(), "Section directory does not exist");

    let file_path = section_path.join(label);
    assert!(file_path.exists(), "Data file does not exist");

    let file = File::open(file_path).expect("Failed to open data file");
    let reader = BufReader::new(file);

    bincode::deserialize_from(reader).expect("Failed to read data")
}

// fn halo_to_icicle_scalar(scalar: &bn256::Fr) -> ScalarField {
//     ScalarField::from_bytes_le(scalar.to_bytes().as_ref().try_into().unwrap())
// }

// fn halo_to_icicle_point(point: &bn256::G1Affine) -> G1Affine {
//     let x = BaseField::from_bytes_le(point.x.to_bytes().as_ref().try_into().unwrap());
//     let y = BaseField::from_bytes_le(point.y.to_bytes().as_ref().try_into().unwrap());

//     G1Affine { x, y }
// }

fn icicle_to_bn256_point(point: &icicle_bn254::G1Affine) -> bn256::G1Affine {
    bn256::G1Affine {
        x: bn256::Fq::from_repr(point.x.to_bytes_le().try_into().unwrap()).unwrap(),
        y: bn256::Fq::from_repr(point.y.to_bytes_le().try_into().unwrap()).unwrap(),
    }
}

fn icicle_to_grumpkin_point(point: &icicle_grumpkin::G1Affine) -> grumpkin::G1Affine {
    grumpkin::G1Affine {
        x: grumpkin::Fq::from_repr(point.x.to_bytes_le().try_into().unwrap()).unwrap(),
        y: grumpkin::Fq::from_repr(point.y.to_bytes_le().try_into().unwrap()).unwrap(),
    }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct Config {
    pub stream: CudaStream,
}

unsafe impl Sync for Config {}

impl Config {
    pub fn new() -> Self {
        let stream = CudaStream::create().expect("Failed to create CUDA stream");
        Config { stream }
    }
}

pub fn default_config(stream: &CudaStream) -> msm::MSMConfig {
    let mut cfg = msm::MSMConfig::default();
    cfg.ctx.stream = &stream;
    cfg.is_async = true; // Enable asynchronous execution
    cfg.are_points_montgomery_form = true;
    cfg.are_scalars_montgomery_form = true;
    cfg
}

pub fn bn256_msm(
    points: &[bn256::G1Affine],
    scalars: &[bn256::Fr],
    cfg: &msm::MSMConfig,
) -> bn256::G1 {
    // Wrap points and scalars in HostOrDeviceSlice for MSM
    let points =
        unsafe { &*(points as *const [bn256::G1Affine] as *const [icicle_bn254::G1Affine]) };
    let scalars =
        unsafe { &*(scalars as *const [bn256::Fr] as *const [icicle_bn254::ScalarField]) };
    let points_host = HostSlice::from_slice(points);
    let scalars_host = HostSlice::from_slice(scalars);

    // Allocate memory on the CUDA device for MSM results
    let mut msm_results = DeviceVec::<icicle_bn254::G1Projective>::cuda_malloc(1).unwrap();

    // Execute MSM on the device
    let mut msm_host_result = vec![icicle_bn254::G1Projective::zero(); 1];
    msm::msm(scalars_host, points_host, &cfg, &mut msm_results[..]).expect("Failed to execute MSM");

    msm_results
        .copy_to_host(HostSlice::from_mut_slice(&mut msm_host_result[..]))
        .unwrap();

    icicle_to_bn256_point(&icicle_bn254::G1Affine::from(msm_host_result[0])).into()
}

pub fn grumpkin_msm(
    points: &[grumpkin::G1Affine],
    scalars: &[grumpkin::Fr],
    cfg: &msm::MSMConfig,
) -> grumpkin::G1 {
    // Wrap points and scalars in HostOrDeviceSlice for MSM
    let points =
        unsafe { &*(points as *const [grumpkin::G1Affine] as *const [icicle_grumpkin::G1Affine]) };
    let scalars =
        unsafe { &*(scalars as *const [grumpkin::Fr] as *const [icicle_grumpkin::ScalarField]) };
    let points_host = HostSlice::from_slice(points);
    let scalars_host = HostSlice::from_slice(scalars);

    // Allocate memory on the CUDA device for MSM results
    let mut msm_results = DeviceVec::<icicle_grumpkin::G1Projective>::cuda_malloc(1).unwrap();

    // Execute MSM on the device
    let mut msm_host_result = vec![icicle_grumpkin::G1Projective::zero(); 1];
    msm::msm(scalars_host, points_host, &cfg, &mut msm_results[..]).expect("Failed to execute MSM");

    msm_results
        .copy_to_host(HostSlice::from_mut_slice(&mut msm_host_result[..]))
        .unwrap();

    icicle_to_grumpkin_point(&icicle_grumpkin::G1Affine::from(msm_host_result[0])).into()
}

#[cfg(test)]
mod test {
    use core::mem::transmute;
    use core::sync::atomic::*;
    use halo2curves::ff::PrimeField;
    use halo2curves::group::Curve;
    use halo2curves::CurveExt;
    use halo2curves::{bn256, grumpkin};
    use icicle_cuda_runtime::stream::CudaStream;
    use rand::{RngCore, SeedableRng};
    use rand_chacha::ChaCha20Rng;
    use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
    use std::sync::{Arc, Mutex};

    use crate::default_config;

    pub fn gen_points<G1: Curve + CurveExt<AffineExt = Affine>, Affine: Clone + Default + Sync>(
        npoints: usize,
    ) -> Vec<Affine> {
        let ret = vec![Affine::default(); npoints];

        let mut rnd = vec![0u8; 32 * npoints];
        ChaCha20Rng::from_entropy().fill_bytes(&mut rnd);

        let n_workers = rayon::current_num_threads();
        let work = AtomicUsize::new(0);
        rayon::scope(|s| {
            for _ in 0..n_workers {
                s.spawn(|_| {
                    let hash = G1::hash_to_curve("foobar");

                    let mut stride = 1024;
                    let mut tmp = vec![G1::default(); stride];

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
                        G1::batch_normalize(&tmp, unsafe {
                            transmute::<&[Affine], &mut [Affine]>(&ret[work..work + stride])
                        });
                    }
                })
            }
        });

        ret
    }

    pub fn gen_scalars<Fr: PrimeField>(npoints: usize) -> Vec<Fr> {
        let ret = Arc::new(Mutex::new(vec![Fr::default(); npoints]));

        let n_workers = rayon::current_num_threads();
        let work = Arc::new(AtomicUsize::new(0));

        rayon::scope(|s| {
            for _ in 0..n_workers {
                let ret_clone = Arc::clone(&ret);
                let work_clone = Arc::clone(&work);

                s.spawn(move |_| {
                    let mut rng = ChaCha20Rng::from_entropy();
                    loop {
                        let work = work_clone.fetch_add(1, Ordering::Relaxed);
                        if work >= npoints {
                            break;
                        }
                        let mut ret = ret_clone.lock().unwrap();
                        ret[work] = Fr::random(&mut rng);
                    }
                });
            }
        });

        Arc::try_unwrap(ret).unwrap().into_inner().unwrap()
    }

    pub fn bn256_naive_multiscalar_mul(
        points: &[bn256::G1Affine],
        scalars: &[bn256::Fr],
    ) -> bn256::G1Affine {
        let ret: bn256::G1 = points
            .par_iter()
            .zip_eq(scalars.par_iter())
            .map(|(p, s)| p * s)
            .sum();

        ret.to_affine()
    }

    pub fn grumpkin_naive_multiscalar_mul(
        points: &[grumpkin::G1Affine],
        scalars: &[grumpkin::Fr],
    ) -> grumpkin::G1Affine {
        let ret: grumpkin::G1 = points
            .par_iter()
            .zip_eq(scalars.par_iter())
            .map(|(p, s)| p * s)
            .sum();

        ret.to_affine()
    }

    #[test]
    fn bn256_it_works() {
        #[cfg(not(debug_assertions))]
        const NPOINTS: usize = 128 * 1024;
        #[cfg(debug_assertions)]
        const NPOINTS: usize = 8 * 1024;

        let points = gen_points::<bn256::G1, bn256::G1Affine>(NPOINTS);
        let scalars = gen_scalars(NPOINTS);

        let naive = bn256_naive_multiscalar_mul(&points, &scalars);

        let stream = CudaStream::create().expect("Failed to create CUDA stream");
        let cfg = default_config(&stream);
        let ret = crate::bn256_msm(&points, &scalars, &cfg).to_affine();

        assert_eq!(ret, naive);
    }

    #[test]
    fn grumpkin_it_works() {
        #[cfg(not(debug_assertions))]
        const NPOINTS: usize = 128 * 1024;
        #[cfg(debug_assertions)]
        const NPOINTS: usize = 8 * 1024;

        let points = gen_points::<grumpkin::G1, grumpkin::G1Affine>(NPOINTS);
        let scalars = gen_scalars(NPOINTS);

        let naive = grumpkin_naive_multiscalar_mul(&points, &scalars);

        let stream = CudaStream::create().expect("Failed to create CUDA stream");
        let cfg = default_config(&stream);
        let ret = crate::grumpkin_msm(&points, &scalars, &cfg).to_affine();

        assert_eq!(ret, naive);
    }
}
