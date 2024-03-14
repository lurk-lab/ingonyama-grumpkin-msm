use halo2curves::{bn256, ff::PrimeField};
use icicle_bn254::curve::{G1Affine, G1Projective, ScalarField};
use icicle_core::{msm, traits::FieldImpl};
use icicle_cuda_runtime::memory::{DeviceVec, HostSlice};
use icicle_cuda_runtime::stream::CudaStream;

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

fn icicle_to_bn256_point(point: &G1Affine) -> bn256::G1Affine {
    bn256::G1Affine {
        x: bn256::Fq::from_repr(point.x.to_bytes_le().try_into().unwrap()).unwrap(),
        y: bn256::Fq::from_repr(point.y.to_bytes_le().try_into().unwrap()).unwrap(),
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
    let points = unsafe { &*(points as *const [bn256::G1Affine] as *const [G1Affine]) };
    let scalars = unsafe { &*(scalars as *const [bn256::Fr] as *const [ScalarField]) };
    let points_host = HostSlice::from_slice(points);
    let scalars_host = HostSlice::from_slice(scalars);

    // Allocate memory on the CUDA device for MSM results
    let mut msm_results = DeviceVec::<G1Projective>::cuda_malloc(1).unwrap();

    // Execute MSM on the device
    let mut msm_host_result = vec![G1Projective::zero(); 1];
    msm::msm(scalars_host, points_host, &cfg, &mut msm_results[..]).expect("Failed to execute MSM");

    msm_results
        .copy_to_host(HostSlice::from_mut_slice(&mut msm_host_result[..]))
        .unwrap();

    icicle_to_bn256_point(&G1Affine::from(msm_host_result[0])).into()
}
