use halo2curves::{bn256, ff::PrimeField};
use icicle_bn254::curve::{BaseField, CurveCfg, G1Affine, G1Projective, ScalarCfg, ScalarField};
use icicle_core::{
    curve::Curve,
    msm,
    traits::{FieldImpl, GenerateRandom},
};
use icicle_cuda_runtime::{memory::HostOrDeviceSlice, stream::CudaStream};

use serde::de::DeserializeOwned;
use std::{convert::TryInto, time::Instant};
use std::fs::File;
use std::io::BufReader;

/// Path to the directory where Arecibo data will be stored.
pub static ARECIBO_DATA: &str = ".arecibo_data";

/// Reads and deserializes data from a specified section and label.
pub fn read_arecibo_data<T: DeserializeOwned>(section: String, label: String) -> Vec<T> {
    let root_dir = home::home_dir()
        .unwrap()
        .join(ARECIBO_DATA);
    let section_path = root_dir.join(section);
    assert!(section_path.exists(), "Section directory does not exist");

    let file_path = section_path.join(label);
    assert!(file_path.exists(), "Data file does not exist");

    let file = File::open(file_path).expect("Failed to open data file");
    let reader = BufReader::new(file);

    bincode::deserialize_from(reader).expect("Failed to read data")
}

fn halo_to_icicle_scalar(scalar: &bn256::Fr) -> ScalarField {
    ScalarField::from_bytes_le(scalar.to_bytes().as_ref().try_into().unwrap())
}

fn halo_to_icicle_point(point: &bn256::G1Affine) -> G1Affine {
    let x = BaseField::from_bytes_le(point.x.to_bytes().as_ref().try_into().unwrap());
    let y = BaseField::from_bytes_le(point.y.to_bytes().as_ref().try_into().unwrap());

    G1Affine { x, y }
}

fn icicle_to_bn256_point(point: &G1Affine) -> bn256::G1Affine {
    bn256::G1Affine {
        x: bn256::Fq::from_repr(point.x.to_bytes_le().try_into().unwrap()).unwrap(),
        y: bn256::Fq::from_repr(point.y.to_bytes_le().try_into().unwrap()).unwrap(),
    }
}

pub fn bn256_msm(points: &[bn256::G1Affine], scalars: &[bn256::Fr]) -> bn256::G1 {
    let start = Instant::now();

    // Wrap points and scalars in HostOrDeviceSlice for MSM
    let points = points
        .into_iter()
        .map(|p| halo_to_icicle_point(p))
        .collect::<Vec<_>>();
    let scalars = scalars
        .into_iter()
        .map(|x| halo_to_icicle_scalar(x))
        .collect::<Vec<_>>();
    let points_host = HostOrDeviceSlice::Host(points);
    let scalars_host = HostOrDeviceSlice::Host(scalars);

    let transfer = start.elapsed();
    println!("cloning points and scalars: {:?}", transfer);

    // Create a CUDA stream for asynchronous execution
    let stream = CudaStream::create().expect("Failed to create CUDA stream");

    // Allocate memory on the CUDA device for MSM results
    let mut msm_results: HostOrDeviceSlice<'_, G1Projective> =
        HostOrDeviceSlice::cuda_malloc(1).expect("Failed to allocate CUDA memory for MSM results");

    let malloc = start.elapsed();
    println!("malloc result memory: {:?}", malloc - transfer);

    let mut cfg = msm::MSMConfig::default();
    cfg.ctx.stream = &stream;
    cfg.is_async = true; // Enable asynchronous execution

    // Execute MSM on the device
    let mut msm_host_result = vec![G1Projective::zero(); 1];
    msm::msm(&scalars_host, &points_host, &cfg, &mut msm_results).expect("Failed to execute MSM");

    msm_results
        .copy_to_host(&mut msm_host_result[..])
        .unwrap();

    let msm = start.elapsed();
    println!("compute end-to-end msm: {:?}", msm - malloc);

    icicle_to_bn256_point(&G1Affine::from(msm_host_result[0])).into()
}
