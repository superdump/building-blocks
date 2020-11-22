use building_blocks::{
    core::prelude::*,
    storage::{Array3, BytesCompression, Compression, FastArrayCompression, Lz4, Snappy},
    vox::{decode_vox, dot_vox::load, VoxColor},
};

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let vox_path = &args[1];

    println!("Loading {}", vox_path);

    let vox_data = load(vox_path).unwrap();
    let vox_array = decode_vox(&vox_data, 0);

    println!("Compressing with Snappy: \n");
    measure_compression_rate(Snappy, &vox_array);

    println!("Compressing with LZ4: \n");
    measure_compression_rate(Lz4 { level: 10 }, &vox_array);
}

fn measure_compression_rate<B: BytesCompression>(
    bytes_compression: B,
    vox_array: &Array3<VoxColor>,
) {
    let source_size_bytes = vox_array.extent().num_points() * std::mem::size_of::<VoxColor>();

    let compression = FastArrayCompression::new(bytes_compression);
    let compressed_array = compression.compress(vox_array);

    let compressed_size_bytes = compressed_array.compressed_data.compressed_bytes().len();

    println!(
        "source = {} bytes, compressed = {} bytes; rate = {:.1}%\n",
        source_size_bytes,
        compressed_size_bytes,
        100.0 * (compressed_size_bytes as f32 / source_size_bytes as f32)
    );
}
