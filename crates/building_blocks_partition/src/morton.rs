// Ported directly from: https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/

pub fn encode_2d(x: u32, y: u32) -> u32 {
    (part_1_by_1(y) << 1) + part_1_by_1(x)
}

pub fn encode_3d(x: u32, y: u32, z: u32) -> u32 {
    (part_1_by_2(z) << 2) + (part_1_by_2(y) << 1) + part_1_by_2(x)
}

// "Insert" a 0 bit after each of the 16 low bits of x
fn part_1_by_1(mut x: u32) -> u32 {
    x &= 0x0000ffff; // x = ---- ---- ---- ---- fedc ba98 7654 3210
    x = (x ^ (x << 8)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
    x = (x ^ (x << 4)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
    x = (x ^ (x << 2)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
    x = (x ^ (x << 1)) & 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    x
}

// "Insert" two 0 bits after each of the 10 low bits of x
fn part_1_by_2(mut x: u32) -> u32 {
    x &= 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210
    x = (x ^ (x << 16)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x ^ (x << 8)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x ^ (x << 4)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x ^ (x << 2)) & 0x09249249; // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    x
}

// Inverse of Part1By1 - "delete" all odd-indexed bits
fn compact_1_by_1(mut x: u32) -> u32 {
    x &= 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    x = (x ^ (x >> 1)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
    x = (x ^ (x >> 2)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
    x = (x ^ (x >> 4)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
    x = (x ^ (x >> 8)) & 0x0000ffff; // x = ---- ---- ---- ---- fedc ba98 7654 3210
    x
}

// Inverse of Part1By2 - "delete" all bits not at positions divisible by 3
fn compact_1_by_2(mut x: u32) -> u32 {
    x &= 0x09249249; // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    x = (x ^ (x >> 2)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x ^ (x >> 4)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x ^ (x >> 8)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x ^ (x >> 16)) & 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210
    x
}

pub fn decode_2x(code: u32) -> u32 {
    compact_1_by_1(code >> 0)
}

pub fn decode_2y(code: u32) -> u32 {
    compact_1_by_1(code >> 1)
}

pub fn decode_3x(code: u32) -> u32 {
    compact_1_by_2(code >> 0)
}

pub fn decode_3y(code: u32) -> u32 {
    compact_1_by_2(code >> 1)
}

pub fn decode_3z(code: u32) -> u32 {
    compact_1_by_2(code >> 2)
}

pub fn decode_2d(code: u32) -> [u32; 2] {
    [decode_2x(code), decode_2y(code)]
}

pub fn decode_3d(code: u32) -> [u32; 3] {
    [decode_3x(code), decode_3y(code), decode_3z(code)]
}
