use crate::{octree::VisitStatus, Octant, OctreeVisitor};
use building_blocks_core::{
    ComponentwiseIntegerOps, Extent3i, IntegerPoint, Neighborhoods, Point3i, PointN,
};
use building_blocks_storage::{access::GetUncheckedRelease, Array, IsEmpty, Local, Stride};

pub struct ESVO {
    pub extent: Extent3i,
    pub children: Vec<u32>,
}

impl Default for ESVO {
    fn default() -> Self {
        ESVO::new()
    }
}

struct Layers {
    children: Vec<Vec<u32>>,
}

impl Layers {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            children: Vec::with_capacity(capacity),
        }
    }
}

impl Default for Layers {
    fn default() -> Self {
        Self {
            children: Vec::new(),
        }
    }
}

impl ESVO {
    pub fn new() -> Self {
        Self {
            extent: Extent3i::from_min_and_shape(PointN([0, 0, 0]), PointN([0, 0, 0])),
            children: Vec::new(),
        }
    }

    /// Constructs an `Octree` which contains all of the points in `extent` which are not empty (as
    /// defined by the `IsEmpty` trait). `extent` must be cube-shaped with edge length being a power
    /// of 2. For exponent E where edge length is 2^E, we must have `0 < E <= 6`, because there is a
    /// maximum fixed depth of the octree.
    pub fn from_array3<A, T>(array: &A, extent: Extent3i) -> Self
    where
        A: Array<[i32; 3]> + GetUncheckedRelease<Stride, T>,
        T: Clone + IsEmpty,
    {
        assert!(extent.shape.dimensions_are_powers_of_2());
        assert!(extent.shape.is_cube());

        let power = extent.shape.x().trailing_zeros();
        let edge_len = 1 << power;

        // These are the corners of the root octant, in local coordinates.
        let corner_offsets: Vec<_> = Point3i::corner_offsets()
            .into_iter()
            .map(|p| Local(p * edge_len))
            .collect();
        // Convert into strides for indexing efficiency.
        let mut corner_strides = [Stride(0); 8];
        array.strides_from_local_points(&corner_offsets, &mut corner_strides);

        let mut layers = Layers::with_capacity(power as usize);
        let min_local = Local(extent.minimum - array.extent().minimum);
        let root_minimum = array.stride_from_local_point(&min_local);
        let (root_exists, _full) = Self::partition_array(
            extent,
            root_minimum,
            edge_len,
            &corner_strides,
            array,
            0,
            &mut layers,
        );
        assert!(root_exists);

        let mut children = Vec::new();
        for i in 0..layers.children.len() {
            let mut offset = layers.children[i].len();
            for node in &mut layers.children[i] {
                if node.has_non_leaf_children() {
                    node.set_child_offset(node.child_offset() + offset as i16);
                }
                offset -= 1;
            }
            children.append(&mut layers.children[i]);
        }
        Self {
            extent,
            children,
            ..Default::default()
        }
    }

    fn partition_array<A, T>(
        extent: Extent3i,
        minimum: Stride,
        edge_len: i32,
        corner_strides: &[Stride],
        array: &A,
        layer: usize,
        layers: &mut Layers,
    ) -> (bool, bool)
    where
        A: Array<[i32; 3]> + GetUncheckedRelease<Stride, T>,
        T: Clone + IsEmpty,
    {
        if layer == layers.children.len() {
            layers.children.push(Vec::new());
        }

        // Base case where the octant is a single voxel.
        if edge_len == 1 {
            let exists = !array.get_unchecked_release(minimum).is_empty();
            return (exists, exists);
        }

        let mut octant_corner_strides = [Stride(0); 8];
        for (child_corner, parent_corner) in
            octant_corner_strides.iter_mut().zip(corner_strides.iter())
        {
            *child_corner = Stride(parent_corner.0 >> 1);
        }

        let half_edge_len = edge_len >> 1;
        let mut leaf_bitmask = ChildMask(0);
        let mut child_bitmask = ChildMask(0);
        let child_layer_index = if layer + 1 >= layers.children.len() {
            0
        } else {
            assert!(layers.children[layer + 1].len() < std::i16::MAX as usize);
            layers.children[layer + 1].len() as i16
        };
        for (child_octant, offset) in octant_corner_strides.iter().enumerate() {
            let octant_min = minimum + *offset;
            let octant_extent = octant_to_extent(&extent, child_octant as u8);
            let (has_child, is_leaf) = Self::partition_array(
                octant_extent,
                octant_min,
                half_edge_len,
                &octant_corner_strides,
                array,
                layer + 1,
                layers,
            );
            if has_child {
                child_bitmask.add_child(child_octant);
            }
            if is_leaf {
                leaf_bitmask.add_child(child_octant);
            }
        }

        let has_children = !child_bitmask.is_empty();
        let is_leaf = leaf_bitmask.is_full();

        if has_children && (!is_leaf || layer == 0) {
            layers.children[layer].push(child_descriptor(
                Some(leaf_bitmask),
                Some(child_bitmask),
                Some(child_layer_index as i16),
            ));
        }

        (has_children, is_leaf)
    }

    pub fn insert(&mut self, minimum: Point3i) {
        if self.children.is_empty() {
            let (extent, child_descriptor) = initial_voxel(minimum);
            self.children.push(child_descriptor);
            self.extent = extent;
            return;
        }
        while !self.extent.contains(&minimum) {
            // expand with 7 new octants in the direction of minimum
            let octant_dir = octant_direction(&self.extent.minimum, &minimum);
            let new_extent = grow_extent_in_dir(&self.extent, octant_dir);
            self.extent = new_extent;

            // Add original as child of this
            let old_octant = (!octant_dir) & 0x7;
            self.children.insert(
                0,
                child_descriptor(
                    if self.children.is_empty() {
                        Some(ChildMask((1 << old_octant) as u8))
                    } else {
                        None
                    },
                    Some(ChildMask((1 << old_octant) as u8)),
                    Some(1i16),
                ),
            );
            if self.children[1].is_full() {
                self.children[0].add_leaf(old_octant as usize);
                self.children[0].set_child_offset(0);
                self.children.remove(1);
            }
        }
        // find the octant that contains the voxel at minimum
        let mut parents = vec![(0, self.extent)];
        let mut parent_extent = self.extent;
        let mut index = 0;
        loop {
            let edge_length = parent_extent.shape.x();
            if edge_length == 1 {
                panic!("Should not happen!");
            }
            let (octant, octant_extent) = extent_to_octant(&parent_extent, &minimum);
            if edge_length == 2 {
                // base case: add the new voxel as a child and leaf
                self.children[index].add_leaf(octant as usize);
                if self.children[index].is_full() {
                    self.collapse_child_octant(&mut parents, index, octant_extent);
                }
                return;
            } else if !self.children[index].has_child(octant as usize) {
                // create child octant and add it
                self.create_child_octant(index, octant);
                self.children[index].add_child(octant as usize);
            }
            // traverse into child octant
            parents.push((index, octant_extent));
            parent_extent = octant_extent;
            index = add_child_offset(index, self.children[index].child_offset())
                + if octant_extent.shape.x() <= 2 {
                    0
                } else {
                    self.children[index].child_index(octant as usize)
                };
        }
    }

    pub fn create_child_octant(&mut self, index: usize, octant: u8) {
        let (child_offset, mut insert) = if self.children[index].child_offset() == 0 {
            ((self.children.len() - index) as i16, false)
        } else {
            (self.children[index].child_offset(), true)
        };
        let child_index = add_child_offset(index, child_offset)
            + self.children[index].child_index(octant as usize);
        if child_index >= self.children.len() {
            insert = false;
        }
        let child_descriptor = child_descriptor(None, None, None);
        if insert {
            self.children.insert(child_index, child_descriptor);
            for i in 0..child_index {
                let child_offset_i = self.children[i].child_offset();
                if add_child_offset(i, child_offset_i) >= child_index {
                    self.children[i].set_child_offset(child_offset_i + 1);
                }
            }
        } else {
            self.children.push(child_descriptor);
        }
        self.children[index].set_child_offset(child_offset);
    }

    pub fn collapse_child_octant(
        &mut self,
        parents: &mut Vec<(usize, Extent3i)>,
        index: usize,
        extent: Extent3i,
    ) {
        if index == 0 {
            return;
        }
        let (parent_index, parent_extent) = parents.pop().expect("Failed to pop parent");
        let (octant, _extent) = extent_to_octant(&parent_extent, &extent.minimum);
        self.children[parent_index].add_leaf(octant as usize);
        // Remove full child
        self.children.remove(index);
        // Update indices
        for i in 0..index {
            let child_offset = self.children[i].child_offset();
            if add_child_offset(i, child_offset) >= index {
                self.children[i].set_child_offset(child_offset - 1);
            }
        }
        if parent_index > 0 && self.children[parent_index].is_full() {
            self.collapse_child_octant(parents, parent_index, parent_extent);
        }
    }

    /// Visit every non-empty octant of the octree.
    pub fn visit(&self, visitor: &mut impl OctreeVisitor) -> VisitStatus {
        if self.children.is_empty() {
            return VisitStatus::Continue;
        }

        let minimum = self.extent.minimum;
        let edge_len = self.extent.shape.x();
        let corner_offsets: Vec<_> = Point3i::corner_offsets()
            .into_iter()
            .map(|p| p * edge_len)
            .collect();

        self._visit(0, minimum, edge_len, &corner_offsets, visitor)
    }

    fn _visit(
        &self,
        index: usize,
        minimum: Point3i,
        edge_length: i32,
        corner_offsets: &[Point3i],
        visitor: &mut impl OctreeVisitor,
    ) -> VisitStatus {
        let octant = Octant {
            minimum,
            edge_length,
        };

        // VISIT THIS NODE

        // Base case where the octant is a single leaf voxel.
        if edge_length == 1 {
            return visitor.visit_octant(octant, true);
        }

        // Continue traversal of this branch.

        // Definitely not at a leaf node.
        let status = visitor.visit_octant(octant, false);
        if status != VisitStatus::Continue {
            return status;
        }

        // VISIT THIS NODE'S CHILDREN

        if index >= self.children.len() {
            panic!("Tried to visit a node with edge_len != 1 that was outside the known nodes");
        }
        let child_descriptor = &self.children[index];

        let mut octant_corner_offsets = [PointN([0; 3]); 8];
        for (child_corner, parent_corner) in
            octant_corner_offsets.iter_mut().zip(corner_offsets.iter())
        {
            *child_corner = parent_corner.scalar_right_shift(1);
        }

        let half_edge_length = edge_length >> 1;
        let mut child_offset = child_descriptor.child_offset();
        for (octant, offset) in octant_corner_offsets.iter().enumerate() {
            let octant_min = minimum + *offset;
            if child_descriptor.has_leaf(octant) {
                let status = visitor.visit_octant(
                    Octant {
                        minimum: octant_min,
                        edge_length: half_edge_length,
                    },
                    true,
                );
                if status != VisitStatus::Continue {
                    return status;
                }
            } else if child_descriptor.has_child(octant) {
                if self._visit(
                    add_child_offset(index, child_offset),
                    octant_min,
                    half_edge_length,
                    &octant_corner_offsets,
                    visitor,
                ) == VisitStatus::ExitEarly
                {
                    return VisitStatus::ExitEarly;
                }
                // child_offset is only incremented for child descriptors that exist
                child_offset += 1;
            }
        }

        // Continue with the rest of the tree.
        VisitStatus::Continue
    }
}

impl std::fmt::Debug for ESVO {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "")?;
        for (i, desc) in self.children.iter().enumerate() {
            writeln!(f, "{}: {:?}", i, desc)?;
        }
        Ok(())
    }
}

fn initial_voxel(minimum: Point3i) -> (Extent3i, u32) {
    let parent_extent = Extent3i::from_min_and_shape(minimum, PointN([2, 2, 2]));
    let child_mask = ChildMask(1 << extent_to_octant(&parent_extent, &minimum).0);
    (
        parent_extent,
        child_descriptor(Some(child_mask), Some(child_mask), None),
    )
}

fn add_child_offset(index: usize, child_offset: i16) -> usize {
    (index as isize + child_offset as isize) as usize
}

pub fn octant_direction(center: &Point3i, point: &Point3i) -> u8 {
    let mut octant = 0;
    if point.x() >= center.x() {
        octant |= 1 << 0;
    }
    if point.y() >= center.y() {
        octant |= 1 << 1;
    }
    if point.z() >= center.z() {
        octant |= 1 << 2;
    }
    octant
}

/// Gets the octant and octant extent within the extent
pub fn extent_to_octant(parent: &Extent3i, point: &Point3i) -> (u8, Extent3i) {
    let center = extent_center(parent);
    let octant = octant_direction(&center, point);
    (octant, octant_to_extent(parent, octant))
}

pub fn extent_center(extent: &Extent3i) -> Point3i {
    extent.minimum + extent.shape.scalar_right_shift(1)
}

pub fn octant_to_extent(parent: &Extent3i, octant: u8) -> Extent3i {
    let shape = parent.shape.scalar_right_shift(1);
    let minimum = match octant {
        0 => parent.minimum,
        1 => parent.minimum + PointN([1, 0, 0]) * shape.x(),
        2 => parent.minimum + PointN([0, 1, 0]) * shape.x(),
        3 => parent.minimum + PointN([1, 1, 0]) * shape.x(),
        4 => parent.minimum + PointN([0, 0, 1]) * shape.x(),
        5 => parent.minimum + PointN([1, 0, 1]) * shape.x(),
        6 => parent.minimum + PointN([0, 1, 1]) * shape.x(),
        7 => parent.minimum + PointN([1, 1, 1]) * shape.x(),
        _ => panic!("Invalid octant"),
    };
    Extent3i::from_min_and_shape(minimum, shape)
}

pub fn grow_extent_in_dir(extent: &Extent3i, octant_direction: u8) -> Extent3i {
    let mut new_extent = *extent;

    if octant_direction & (1 << 0) == 0 {
        *new_extent.minimum.x_mut() -= new_extent.shape.x();
    }
    if octant_direction & (1 << 1) == 0 {
        *new_extent.minimum.y_mut() -= new_extent.shape.y();
    }
    if octant_direction & (1 << 2) == 0 {
        *new_extent.minimum.z_mut() -= new_extent.shape.z();
    }

    new_extent.shape = new_extent.shape.scalar_left_shift(1);

    new_extent
}

const LEAF_MASK_BITS: u32 = 8;
const CHILD_MASK_BITS: u32 = 8;
// const CHILD_OFFSET_BITS: u32 = 16;

const CHILDREN_SHIFT_BITS: u32 = LEAF_MASK_BITS;
const CHILD_OFFSET_SHIFT_BITS: u32 = LEAF_MASK_BITS + CHILD_MASK_BITS;

const BIT_MASK_8: u32 = 0xff;
const BIT_MASK_16: u32 = 0xffff;

// [0..7] leaf_mask - octant bit mask indicating the child octant is a leaf
// [8..15] valid_mask - octant bit mask indicating the child octant has some of its volume filled
// [16..31] relative index of child descriptors
pub trait ChildDescriptor {
    fn leaves(&self) -> ChildMask;
    fn set_leaves(&mut self, leaves: ChildMask);
    fn children(&self) -> ChildMask;
    fn set_children(&mut self, children: ChildMask);
    fn child_offset(&self) -> i16;
    fn set_child_offset(&mut self, child_offset: i16);
    fn add_leaf(&mut self, index: usize);
    fn remove_leaf(&mut self, index: usize);
    fn has_leaf(&self, index: usize) -> bool;
    fn is_full(&self) -> bool;
    fn has_children(&self) -> bool;
    fn has_non_leaf_children(&self) -> bool;
    fn has_child(&self, index: usize) -> bool;
    fn child_index(&self, index: usize) -> usize;
    fn add_child(&mut self, index: usize);
    fn remove_child(&mut self, index: usize);
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
    fn print(&self) -> String;
}

fn child_descriptor(
    leaves: Option<ChildMask>,
    children: Option<ChildMask>,
    child_offset: Option<i16>,
) -> u32 {
    let mut child_descriptor = u32::default();
    if let Some(leaves) = leaves {
        child_descriptor.set_leaves(leaves);
    }
    if let Some(children) = children {
        child_descriptor.set_children(children);
    }
    if let Some(child_offset) = child_offset {
        child_descriptor.set_child_offset(child_offset);
    }
    child_descriptor
}

impl ChildDescriptor for u32 {
    fn leaves(&self) -> ChildMask {
        ChildMask((self & BIT_MASK_8) as u8)
    }

    fn set_leaves(&mut self, leaves: ChildMask) {
        // !BIT_MASK_8 makes 0x000000ff into 0xffffff00
        // and-ing the current value with this zeros out the bits at BIT_MASK_8
        // or-ing this with the value sets the relevant bits
        *self = (*self & !BIT_MASK_8) | leaves.0 as u32;
    }

    fn children(&self) -> ChildMask {
        ChildMask(((self >> CHILDREN_SHIFT_BITS) & BIT_MASK_8) as u8)
    }

    fn set_children(&mut self, children: ChildMask) {
        // !(BIT_MASK_8 << CHILDREN_SHIFT_BITS) makes 0x0000ff00 into 0xffff00ff
        // and-ing the current value with this zeros out the bits at BIT_MASK_8 << CHILDREN_SHIFT_BITS
        // or-ing this with the (value << CHILDREN_SHIFT_BITS) sets the relevant bits
        *self = (*self & !(BIT_MASK_8 << CHILDREN_SHIFT_BITS))
            | (children.0 as u32).overflowing_shl(CHILDREN_SHIFT_BITS).0;
    }

    fn child_offset(&self) -> i16 {
        ((self >> CHILD_OFFSET_SHIFT_BITS) & BIT_MASK_16) as i16
    }

    fn set_child_offset(&mut self, child_offset: i16) {
        // !(BIT_MASK_16 << CHILD_OFFSET_SHIFT_BITS) makes 0xffff0000 into 0x0000ffff
        // and-ing the current value with this zeros out the bits at BIT_MASK_16 << CHILD_OFFSET_SHIFT_BITS
        // or-ing this with the (value << CHILD_OFFSET_SHIFT_BITS) sets the relevant bits
        *self = (*self & !(BIT_MASK_16 << CHILD_OFFSET_SHIFT_BITS))
            | (child_offset as u32)
                .overflowing_shl(CHILD_OFFSET_SHIFT_BITS)
                .0;
    }

    fn add_leaf(&mut self, index: usize) {
        *self |= 1 << index;
        self.add_child(index);
    }

    fn remove_leaf(&mut self, index: usize) {
        *self &= !(1 << index);
    }

    fn has_leaf(&self, index: usize) -> bool {
        self & (1 << index) != 0
    }

    fn is_full(&self) -> bool {
        self & BIT_MASK_16 == BIT_MASK_16
    }

    fn has_children(&self) -> bool {
        self & (BIT_MASK_8 << CHILDREN_SHIFT_BITS) != 0
    }

    fn has_non_leaf_children(&self) -> bool {
        for i in 0..8 {
            if self.has_child(i) && !self.has_leaf(i) {
                return true;
            }
        }
        false
    }

    fn has_child(&self, index: usize) -> bool {
        self & (1 << (CHILDREN_SHIFT_BITS as usize + index)) != 0
    }

    fn child_index(&self, index: usize) -> usize {
        let mut child_index = 0;
        for i in 0..index {
            if self.has_child(i) && !self.has_leaf(i) {
                child_index += 1;
            }
        }
        child_index
    }

    fn add_child(&mut self, index: usize) {
        *self |= 1 << (CHILDREN_SHIFT_BITS as usize + index);
    }

    fn remove_child(&mut self, index: usize) {
        *self &= !(1 << (CHILDREN_SHIFT_BITS as usize + index));
    }

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChildDescriptor")
            .field("child_offset", &self.child_offset())
            .field("children", &self.children())
            .field("leaves", &self.leaves())
            .finish()
    }

    fn print(&self) -> String {
        format!(
            "ChildDescriptor {{ leaves: {:?}, children: {:?}, child_offset: {} }}",
            self.leaves(),
            self.children(),
            self.child_offset()
        )
    }
}

#[derive(Clone, Copy, Default, PartialEq)]
pub struct ChildMask(u8);

impl ChildMask {
    pub fn is_empty(&self) -> bool {
        self.0 == 0
    }

    pub fn is_full(&self) -> bool {
        self.0 & BIT_MASK_8 as u8 == BIT_MASK_8 as u8
    }

    pub fn has_child(&self, index: usize) -> bool {
        self.0 & (1 << index) != 0
    }

    pub fn child_index(&self, index: usize) -> usize {
        let mut child_index = 0;
        for i in 0..index {
            if self.has_child(i) {
                child_index += 1;
            }
        }
        child_index
    }

    pub fn add_child(&mut self, index: usize) {
        self.0 |= 1 << index;
    }

    pub fn remove_child(&mut self, index: usize) {
        self.0 &= !(1 << index);
    }
}

impl std::fmt::Debug for ChildMask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#b}", self.0)
    }
}

impl std::ops::BitAnd for ChildMask {
    type Output = ChildMask;

    fn bitand(self, rhs: Self) -> Self::Output {
        ChildMask(self.0 & rhs.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_voxel() {
        let mut octree = ESVO::new();
        let p = PointN([1, -2, 3]);
        octree.insert(p);
        assert_eq!(
            octree.extent,
            Extent3i::from_min_and_shape(p, PointN([2, 2, 2]))
        );
        assert_eq!(
            octree.children[0],
            child_descriptor(Some(ChildMask(1)), Some(ChildMask(1)), None,),
        );
    }

    #[test]
    fn test_corners() {
        let mut octree = ESVO::new();
        let points: Vec<Point3i> = Point3i::corner_offsets()
            .iter_mut()
            .map(|p| p.scalar_left_shift(1))
            .collect();
        for p in &points {
            octree.insert(*p);
        }
        assert_eq!(
            octree.extent,
            Extent3i::from_min_and_shape(points[0], PointN([4, 4, 4]))
        );
        let child_descriptors = [
            child_descriptor(Some(ChildMask(0)), Some(ChildMask(0b11111111)), Some(1)),
            child_descriptor(Some(ChildMask(1)), Some(ChildMask(1)), Some(0)),
            child_descriptor(Some(ChildMask(1)), Some(ChildMask(1)), Some(0)),
            child_descriptor(Some(ChildMask(1)), Some(ChildMask(1)), Some(0)),
            child_descriptor(Some(ChildMask(1)), Some(ChildMask(1)), Some(0)),
            child_descriptor(Some(ChildMask(1)), Some(ChildMask(1)), Some(0)),
            child_descriptor(Some(ChildMask(1)), Some(ChildMask(1)), Some(0)),
            child_descriptor(Some(ChildMask(1)), Some(ChildMask(1)), Some(0)),
            child_descriptor(Some(ChildMask(1)), Some(ChildMask(1)), Some(0)),
        ];
        for (i, desc) in octree.children.iter().enumerate() {
            assert_eq!(child_descriptors[i], *desc);
        }
    }

    #[test]
    fn test_insert_moves_subtrees() {
        let mut octree = ESVO::new();
        let points = [PointN([0, 0, 0]), PointN([4, 4, 4])];
        for p in &points {
            octree.insert(*p);
        }
        assert_eq!(
            octree.extent,
            Extent3i::from_min_and_shape(points[0], PointN([8, 8, 8]))
        );
        let child_descriptors = [
            child_descriptor(Some(ChildMask(0)), Some(ChildMask(0b10000001)), Some(1)),
            child_descriptor(Some(ChildMask(0)), Some(ChildMask(1)), Some(2)),
            child_descriptor(Some(ChildMask(0)), Some(ChildMask(1)), Some(2)),
            child_descriptor(Some(ChildMask(1)), Some(ChildMask(1)), Some(0)),
            child_descriptor(Some(ChildMask(1)), Some(ChildMask(1)), Some(0)),
        ];
        for (i, desc) in octree.children.iter().enumerate() {
            assert_eq!(child_descriptors[i], *desc);
        }
    }

    #[test]
    fn test_filled_collapse() {
        let mut octree = ESVO::new();
        for i in 0..4 * 4 * 4 {
            let p = crate::morton::decode_3d(i as u32);
            let p = PointN([p[0] as i32, p[1] as i32, p[2] as i32]);
            octree.insert(p);
        }
        assert_eq!(
            octree.extent,
            Extent3i::from_min_and_shape(PointN([0, 0, 0]), PointN([4, 4, 4])),
        );
        assert_eq!(octree.children.len(), 1);
        assert_eq!(
            octree.children[0],
            child_descriptor(Some(ChildMask(0xff)), Some(ChildMask(0xff)), None),
        );
    }
}
