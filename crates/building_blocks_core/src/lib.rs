//! The core data types for defining 2D and 3D integer lattices:
//! - `PointN`: an N-dimensional point, most importantly `Point2i` and `Point3i`
//! - `ExtentN`: an N-dimensional extent, most importantly `Extent2i` and `Extent3i`

pub mod axis;
pub mod extent;
pub mod extent2;
pub mod extent3;
pub mod point;
pub mod point2;
pub mod point3;
pub mod point_traits;

pub use axis::{Axis2, Axis3, Axis3Permutation, SignedAxis2, SignedAxis3};
pub use extent::{bounding_extent, Extent, ExtentN, IntegerExtent};
pub use extent2::{Extent2, Extent2f, Extent2i};
pub use extent3::{Extent3, Extent3f, Extent3i};
pub use point::PointN;
pub use point2::{Point2, Point2f, Point2i};
pub use point3::{Point3, Point3f, Point3i};
pub use point_traits::{
    Bounded, ComponentwiseIntegerOps, Distance, DotProduct, GetComponent, IntegerPoint,
    MapComponents, Neighborhoods, Norm, NormSquared, Ones, Point, SmallZero,
};

pub use num;

pub mod prelude {
    pub use super::{
        Axis2, Axis3, Bounded, ComponentwiseIntegerOps, Distance, DotProduct, Extent, Extent2,
        Extent2f, Extent2i, Extent3, Extent3f, Extent3i, ExtentN, GetComponent, IntegerExtent,
        IntegerPoint, MapComponents, Neighborhoods, Norm, NormSquared, Ones, Point, Point2,
        Point2f, Point2i, Point3, Point3f, Point3i, PointN, SmallZero,
    };
}
