use std::collections::{HashMap, HashSet};

use bevy::{
    input::{
        mouse::{MouseMotion, MouseWheel},
        system::exit_on_esc_system,
    },
    prelude::*,
};
use building_blocks_core::{Extent3i, PointN};
use building_blocks_partition::{
    esvo::ESVO, morton::decode_3d, octree::VisitStatus, Octant, OctreeVisitor,
};
use building_blocks_storage::{Array, Array3, Get, GetMut, IsEmpty};
use noise::{MultiFractal, NoiseFn, RidgedMulti, Seedable};

struct ESVOTree {
    octree: ESVO,
    current_index: u32,
    array: Array3<Voxel>,
}

impl Default for ESVOTree {
    fn default() -> Self {
        Self {
            octree: ESVO::default(),
            current_index: 0,
            array: Array3::fill(
                Extent3i::from_min_and_shape(PointN([0, 0, 0]), PointN([2, 2, 2])),
                Voxel(false),
            ),
        }
    }
}

#[derive(Debug, Default)]
struct ESVOMeshes {
    material: Handle<StandardMaterial>,
    entities: HashMap<Octant, Entity>,
}

fn main() {
    App::build()
        .add_resource(ClearColor(Color::BLACK))
        .init_resource::<ESVOTree>()
        .init_resource::<ESVOMeshes>()
        .init_resource::<InputState>()
        .add_plugins(DefaultPlugins)
        .add_startup_system(setup_world.system())
        .add_startup_system(create_octree.system())
        .add_startup_system(init_material.system())
        .add_startup_system(spawn_array_voxels.system())
        .add_system(exit_on_esc_system.system())
        .add_system(pan_orbit_camera.system())
        // .add_system(insert_voxel.system())
        .add_system(update_meshes.system())
        .run();
}

fn setup_world(mut commands: Commands) {
    commands
        .spawn(Camera3dComponents {
            transform: Transform::from_matrix(Mat4::face_toward(
                Vec3::new(0.0, 0.0, 40.0),
                Vec3::zero(),
                Vec3::unit_y(),
            )),
            ..Default::default()
        })
        .with(PanOrbitCamera::default())
        .spawn(LightComponents {
            transform: Transform::from_matrix(Mat4::face_toward(
                Vec3::new(-30.0, 30.0, -30.0),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::unit_y(),
            )),
            ..Default::default()
        });
}

fn init_material(mut materials: ResMut<Assets<StandardMaterial>>, mut esvo: ResMut<ESVOMeshes>) {
    let transparent = materials.add(Color::rgb(1.0, 1.0, 1.0).into());
    esvo.material = transparent;
}

#[derive(Clone)]
pub struct Voxel(bool);

impl IsEmpty for Voxel {
    fn is_empty(&self) -> bool {
        !self.0
    }
}

fn create_octree(mut esvo: ResMut<ESVOTree>) {
    let n: usize = 32;
    let extent =
        Extent3i::from_min_and_shape(PointN([0, 0, 0]), PointN([n as i32, n as i32, n as i32]));
    let mut array = Array3::fill(extent, Voxel(false));
    // for z in 0..n {
    //     for y in 0..n {
    //         for x in 0..n {
    //             // let y_max = heightmap[z][x];
    //             if (x + y + z) < n {
    //                 *array.get_mut(array.stride_from_local_point(
    //                     &building_blocks_storage::array::Local(PointN([
    //                         x as i32, y as i32, z as i32,
    //                     ])),
    //                 )) = Voxel(true);
    //             }
    //         }
    //     }
    // }
    let noise = RidgedMulti::new()
        .set_seed(1234)
        .set_frequency(0.08)
        .set_octaves(5);
    let yoffset = n as f64 * 0.5;
    let yscale = 0.8 * yoffset;
    let heightmap: Vec<Vec<i32>> = (0..n)
        .map(|z| {
            (0..n)
                .map(|x| (noise.get([x as f64, z as f64]) * yscale + yoffset).round() as i32)
                .collect()
        })
        .collect();
    for z in 0..n {
        for x in 0..n {
            let y_max = heightmap[z][x];
            for y in 0..y_max {
                *array.get_mut(array.stride_from_local_point(
                    &building_blocks_storage::array::Local(PointN([x as i32, y as i32, z as i32])),
                )) = Voxel(true);
            }
        }
    }
    esvo.octree = ESVO::from_array3(&array, extent);
    esvo.array = array;
}

fn spawn_array_voxels(
    mut commands: Commands,
    esvo: Res<ESVOTree>,
    esvo_meshes: Res<ESVOMeshes>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let mesh = meshes.add(Mesh::from(shape::Cube { size: 0.5 }));
    let array_extent = esvo.array.extent();
    esvo.array.for_each_point_and_stride(array_extent, |p, s| {
        if !esvo.array.get(s).is_empty() {
            commands.spawn(PbrComponents {
                material: esvo_meshes.material.clone(),
                mesh: mesh.clone(),
                transform: Transform::from_translation(Vec3::new(
                    p.x() as f32,
                    p.y() as f32,
                    p.z() as f32,
                )),
                draw: Draw {
                    is_transparent: true,
                    ..Default::default()
                },
                ..Default::default()
            });
        }
    });
}

fn insert_voxel(input: Res<Input<KeyCode>>, mut esvo: ResMut<ESVOTree>) {
    if input.just_pressed(KeyCode::N) {
        let index = esvo.current_index;
        let p = decode_3d(index);
        esvo.octree
            .insert(PointN([p[0] as i32, p[1] as i32, p[2] as i32]));
        esvo.current_index += 1;
    }
}

fn update_meshes(
    mut commands: Commands,
    esvo: ChangedRes<ESVOTree>,
    mut esvo_meshes: ResMut<ESVOMeshes>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let (new_entities, removed_entities) = {
        let mut visitor = ESVOMeshesVisitor {
            entities: &esvo_meshes.entities,
            removed_entities: esvo_meshes.entities.keys().cloned().collect(),
            new_entities: HashSet::new(),
        };
        esvo.octree.visit(&mut visitor);
        (visitor.new_entities, visitor.removed_entities)
    };
    let offset = Vec3::new((esvo.array.extent().shape.x() + 10) as f32, 0.0, 0.0);
    for octant in &new_entities {
        let size = 0.5 * octant.edge_length as f32;
        let center = octant_center(&octant);
        let mesh = meshes.add(Mesh::from(shape::Cube { size }));
        let entity = commands
            .spawn(PbrComponents {
                material: esvo_meshes.material.clone(),
                mesh: mesh.clone(),
                transform: Transform::from_translation(center - offset),
                draw: Draw {
                    is_transparent: true,
                    ..Default::default()
                },
                ..Default::default()
            })
            .current_entity()
            .expect("Failed to spawn mesh");
        esvo_meshes.entities.insert(*octant, entity);
    }
    for octant in &removed_entities {
        let entity = esvo_meshes
            .entities
            .remove(octant)
            .expect("Failed to remove octant");
        commands.despawn(entity);
    }
}

fn octant_center(octant: &Octant) -> Vec3 {
    let offset = 0.5 * octant.edge_length as f32;
    Vec3::new(
        octant.minimum.x() as f32 + offset,
        octant.minimum.y() as f32 + offset,
        octant.minimum.z() as f32 + offset,
    )
}

pub struct ESVOMeshesVisitor<'a> {
    pub entities: &'a HashMap<Octant, Entity>,
    pub removed_entities: HashSet<Octant>,
    pub new_entities: HashSet<Octant>,
}

impl<'a> OctreeVisitor for ESVOMeshesVisitor<'a> {
    fn visit_octant(&mut self, octant: Octant, is_leaf: bool) -> VisitStatus {
        if is_leaf {
            if self.entities.contains_key(&octant) {
                self.removed_entities.remove(&octant);
            } else {
                self.new_entities.insert(octant);
            }
        }
        VisitStatus::Continue
    }
}

/// Tags an entity as capable of panning and orbiting.
struct PanOrbitCamera {
    /// The "focus point" to orbit around. It is automatically updated when panning the camera
    pub focus: Vec3,
}

impl Default for PanOrbitCamera {
    fn default() -> Self {
        PanOrbitCamera {
            focus: Vec3::zero(),
        }
    }
}

/// Hold readers for events
#[derive(Default)]
struct InputState {
    pub reader_motion: EventReader<MouseMotion>,
    pub reader_scroll: EventReader<MouseWheel>,
}

/// Pan the camera with LHold or scrollwheel, orbit with rclick.
fn pan_orbit_camera(
    time: Res<Time>,
    windows: Res<Windows>,
    mut state: ResMut<InputState>,
    ev_motion: Res<Events<MouseMotion>>,
    mousebtn: Res<Input<MouseButton>>,
    ev_scroll: Res<Events<MouseWheel>>,
    mut query: Query<(&mut PanOrbitCamera, &mut Transform)>,
) {
    let mut translation = Vec2::zero();
    let mut rotation_move = Vec2::default();
    let mut scroll = 0.0;
    let dt = time.delta_seconds;

    if mousebtn.pressed(MouseButton::Right) {
        for ev in state.reader_motion.iter(&ev_motion) {
            rotation_move += ev.delta;
        }
    } else if mousebtn.pressed(MouseButton::Left) {
        // Pan only if we're not rotating at the moment
        for ev in state.reader_motion.iter(&ev_motion) {
            translation += ev.delta;
        }
    }

    for ev in state.reader_scroll.iter(&ev_scroll) {
        scroll += ev.y;
    }

    // Either pan+scroll or arcball. We don't do both at once.
    for (mut camera, mut trans) in query.iter_mut() {
        if rotation_move.length_squared() > 0.0 {
            let window = windows.get_primary().unwrap();
            let window_w = window.width() as f32;
            let window_h = window.height() as f32;

            // Link virtual sphere rotation relative to window to make it feel nicer
            let delta_x = rotation_move.x() / window_w * std::f32::consts::PI * 2.0;
            let delta_y = rotation_move.y() / window_h * std::f32::consts::PI;

            let delta_yaw = Quat::from_rotation_y(delta_x);
            let delta_pitch = Quat::from_rotation_x(delta_y);

            trans.translation =
                delta_yaw * delta_pitch * (trans.translation - camera.focus) + camera.focus;

            let look = Mat4::face_toward(trans.translation, camera.focus, Vec3::new(0.0, 1.0, 0.0));
            trans.rotation = look.to_scale_rotation_translation().1;
        } else {
            // The plane is x/y while z is "up". Multiplying by dt allows for a constant pan rate
            let mut translation = Vec3::new(-translation.x() * dt, translation.y() * dt, 0.0);
            camera.focus += translation;
            *translation.z_mut() = -scroll;
            trans.translation += translation;
        }
    }
}
