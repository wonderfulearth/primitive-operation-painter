// src/types.rs
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Config {
    pub seed: u32,
    pub step: u32,
    pub num_images: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuShape {
    pub cx: f32,
    pub cy: f32,
    pub hw: f32,
    pub hh: f32,
    pub theta: f32,
    pub shape_type: u32,
    pub r: u32,
    pub g: u32,
    pub b: u32,
    pub delta_err: f32,
    pub pad1: u32,
    pub pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ImageState {
    pub current_ssd: f32,
    pub initial_ssd: f32,
    pub last_delta_err: f32,
    pub is_done: u32,
}
