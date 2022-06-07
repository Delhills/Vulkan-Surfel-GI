#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable

#include "raycommon.glsl"
#include "helpers.glsl"
#include "surfelGIutils.glsl"

layout (location = 0) rayPayloadInEXT hitPayload prd;
layout (location = 1) rayPayloadEXT bool shadowed;
hitAttributeEXT vec3 attribs;

layout (set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;
layout (set = 0, binding = 4) buffer Lights { Light lights[]; } lightsBuffer;
layout (set = 0, binding = 5, scalar) buffer Vertices { Vertex v[]; } vertices[];
layout (set = 0, binding = 6) buffer Indices { int i[]; } indices[];
layout (set = 0, binding = 7) uniform sampler2D[] textures;
layout (set = 0, binding = 8) buffer sceneBuffer { vec4 idx[]; } objIndices;
layout (set = 0, binding = 9) buffer MaterialBuffer { Material mat[]; } materials;
layout (set = 0, binding = 10) uniform sampler2D[] environmentTexture;
layout (set = 0, binding = 11, scalar) buffer Matrices { mat4 m[]; } matrices;


layout (set = 0, binding = 13) buffer SurfelBuffer {
	Surfel surfelInBuffer[];
} surfels;

layout (set = 0, binding = 14) buffer StatsBuffer {uint stats[8];} statsBuffer;

layout (set = 0, binding = 15) buffer GridBuffer {
	SurfelGridCell cells[];
} gridcells;
layout (set = 0, binding = 16) buffer CellBuffer {uint indexSurf[];} surfelcells;

layout (set = 0, binding = 17) buffer SurfelDataBuffer {
	SurfelData surfelDataInBuffer[];
} surfelsData;

void main()
{
  // Do all vertices, indices and barycentrics calculations
 
  prd = hitPayload(vec4(1.0, 1.0, 0.0, 0.0), prd.seed);
}