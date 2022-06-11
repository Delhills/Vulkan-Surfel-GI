#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_GOOGLE_include_directive : enable

#include "raycommon.glsl"
#include "surfelGIutils.glsl"
#include "bitwise.glsl"

layout(location = 0) rayPayloadInEXT hitPayload prd;
layout (location = 1) rayPayloadEXT bool shadowed;

layout (set = 0, binding = 10) uniform sampler2D[] skybox;

layout (set = 0, binding = 15) buffer SurfelDataBuffer {
	SurfelData surfelDataInBuffer[];
} surfelsData;

//#define PI 3.141592

void main()
{

    // vec4 result = prd.colorAndDist;

    surfelsData.surfelDataInBuffer[prd.surfel_index].hitpos = vec3(0);
    surfelsData.surfelDataInBuffer[prd.surfel_index].hitnormal = vec3(not(0));

    // vec3 dir            = normalize(gl_WorldRayDirectionEXT);
    // vec2 uv             = vec2(0.5 + atan(dir.x, dir.z) / (2 * PI), 0.5 - asin(dir.y) / PI);
    // vec3 color          = pow(texture(skybox[0], uv).xyz, vec3(2.2));
    // prd.colorAndDist += max(vec4(0.0), prd.pEnergy * vec4(color, 1.0));
    

    // //vec3 color = vec3(0.0, 1.0, 0.0);
    // //prd = hitPayload(vec4(1.0, 0.0, 0.0, 1.0), prd.pEnergy, prd.seed);

    // prd = hitPayload(prd.colorAndDist, prd.pEnergy, prd.seed);

    shadowed = false;

    vec3 dir            = normalize(gl_WorldRayDirectionEXT);
    vec2 uv             = vec2(0.5 + atan(dir.x, dir.z) / (2 * PI), 0.5 - asin(dir.y) / PI);
    vec3 color          = pow(texture(skybox[0], uv).xyz, vec3(2.2));

    color = prd.energy.xyz * color;

    prd.energy.xyz = vec3(0.0);

    
    //prd.colorAndDist.xyz    += color;
    //prd.colorAndDist    = vec4(vec3(0.0), -1);

    //prd.colorAndDist    = vec4(vec3(1.0, 0.0, 0.0), -1);
}