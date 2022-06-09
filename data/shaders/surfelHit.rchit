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

  	const vec3 barycentricCoords = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);
  
	vec4 objIdx = objIndices.idx[gl_InstanceCustomIndexEXT];

	int instanceID        = int(objIdx.x);
	int materialID        = int(objIdx.y);
	int transformationID  = int(objIdx.z);
	int firstIndex        = int(objIdx.w);

	ivec3 ind     = ivec3(indices[instanceID].i[3 * gl_PrimitiveID + firstIndex + 0], 
	 					indices[instanceID].i[3 * gl_PrimitiveID + firstIndex + 1], 
	 					indices[instanceID].i[3 * gl_PrimitiveID + firstIndex + 2]);

	Vertex v0     = vertices[instanceID].v[ind.x];
	Vertex v1     = vertices[instanceID].v[ind.y];
	Vertex v2     = vertices[instanceID].v[ind.z];

	const mat4 model      = matrices.m[transformationID];

	// // Use above results to calculate normal vector
	// // Calculate worldPos by using ray information
	const vec3 normal     = v0.normal.xyz * barycentricCoords.x + v1.normal.xyz * barycentricCoords.y + v2.normal.xyz * barycentricCoords.z;
	const vec2 uv         = v0.uv.xy * barycentricCoords.x + v1.uv.xy * barycentricCoords.y + v2.uv.xy * barycentricCoords.z;
	const vec3 N          = normalize(mat3(transpose(inverse(model))) * normal).xyz;
	const vec3 V          = normalize(-gl_WorldRayDirectionEXT);
	const float NdotV     = clamp(dot(N, V), 0.0, 1.0);
	const vec3 worldPos   = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

	// // Init values used for lightning
	vec3 Lo               = vec3(0);
	//float attenuation     = 1.0;
	float light_intensity = 1.0;

	// // Init all material values
	const Material mat            = materials.mat[materialID];
	const int shadingMode         = int(mat.shadingMetallicRoughness.x);
	vec3 albedo                   = mat.textures.x > -1 ? texture(textures[int(mat.textures.x)], uv).xyz : mat.diffuse.xyz;
	const vec3 emissive           = mat.textures.z > -1 ? texture(textures[int(mat.textures.z)], uv).xyz : vec3(0);
	const vec3 roughnessMetallic  = mat.textures.w > -1 ? texture(textures[int(mat.textures.w)], uv).xyz : vec3(0, mat.shadingMetallicRoughness.z, mat.shadingMetallicRoughness.y);

	albedo                        = pow(albedo, vec3(2.2));
	const float roughness         = roughnessMetallic.y;
	const float metallic          = roughnessMetallic.z;
	vec3 F0                       = mix(vec3(0.04), albedo, metallic);

	// // Environment 

  	//vec4 result = prd.colorAndDist;

	// //vec4 direction = vec4(1, 1, 1, 0);
	vec4 origin = vec4(worldPos, 0);

	float seed = 0.123456;

	//prd.colorAndDist.xyz += max(vec3(0.0), prd.energy.xyz * emissive);

	float f90 = clamp(0.0, 1.0, 50.0 * dot(F0, vec3(0.33)));

	//float seed = rnd(prd.seed);

	vec3 F = F0 + (f90 - F0) * pow(1.0 - NdotV, 5);// Schlick 1994, "An Inexpensive BRDF Model for Physically-Based Rendering"

	
	const float specChance = dot(F, vec3(0.333));

	//vec3 direction = SampleHemisphere_cos(N, seed, uv);
	prd.energy.xyz *= albedo / (1 - specChance);

	vec3 result;

	// modu = sqrt(modu);

	// float ss = 1.0-(modu/5.0);

	// prd.energy.xyz *= ss;

	for(int i = 0; i < lightsBuffer.lights.length(); i++)
	{
	//Init basic light information

		vec3 lightResult = vec3(0.0);

		Light light 				    = lightsBuffer.lights[i];
		const bool isDirectional        = light.pos.w < 0;
		vec3 L 						    = isDirectional ? light.pos.xyz : (light.pos.xyz - worldPos);
		const float light_max_distance 	= light.pos.w;
		const float light_distance 		= length(L);
		L                               = normalize(L);
		light_intensity 	= isDirectional ? 1.0f : (light.color.w / (light_distance * light_distance));
		//light_intensity 	= 30.0/ (light_distance * light_distance);
		const vec3 H                    = normalize(V + L);
		const float NdotL 				= clamp(dot(N, L), 0.0, 1.0);
		const float NdotH               = clamp(dot(N, H), 0.0, 1.0);
		float shadowFactor              = 1.0;

		float dist = 0;
		if(isDirectional){

			dist = 10000000;
			lightResult = light.color.xyz * light_intensity;
		}
		else{
			float dist2 = dot(L, L);
			float range2 = light.radius * light.radius;

			if(dist2<range2){
				dist = sqrt(dist2);
				L /= dist;

				if(NdotL>0){
					lightResult = light.color.xyz * light_intensity;
					float att = clamp(0.0, 1.0, (1.0 - (dist2/range2)));
					float attenuation = att * att;

					lightResult *= attenuation;
				}
			}
		}

		if(NdotL > 0 && dist > 0){
			vec3 shadow = NdotL * prd.energy.xyz;


			result += max(vec3(0), shadow * lightResult / PI);
		}
	}

	vec3 distanceHit = surfels.surfelInBuffer[prd.surfel_index].position - worldPos;

	float modu = dot(distanceHit, distanceHit);

	float range2 = 5.0 * 5.0;

	prd.colorAndDist.xyz = result;

	// if(modu<range2){
	// 	float hitdist = sqrt(modu);
	// 	distanceHit /= hitdist;

	// 	float hitdotN = dot(distanceHit, N);

	// 	if(hitdotN>0){
	// 		prd.colorAndDist.xyz = result;

	// 		float att = clamp(0.0, 1.0, (1.0 - (modu/range2)));
	// 		float attenuation = att * att;

	// 		prd.colorAndDist.xyz *= attenuation;
	// 	}
	// }
	

  //prd = hitPayload(vec4(1.0, 0.0, 0.0, 0.0), prd.pEnergy, prd.seed);

//   surfelsData.surfelDataInBuffer[gl_LaunchIDEXT.x].hitnormal = N;

//   surfelsData.surfelDataInBuffer[gl_LaunchIDEXT.x].hitpos = worldPos;
  
//   surfelsData.surfelDataInBuffer[gl_LaunchIDEXT.x].hitenergy = prd.pEnergy.xyz;
//   surfelsData.surfelDataInBuffer[gl_LaunchIDEXT.x].traceresult = result.xyz;


  //prd = hitPayload(result, prd.pEnergy, prd.seed);

  //prd = hitPayload(vec4(0.0, 0.0, 1.0, 1.0), prd.pEnergy, prd.seed);

	//color = vec3(1.0, 0.0, 1.0);

	// prd = hitPayload(vec4(color, gl_HitTEXT), prd.energy, prd.surfel_index);


	surfelsData.surfelDataInBuffer[prd.surfel_index].hitpos = worldPos;
    surfelsData.surfelDataInBuffer[prd.surfel_index].hitnormal = N;

	//prd = hitPayload(vec4(result.xyz, gl_HitTEXT), prd.pEnergy, prd.seed);

}