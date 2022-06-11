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

	// Use above results to calculate normal vector
	// Calculate worldPos by using ray information
	const vec3 normal     = v0.normal.xyz * barycentricCoords.x + v1.normal.xyz * barycentricCoords.y + v2.normal.xyz * barycentricCoords.z;
	const vec2 uv         = v0.uv.xy * barycentricCoords.x + v1.uv.xy * barycentricCoords.y + v2.uv.xy * barycentricCoords.z;
	const vec3 N          = normalize(mat3(transpose(inverse(model))) * normal).xyz;
	const vec3 V          = normalize(-gl_WorldRayDirectionEXT);
	const float NdotV     = clamp(dot(N, V), 0.0, 1.0);
	const vec3 worldPos   = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

  // Init values used for lightning
	vec3 Lo               = vec3(0);
	float attenuation     = 1.0;
	float light_intensity = 1.0;

	// Init all material values
	const Material mat            = materials.mat[materialID];
	const int shadingMode         = int(mat.shadingMetallicRoughness.x);
	vec3 albedo                   = mat.textures.x > -1 ? texture(textures[int(mat.textures.x)], uv).xyz : mat.diffuse.xyz;
	const vec3 emissive           = mat.textures.z > -1 ? texture(textures[int(mat.textures.z)], uv).xyz : vec3(0);
	const vec3 roughnessMetallic  = mat.textures.w > -1 ? texture(textures[int(mat.textures.w)], uv).xyz : vec3(0, mat.shadingMetallicRoughness.z, mat.shadingMetallicRoughness.y);

	albedo                        = pow(albedo, vec3(2.2));
	const float roughness         = roughnessMetallic.y;
	const float metallic          = roughnessMetallic.z;
	vec3 F0                       = mix(vec3(0.04), albedo, metallic);

	// Environment 
	vec2 environmentUV = vec2(0.5 + atan(N.x, N.z) / (2 * PI), 0.5 - asin(N.y) / PI);
	vec3 irradiance = texture(environmentTexture[1], environmentUV).xyz;

	vec4 direction = vec4(1, 1, 1, 0);
	vec4 origin = vec4(worldPos, 0);
	vec3 difColor = vec3(0);
	for(int i = 0; i < lightsBuffer.lights.length(); i++)
	{
		// Init basic light information
		Light light						= lightsBuffer.lights[i];
		const bool isDirectional        = light.pos.w < 0;
		vec3 L							= isDirectional ? light.pos.xyz : (light.pos.xyz - worldPos);


		const float light_max_distance 	= light.pos.w;
		const float light_distance		= length(L);
		L 								= normalize(L);
		const vec3 H                    = normalize(V + L);
		const float light_intensity		= isDirectional ? 1.0f : light.color.w;

		//const float light_intensity		= (100.0 / (light_distance * light_distance));
		const float NdotL				= clamp(dot(N, L), 0.0, 1.0);
		const float NdotH               = clamp(dot(N, H), 0.0, 1.0);
		float shadowFactor              = 1.0;

		// Check if light has impact
		// Calculate attenuation factor
		if(light_intensity == 0){
			attenuation = 0.0;
		}
		else{
			attenuation = light_max_distance - light_distance;
			attenuation /= light_max_distance;
			attenuation = max(attenuation, 0.0);
		}

		if(shadingMode == 0)  // DIFUS
		{
			if(NdotL > 0)
			{
				for(int a = 0; a < 1; a++)
				{
					// Init as shadowed
					shadowed 	        = true;
					// if(light_distance < light_max_distance)
					// {
					vec3 dir          = L;
					const uint flags  = gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT;
					float tmin = 0.001, tmax  = light_distance - 0.001;

					// Shadow ray cast
					traceRayEXT(topLevelAS, flags, 0xff, 0, 0, 1, 
					worldPos.xyz + dir * 0.01, tmin, dir, tmax, 1);
					// }

					if(shadowed){
						shadowFactor = 0.0;
					}
					else{
						shadowFactor = 1.0;
					}
				}
			}

			vec3 radiance = light_intensity * (light.color.xyz) * attenuation * shadowFactor;
			vec3 F        = FresnelSchlick(NdotH, F0);
			float D       = DistributionGGX(N, H, roughness);
			float G       = GeometrySmith(N, V, L, roughness);

			vec3 numerator    = D * G * F;
			float denominator = max(4.0 * clamp(dot(N, V), 0.0, 1.0) * NdotL, 0.000001);
			vec3 specular     = numerator / denominator;

			vec3 kS = F;
			vec3 kD = (vec3(1.0) - kS) * (1.0 - metallic);

			prd.energy.xyz *= kD * albedo;

			Lo += (prd.energy.xyz / PI + specular) * radiance * NdotL;
			//direction = vec4(1, 1, 1, 0);
		}

		vec3 radiance = (light.color.xyz) * light_intensity * attenuation;

		float NDF 	= DistributionGGX(N, H, roughness);
		float G 	= GeometrySmith(N, V, L, roughness);
		vec3 F 		= FresnelSchlick(max(dot(H, V), 0.0), F0);
		vec3 kD = vec3(1.0) - F;
		kD *= 1.0 - metallic;

		vec3 numerator 		= NDF * G * F;
		float denominator 	= 4.0 * NdotV * max(dot(N, L), 0.000001);
		vec3 specular 		= numerator / denominator;

		vec3 kS = F;

		prd.energy.xyz = (kD * albedo);

		Lo += NdotL * prd.energy.xyz * radiance / PI;

	}

	surfelsData.surfelDataInBuffer[prd.surfel_index].hitpos = worldPos;
	surfelsData.surfelDataInBuffer[prd.surfel_index].hitnormal = N;

	vec3 color = Lo + degamma(emissive);
	prd.colorAndDist.xyz    += color;
}