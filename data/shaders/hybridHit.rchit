#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable

#include "raycommon.glsl"
#include "helpers.glsl"

layout (location = 0) rayPayloadInEXT hitPayload prd;
layout (location = 1) rayPayloadEXT bool shadowed;
hitAttributeEXT vec3 attribs;

layout (set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;
layout (set = 0, binding = 6) buffer Lights { Light lights[]; } lightsBuffer;
layout (set = 0, binding = 7, scalar) buffer Vertices { Vertex v[]; } vertices[];
layout (set = 0, binding = 8) buffer Indices { int i[]; } indices[];
layout (set = 0, binding = 9) uniform sampler2D[] textures;
layout (set = 0, binding = 11) buffer MaterialBuffer { Material mat[]; } materials;
layout (set = 0, binding = 12) buffer sceneBuffer { vec4 idx[]; } objIndices;
layout (set = 0, binding = 13, scalar) buffer Matrices { mat4 m[]; } matrices;

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


  // Use above results to calculate normal vector
  // Calculate worldPos by using ray information
  const mat4 model      = matrices.m[transformationID];
  const vec3 normal     = v0.normal.xyz * barycentricCoords.x + v1.normal.xyz * barycentricCoords.y + v2.normal.xyz * barycentricCoords.z;
  const vec2 uv         = v0.uv.xy * barycentricCoords.x + v1.uv.xy * barycentricCoords.y + v2.uv.xy * barycentricCoords.z;
  const vec3 N          = normalize(model * vec4(normal, 0)).xyz;
  const vec3 worldPos   = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

  const Material mat    = materials.mat[materialID];
  const vec3 albedo     = texture(textures[int(mat.textures.x)], uv).xyz;
  const int shadingMode = int(mat.shadingMetallicRoughness.x);

  // Init values used for lightning
	vec3  color           = vec3(0);
	float attenuation     = 1.0;
  float light_intensity = 1.0;

  for(int i = 0; i < lightsBuffer.lights.length(); i++)
  {
    // Init basic light information
		Light light 				            = lightsBuffer.lights[i];
		const bool isDirectional        = light.pos.w < 0;
		vec3 L 						              = isDirectional ? light.pos.xyz : (light.pos.xyz - worldPos);
		const float light_max_distance 	= light.pos.w;
		const float light_distance 		  = length(L);
    L                               = normalize(L);
		const float NdotL 				      = clamp(dot(N, L), 0.0, 1.0);
		const float light_intensity 		= isDirectional ? 1.0f : (light.color.w / (light_distance * light_distance));
    float shadowFactor    = 0.0;

    // Check if light has impact
    if(NdotL > 0)
    {
			for(int a = 0; a < SHADOWSAMPLES; a++)
			{
        // Init as shadowed
				shadowed 	        = true;
				const vec3 dir    = normalize(sampleSphere(prd.seed, light.pos.xyz, light.radius) - worldPos);
        const uint flags  = gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT;
        // Shadow ray cast
				float tmin = 0.001, tmax  = light_distance + 1;
				traceRayEXT(topLevelAS, flags, 0xff, 1, 0, 1, 
          worldPos.xyz + dir * 1e-2, tmin, dir, tmax, 1);

				if(!shadowed){
					shadowFactor++;
        }
			}
			shadowFactor /= SHADOWSAMPLES;
    }

    // Calculate attenuation factor
    if(light_intensity == 0){
      attenuation = 0.1;
    }
    else{
      attenuation = light_max_distance - light_distance;
      attenuation /= light_max_distance;
      attenuation = max(attenuation, 0.0);
      attenuation = isDirectional ? 0.3 : attenuation * attenuation;
    }

    vec3 difColor = vec3(0);

    if(shadingMode == 0)  // DIFUS
    {
      difColor  = computeDiffuse(mat, N, L) * albedo;
      color    += difColor * light_intensity * light.color.xyz * attenuation * shadowFactor;
      prd       = hitPayload(vec4(color, gl_HitTEXT), vec4(1, 1, 1, 0), worldPos, prd.seed);
    }
    else if(shadingMode == 3) // MIRALL
    {
      const vec3 reflected    = reflect(normalize(gl_WorldRayDirectionEXT), N);
      const bool isScattered  = dot(reflected, N) > 0;

      difColor = isScattered ? computeDiffuse(mat, N, L) : vec3(1);
      color += difColor * light_intensity * light.color.xyz * attenuation * shadowFactor;

      prd = hitPayload(vec4(color, gl_HitTEXT), vec4(reflected, isScattered ? 1 : 0), worldPos, prd.seed);
    }
    else if(shadingMode == 4) // VIDRE
    {
      const float ior       = mat.diffuse.w;
      const float NdotV     = dot( N, normalize(gl_WorldRayDirectionEXT));
			const vec3 refrNormal = NdotV > 0.0 ? -N : N;
			const float refrEta   = NdotV > 0.0 ? 1 / ior : ior;

      color += mat.diffuse.xyz * light_intensity * light.color.xyz;
			
      float radicand = 1 + pow(refrEta, 2.0) * (NdotV * NdotV - 1);
      const vec4 direction = radicand < 0.0 ? 
                  vec4(reflect(gl_WorldRayDirectionEXT, N), 1) :
                  vec4(refract( normalize(gl_WorldRayDirectionEXT), refrNormal, refrEta ), 1);

      prd = hitPayload(vec4(color, gl_HitTEXT), direction, worldPos, prd.seed);
    }
  }
}