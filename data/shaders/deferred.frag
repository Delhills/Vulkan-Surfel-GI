#version 460

struct Light{
	vec4 pos;	// w used for max distance
	vec4 color;	// w used for intensity
	float radius;
};

layout (location = 0) out vec4 outFragColor;
layout (location = 0) in vec2 inUV;
layout (location = 1) in vec3 inCamPosition;

layout (set = 0, binding = 0) uniform sampler2D positionTexture;
layout (set = 0, binding = 1) uniform sampler2D normalTexture;
layout (set = 0, binding = 2) uniform sampler2D albedoTexture;
layout (set = 0, binding = 3) uniform sampler2D motionTexture;
layout (std140, set = 0, binding = 4) buffer LightBuffer {Light lights[];} lightBuffer;
layout (set = 0, binding = 5) uniform debugInfo {int target;} debug;
layout (set = 0, binding = 6) uniform sampler2D materialTexture;
layout (set = 0, binding = 8) uniform sampler2D emissiveTexture;
layout (set = 0, binding = 9) uniform sampler2D environmentTexture;
layout (set = 0, binding = 10) uniform sampler2D debugGI;
layout (set = 0, binding = 11) uniform sampler2D resultGI;

const float PI = 3.14159265359;

float DistributionGGX(vec3 N, vec3 H, float a);
float GeometrySchlickGGX(float NdotV, float k);
float GeometrySmith(vec3 N, vec3 V, vec3 L, float k);
vec3 FresnelSchlick(float cosTheta, vec3 F0);

void main() 
{
	vec3 position 	= texture(positionTexture, inUV).xyz;
	vec3 normal 	= texture(normalTexture, inUV).xyz * 2.0 - vec3(1);
	vec3 albedo 	= texture(albedoTexture, inUV).xyz;
	vec3 motion		= texture(motionTexture, inUV).xyz * 2.0 - vec3(1);
	vec3 material 	= texture(materialTexture, inUV).xyz;
	vec3 emissive 	= texture(emissiveTexture, inUV).xyz;

	vec4 debugGI 	= texture(debugGI, inUV);

	vec4 resultGI 	= texture(resultGI, inUV);

	bool background = texture(positionTexture, inUV).w == 0 && texture(normalTexture, inUV).w == 0;
	float metallic 	= material.z;
	float roughness = material.y;

	vec3 N 			= normalize(normal);
	vec3 V 			= normalize(inCamPosition - position.xyz);
	float NdotV 	= max(dot(N, V), 0.0);
	vec3 F0 		= mix(vec3(0.04), pow(albedo, vec3(2.2)), metallic);
	vec2 envUV 		= vec2(0.5 + atan(N.x, N.z) / (2 * PI), 0.5 - asin(N.y) / PI);
	vec3 irradiance = texture(environmentTexture, envUV).xyz;

	if(debug.target > 0.001)
	{
		switch(debug.target){
			case 1:
				outFragColor = vec4(position, 1);
				break;
			case 2:
				outFragColor = vec4(normal, 0);
				break;
			case 3:
				outFragColor = vec4(albedo, 1);
				break;
			case 4:
				outFragColor = vec4(motion, 1);
				break;
			case 5:
				outFragColor = vec4(material, 1);
				break;
			case 6:
				outFragColor = vec4(emissive, 1);
				break;
		}
		return;
	}

	vec3 color = vec3(1), Lo = vec3(0);
	float attenuation = 1.0, light_intensity = 1.0;
	
	for(int i = 0; i < lightBuffer.lights.length(); i++)
	{
		Light light 		= lightBuffer.lights[i];
		bool isDirectional 	= light.pos.w < 0;
		vec3 L 				= isDirectional ? light.pos.xyz : (light.pos.xyz - position.xyz);
		vec3 H 				= normalize(V + normalize(L));
		float NdotL 		= max(dot(N, normalize(L)), 0.0);

		// Calculate the directional light
		if(isDirectional)
		{
			Lo += (NdotL * light.color.xyz);
		}
		else	// Calculate point lights
		{
			float light_max_distance 	= light.pos.w;
			float light_distance 		= length(L);
			light_intensity 			= light.color.w / (light_distance * light_distance);

			attenuation = light_max_distance - light_distance;
			attenuation /= light_max_distance;
			attenuation = max(attenuation, 0.0);
			attenuation = attenuation * attenuation;

			vec3 radiance = light.color.xyz * light_intensity * attenuation;

			float NDF 	= DistributionGGX(N, H, roughness);
			float G 	= GeometrySmith(N, V, L, roughness);
			vec3 F 		= FresnelSchlick(max(dot(H, V), 0.0), F0);
			vec3 kD = vec3(1.0) - F;
			kD *= 1.0 - metallic;

			vec3 numerator 		= NDF * G * F;
			float denominator 	= 4.0 * NdotV * max(dot(N, L), 0.0);
			vec3 specular 		= numerator / max(denominator, 0.001);

			vec3 kS = F;

			Lo += (kD * pow(albedo, vec3(2.2)) / PI + specular) * radiance * NdotL;
		}
	}

	vec3 indirect = clamp(vec3(0.0), vec3(0.999), resultGI.xyz);

	indirect = indirect / (vec3(1.0) + indirect);
	
	if(!background){
		// Ambient from IBL
		vec3 F = FresnelSchlick(NdotV, F0);
  		vec3 kD = (1.0 - F) * (1.0 - metallic);
  		vec3 diffuse = kD * albedo * irradiance;
  		vec3 ambient = diffuse;


		color = Lo + indirect;// * 0.3;
		color += emissive;
	}
	else{
		color = albedo;
	}
	if(debugGI.x != 0 || debugGI.y != 0 || debugGI.z != 0|| debugGI.w != 0){
		//outFragColor = vec4(debugGI, 1.0 );
		//return;

		//color = debugGI.xyz;
	}
	else{
		//color = resultGI.xyz;
	}

	outFragColor = vec4( color.xyz, 1.0f );

}

float DistributionGGX(vec3 N, vec3 H, float roughness)
{
  float a = roughness * roughness;
	float a2 = a * a;
	float NdotH = max(dot(N, H), 0.0);
	float NdotH2 = NdotH * NdotH;

	float denom = (NdotH2 * (a2 - 1.0) + 1.0);
	denom = PI * denom * denom;

	return a2 / denom;
}

// Geometry Function
float GeometrySchlickGGX(float NdotV, float roughness)
{
  float r = roughness + 1.0;
  float k = (r * r) / 8.0;

	float nom = NdotV;
	float denom = NdotV * (1.0 - k) + k;

	return nom/denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
	float NdotV = max(dot(N, V), 0.0);
	float NdotL = max(dot(N, L), 0.0);
	float ggx1 = GeometrySchlickGGX(NdotV, roughness);
	float ggx2 = GeometrySchlickGGX(NdotL, roughness);

	return ggx1 * ggx2;
}

// Fresnel Equation
vec3 FresnelSchlick(float cosTheta, vec3 F0)
{
	return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}