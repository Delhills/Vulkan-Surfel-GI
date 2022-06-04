#extension GL_GOOGLE_include_directive : enable
#include "random.glsl"

// CONSTS ----------------------
const float PI = 3.14159265359;
const int NSAMPLES = 1;
const int SHADOWSAMPLES = 64;
const int MAX_RECURSION = 8;


// STRUCTS --------------------
struct Vertex
{
  vec4 normal;
  vec4 color;
  vec4 uv;
};

struct Light{
  vec4  pos;
  vec4  color;
  float radius;
};

struct Material{
	vec4 diffuse;
    vec4 textures;
    vec4 shadingMetallicRoughness;
};

// FUNCTIONS --------------------------------------------------
// Polynomial approximation by Christophe Schlick
// Trowbridge-Reitz GGX - Normal Distribution Function
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

vec3 computeAmbient(vec3 N, vec3 V, vec3 albedo, vec3 irradiance, vec3 F0)
{
    vec3 kS = FresnelSchlick(clamp(dot(N, V), 0.0, 1.0), F0);
    vec3 kD = 1.0 - kS;
    vec3 diffuse = albedo * irradiance;
    return (kD * diffuse);
}

vec3 sampleDisk(Light light, vec3 position, vec3 L, uint seed)
{
    float radius = light.radius * sqrt(rnd(seed));
    float angle = rnd(seed) * 2.0f * PI;
    vec2 point = vec2(radius * cos(angle), radius * sin(angle));
    vec3 tangent = normalize(cross(L, vec3(0, 1, 0)));
    if(L == vec3(0, 1, 0))
      tangent = vec3(0, 0, 1);
    vec3 bitangent = normalize(cross(tangent, L));
    vec3 target = position + L + point.x * tangent + point.y * bitangent;
    return normalize(target - position);
}