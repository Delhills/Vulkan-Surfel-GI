#extension GL_GOOGLE_include_directive : enable
#include "random.glsl"

// CONSTS ----------------------
const float PI = 3.14159265;
const int NSAMPLES = 1;
const int SHADOWSAMPLES = 1;
const int MAX_RECURSION = 10;

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
float Schlick(const float cosine, const float refractionIndex)
{
	float r0 = (1 - refractionIndex) / (1 + refractionIndex);
	r0 *= r0;
	return r0 + (1 - r0) * pow(1 - cosine, 5);
}

vec3 computeDiffuse(Material m, vec3 normal, vec3 lightDir)
{
    float NdotL = clamp(dot(normal, lightDir), 0.0f, 1.0f);
    return NdotL * m.diffuse.xyz;
};

vec3 computeSpecular(Material m, vec3 normal, vec3 lightDir, vec3 viewDir)
{
    // Specular
    vec3 V          = normalize(viewDir);
    vec3 R          = reflect(-normalize(lightDir), normalize(normal));
    float specular  = pow(clamp(dot(normalize(R), V), 0.0, 1.0), m.shadingMetallicRoughness.z);
    return vec3(1) * specular;
};

mat3 rotMat(vec3 axis, float angle)
{
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;

    return mat3(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,
                oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,
                oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c);
}

mat3 rotationMatrix(vec3 axis, float angle)
{
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;
    
    return mat3(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,
                oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,
                oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c);
}

vec3 sampleCone(inout uint seed, const vec3 direction, const float angle)
{
    float cosAngle = cos(angle);

    const float a = rnd(seed);

    const float cosTheta    = (1 - a) + a * cosAngle;
    const float sinTheta    = sqrt(1 - cosTheta * cosTheta);
    const float phi         = rnd(seed) * 2 * PI;

    vec3 sampleDirection = vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);

    const vec3 north        = vec3(0.0, 0.0, 1.0);
    const vec3 axis         = normalize(cross(north, normalize(direction)));
    const float rotAngle    = acos(dot(normalize(direction), north));

    mat3 rot = rotMat(axis, rotAngle);

    return rot * -sampleDirection;
}

vec3 getConeSample(inout uint rngState, vec3 direction, float coneAngle) {
    float cosAngle = cos(coneAngle);

    // Generate points on the spherical cap around the north pole [1].
    float z = rnd(rngState) * (1.0f - cosAngle) + cosAngle;
    float phi = rnd(rngState) * 2.0f * PI;

    float x = sqrt(1.0f - z * z) * cos(phi);
    float y = sqrt(1.0f - z * z) * sin(phi);
    vec3 north = vec3(0.f, 0.f, 1.f);

    // Find the rotation axis u and rotation angle rot [1]
    vec3 axis = normalize(cross(north, normalize(direction)));
    float angle = acos(dot(normalize(direction), north));

    // Convert rotation axis and angle to 3x3 rotation matrix [2]
    mat3 R = rotMat(axis, angle);

    return R * vec3(x, y, z);
}


vec3 sampleSphere(inout uint seed, const vec3 center, const float r)
{
    const float theta = 2 * PI * rnd(seed);
    const float phi = acos(1 - 2 * rnd(seed));

    const float x = sin(phi) * cos(theta);
    const float y = sin(phi) * sin(theta);
    const float z = cos(phi);

    return center + (r * vec3(x, y, z));
}
