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
  vec4 position;
  vec4 normal;
  vec4 color;
  vec4 uv;
  vec4 tangent;
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

mat3 rotMat(const vec3 axis, const float angle)
{
    float c = cos(angle);
    float s = sin(angle);

    float t = 1 - c;
    float x = axis.x;
    float y = axis.y;
    float z = axis.z;

    return mat3(
        t * x * x + c,      t * x * y - s * z,  t * x * z + s * y,
        t * x * y + s * z,  t * y * y + c,      t * y * z - s * x,
        t * x * z - s * y,  t * y * z + s * x,  t * z * z + c
    );
}

mat4 rotationMatrix(vec3 axis, float angle)
{
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;
    
    return mat4(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,
                oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,
                oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,
                0.0,                                0.0,                                0.0,                                1.0);
}

vec3 sampleCone(inout uint seed, const vec3 direction, const float angle)
{
    float cosAngle = cos(angle); //1

    // This range to [cosTheta, 1]. In case rnd is 1, z will be 1, whereas if rnd is 0, z will be cosTheta.
    float z = rnd(seed) * (1.0 - cosAngle) + cosAngle; // 1

    float phi = rnd(seed) * 2.0 * PI;
    float x = sqrt(1.0 - z * z) * cos(phi); // 0
    float y = sqrt(1.0 - z * z) * sin(phi); // 0
    vec3 north = vec3(0.0, 0.0, 1.0);

    vec3 axis = normalize(cross(north, normalize(direction)));
    float rotAngle = acos(dot(normalize(direction), north));

    mat4 rot = rotationMatrix(axis, rotAngle);

    return vec3(rot * vec4(x, y, z, 1));//vec3(x, y, z);
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
