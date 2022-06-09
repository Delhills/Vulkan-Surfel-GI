// "GPU Random Numbers via the Tiny Encryption Algorithm"

uint tea( uint val0, uint val1 )
{
	uint v0 = val0;
	uint v1 = val1;
	uint s0 = 0;

	for( uint n = 0; n < 16; n++ )
	{
		s0 += 0x9e3779b9;
		v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
		v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
	}

	return v0;
};

// Generate a random unsigned int in [0, 2^24) given the previous RNG state
// using the Numerical Recipes linear congruential generator
uint lcg(inout uint prev)
{
  uint LCG_A = 1664525u;
  uint LCG_C = 1013904223u;
  prev       = (LCG_A * prev + LCG_C);
  return prev & 0x00FFFFFF;
};

// Generate a random float in [0, 1) given the previous RNG state
float rnd(inout uint prev)
{
  return (float(lcg(prev)) / float(0x01000000));
};


vec3 samplingHemisphere(inout uint seed, in vec3 x, in vec3 y, in vec3 z)
{
	const float PI = 3.141592;
	float r1 = rnd(seed);
	float r2 = rnd(seed);
	float sq = sqrt(1 - r2);

	vec3 direction = vec3(cos(2 * PI * r1) * sq, sin(2 * PI * r1) * sq, sqrt(r2));
	direction = direction.x * x + direction.y * y + direction.z * z;

	return direction;
};

void createCoordSystem(in vec3 N, out vec3 tangent, out vec3 binormal)
{
	if(abs(N.x) > abs(N.y))
		tangent = vec3(N.z, 0, -N.x) / sqrt(N.x * N.x + N.z * N.z);
	else
		tangent = vec3(0, -N.z, N.y) / sqrt(N.y * N.y + N.z * N.z);
	binormal = cross(N, tangent); 
};

vec2 hash2(inout float seed) {
    return fract(sin(vec2(seed+=0.1,seed+=0.1))*vec2(43758.5453123,22578.1459123));
}

vec3 cosineSampleHemisphere(vec3 n, inout float seed)
{
    vec2 u = hash2(seed);

    float r = sqrt(u.x);
    float theta = 2.0 * 3.141592 * u.y;
 
    vec3  B = normalize( cross( n, vec3(0.0,1.0,1.0) ) );
	vec3  T = cross( B, n );
    
    return normalize(r * sin(theta) * B + sqrt(1.0 - u.x) * n + r * cos(theta) * T);
}


vec3 uniformSampleHemisphere(vec3 N, inout float seed) {
    vec2 u = hash2(seed);
    
    float r = sqrt(1.0 - u.x * u.x);
    float phi = 2.0 * 3.141592 * u.y;
    
    vec3  B = normalize( cross( N, vec3(0.0,1.0,1.0) ) );
	vec3  T = cross( B, N );
    
    return normalize(r * sin(phi) * B + u.x * N + r * cos(phi) * T);
}