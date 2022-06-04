#version 460

#extension GL_GOOGLE_include_directive : enable

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 3) in vec2 inUV;

layout(location = 0) out vec3 outPosition;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec3 outColor;
layout(location = 3) out vec2 outUV;
layout(location = 4) out vec4 ndcPrev;
layout(location = 5) out vec4 ndc;

struct ObjectData{
	mat4 model;
};

// Set 0 - 
information
layout(set = 0, binding = 0) uniform CameraBuffer
{
	mat4 view;
	mat4 projection;
	mat4 pView;
	mat4 pProj;
} cameraData;

layout(push_constant) uniform constants
{
	mat4 matrix;
	mat4 inv_matrix;
}pushC;

void main()
{
	mat4 transformationMatrix 	= cameraData.projection * cameraData.view * pushC.matrix;
	mat4 previousTransformation = cameraData.pProj * cameraData.pView * pushC.matrix;
	gl_Position 				= transformationMatrix * vec4(inPosition, 1.0);

	outPosition = vec3(pushC.matrix * vec4(inPosition, 1.0)).xyz;
    outColor  	= inColor;
	outNormal 	= mat3(transpose(pushC.inv_matrix)) * vec3(inNormal);
    outUV 		= inUV;
	ndc 		= transformationMatrix * vec4(inPosition, 1.0);	// in homogeneous space
	ndcPrev 	= previousTransformation * vec4(inPosition, 1.0);
}