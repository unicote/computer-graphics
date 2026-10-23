#version 450

layout(set = 0, binding = 0) uniform GlobalUniforms {
    mat4 model;
    mat4 view;
    mat4 projection;
    vec4 colour_multiplier;
} uniforms;

layout(location = 0) in vec3 in_position;

layout(location = 0) out vec3 vertex_color;

void main() {
    gl_Position = uniforms.projection * uniforms.view * uniforms.model * vec4(in_position, 1.0);
    vertex_color = in_position * 0.5 + 0.5;
}
