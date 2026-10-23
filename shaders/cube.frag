#version 450

layout(set = 0, binding = 0) uniform GlobalUniforms {
    mat4 model;
    mat4 view;
    mat4 projection;
    vec4 colour_multiplier;
} uniforms;


layout(location = 0) in vec3 vertex_color;
layout(location = 0) out vec4 out_color;

void main() {
    out_color = vec4(vertex_color * uniforms.colour_multiplier.rgb, uniforms.colour_multiplier.a);
}
