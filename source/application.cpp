#include "application.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <imgui.h>
#include <numbers>

namespace {

struct Vertex {
	glm::vec3 position;
};

struct GlobalUniforms {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 projection;
	glm::vec4 color_multiplier;
};

const Vertex cube_vertices[] = {
	{.position = { -1, -1, -1 }},
	{.position = { +1, -1, -1 }},
	{.position = { +1, +1, -1 }},
	{.position = { -1, +1, -1 }},
	{.position = { -1, -1, +1 }},
	{.position = { +1, -1, +1 }},
	{.position = { +1, +1, +1 }},
	{.position = { -1, +1, +1 }},
};

const uint32_t cube_indices[] = {
	0, 2, 1,  0, 3, 2,
	4, 5, 6,  4, 6, 7,
	0, 4, 7,  0, 7, 3,
	1, 2, 6,  1, 6, 5,
	0, 1, 5,  0, 5, 4,
	3, 7, 6,  3, 6, 2,
};

struct Scene {
	VkBuffer vertex_buffer = VK_NULL_HANDLE;
	VmaAllocation vertex_buffer_allocation = VK_NULL_HANDLE;
	void* vertex_buffer_mapped = nullptr;

	VkBuffer index_buffer = VK_NULL_HANDLE;
	VmaAllocation index_buffer_allocation = VK_NULL_HANDLE;
	void* index_buffer_mapped = nullptr;

	VkBuffer first_uniform_buffer = VK_NULL_HANDLE;
	VmaAllocation first_uniform_buffer_allocation = VK_NULL_HANDLE;
	void* first_uniform_buffer_mapped = nullptr;
	VkBuffer second_uniform_buffer = VK_NULL_HANDLE;
	VmaAllocation second_uniform_buffer_allocation = VK_NULL_HANDLE;
	void* second_uniform_buffer_mapped = nullptr;

	VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
	VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
	VkDescriptorSet first_descriptor_set = VK_NULL_HANDLE;
	VkDescriptorSet second_descriptor_set = VK_NULL_HANDLE;
	VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;

	VkPipeline pipeline = VK_NULL_HANDLE;
};

struct Settings {
	bool perspective = false;
	bool animation_playing = true;
	float animation_speed = 1;

	float radius = 1;
	float height = 1;
	float depth = 1;
	float phase = 1;

	glm::vec3 first_position{};
	glm::vec3 first_rotation{0.0};
	glm::vec3 first_scale{1};
	glm::vec4 first_color{1, 1, 1, 1};

	glm::vec3 second_position{3.5, -2, 0};
	glm::vec3 second_rotation{0};
	glm::vec3 second_scale{0.5};
	glm::vec4 second_color{0.5};
};

Scene scene;
Settings controls;
double previous_time = 0.0;

std::vector<uint32_t> read_spirv(const char* path) {
	std::ifstream file(path, std::ios::ate | std::ios::binary);
	if (!file) {
		std::cerr << "Cannot open shader file: " << path << '\n';
		return {};
	}

	const std::streamsize bytes = file.tellg();
	std::vector<uint32_t> code(static_cast<std::size_t>(bytes) / sizeof(uint32_t));
	file.seekg(0);
	file.read(reinterpret_cast<char*>(code.data()), bytes);
	return code;
}

VkShaderModule load_shader(const char* path) {
	const std::vector<uint32_t> code = read_spirv(path);
	if (code.empty()) return VK_NULL_HANDLE;

	const VkShaderModuleCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = code.size() * sizeof(uint32_t),
		.pCode = code.data(),
	};
	VkShaderModule shader = VK_NULL_HANDLE;
	if (vkCreateShaderModule(graphics::internal::context.device, &info, nullptr, &shader) != VK_SUCCESS) {
		std::cerr << "Failed to create shader module: " << path << '\n';
		return VK_NULL_HANDLE;
	}
	return shader;
}

bool create_buffer(VkBuffer& buffer, VmaAllocation& allocation, void*& mapped,
	const VkDeviceSize size, const VkBufferUsageFlags usage) {
	const VkBufferCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	constexpr VmaAllocationCreateInfo allocation_info = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};
	VmaAllocationInfo allocation_result{};
	if (vmaCreateBuffer(graphics::internal::context.allocator, &info, &allocation_info, &buffer, &allocation, &allocation_result) != VK_SUCCESS) {
		std::cerr << "Failed to create buffer\n";
		return false;
	}
	mapped = allocation_result.pMappedData;
	return true;
}

void destroy_resources() {
	const auto& context = graphics::internal::context;
	if (scene.pipeline != VK_NULL_HANDLE) vkDestroyPipeline(context.device, scene.pipeline, nullptr);
	if (scene.pipeline_layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(context.device, scene.pipeline_layout, nullptr);
	if (scene.descriptor_pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(context.device, scene.descriptor_pool, nullptr);
	if (scene.descriptor_layout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(context.device, scene.descriptor_layout, nullptr);
	if (scene.second_uniform_buffer != VK_NULL_HANDLE) vmaDestroyBuffer(context.allocator, scene.second_uniform_buffer, scene.second_uniform_buffer_allocation);
	if (scene.first_uniform_buffer != VK_NULL_HANDLE) vmaDestroyBuffer(context.allocator, scene.first_uniform_buffer, scene.first_uniform_buffer_allocation);
	if (scene.index_buffer != VK_NULL_HANDLE) vmaDestroyBuffer(context.allocator, scene.index_buffer, scene.index_buffer_allocation);
	if (scene.vertex_buffer != VK_NULL_HANDLE) vmaDestroyBuffer(context.allocator, scene.vertex_buffer, scene.vertex_buffer_allocation);
	scene = {};
}

bool create_descriptors() {
	const auto& context = graphics::internal::context;
	constexpr VkDescriptorSetLayoutBinding binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
	};
	const VkDescriptorSetLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &binding,
	};
	if (vkCreateDescriptorSetLayout(context.device, &layout_info, nullptr, &scene.descriptor_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor set layout\n";
		return false;
	}

	constexpr VkDescriptorPoolSize pool_size = {
		.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 2,
	};
	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 2,
		.poolSizeCount = 1,
		.pPoolSizes = &pool_size,
	};
	if (vkCreateDescriptorPool(context.device, &pool_info, nullptr, &scene.descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor pool\n";
		return false;
	}

	const VkDescriptorSetLayout layouts[] = {
		scene.descriptor_layout,
		scene.descriptor_layout,
	};
	VkDescriptorSet descriptor_sets[2]{};
	const VkDescriptorSetAllocateInfo allocation_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = scene.descriptor_pool,
		.descriptorSetCount = std::size(layouts),
		.pSetLayouts = layouts,
	};
	if (vkAllocateDescriptorSets(context.device, &allocation_info, descriptor_sets) != VK_SUCCESS) {
		std::cerr << "Failed to allocate descriptor sets\n";
		return false;
	}

	scene.first_descriptor_set = descriptor_sets[0];
	scene.second_descriptor_set = descriptor_sets[1];

	const VkDescriptorBufferInfo first_buffer_info = {
		.buffer = scene.first_uniform_buffer,
		.offset = 0,
		.range = VK_WHOLE_SIZE,
	};
	const VkDescriptorBufferInfo second_buffer_info = {
		.buffer = scene.second_uniform_buffer,
		.offset = 0,
		.range = VK_WHOLE_SIZE,
	};
	const VkWriteDescriptorSet writes[] = {
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = scene.first_descriptor_set,
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &first_buffer_info,
		},
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = scene.second_descriptor_set,
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &second_buffer_info,
		},
	};
	vkUpdateDescriptorSets(context.device, std::size(writes), writes, 0, nullptr);
	return true;
}

bool create_pipeline() {
	const auto& context = graphics::internal::context;
	VkShaderModule vertex_shader = load_shader("shaders/cube.vert.spv");
	VkShaderModule fragment_shader = load_shader("shaders/cube.frag.spv");
	if (vertex_shader == VK_NULL_HANDLE || fragment_shader == VK_NULL_HANDLE) {
		if (vertex_shader != VK_NULL_HANDLE) vkDestroyShaderModule(context.device, vertex_shader, nullptr);
		if (fragment_shader != VK_NULL_HANDLE) vkDestroyShaderModule(context.device, fragment_shader, nullptr);
		return false;
	}

	const VkPipelineShaderStageCreateInfo stages[] = {
		{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vertex_shader, .pName = "main" },
		{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fragment_shader, .pName = "main" },
	};
	constexpr VkVertexInputBindingDescription vertex_binding = {
		.binding = 0,
		.stride = sizeof(Vertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};
	constexpr std::array vertex_attributes = {
		VkVertexInputAttributeDescription{ .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, position) },
	};
	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &vertex_binding,
		.vertexAttributeDescriptionCount = vertex_attributes.size(),
		.pVertexAttributeDescriptions = vertex_attributes.data(),
	};
	constexpr VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
	};
	constexpr VkPipelineViewportStateCreateInfo viewport = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};

	constexpr VkPipelineRasterizationStateCreateInfo rasterizer = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_BACK_BIT,
		.frontFace = VK_FRONT_FACE_CLOCKWISE,
		.lineWidth = 1.0,
	};

	constexpr VkPipelineMultisampleStateCreateInfo multisampling = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};

	constexpr VkPipelineDepthStencilStateCreateInfo depth_stencil = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS,
	};
	constexpr VkPipelineColorBlendAttachmentState color_attachment = {
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	};
	const VkPipelineColorBlendStateCreateInfo color_blending = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &color_attachment,
	};

	constexpr VkDynamicState dynamic_states[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};

	const VkPipelineDynamicStateCreateInfo dynamic_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = std::size(dynamic_states),
		.pDynamicStates = dynamic_states,
	};

	constexpr VkPipelineLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &scene.descriptor_layout,
	};

	if (vkCreatePipelineLayout(context.device, &layout_info, nullptr, &scene.pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create pipeline layout\n";
		vkDestroyShaderModule(context.device, fragment_shader, nullptr);
		vkDestroyShaderModule(context.device, vertex_shader, nullptr);
		return false;
	}

	const VkGraphicsPipelineCreateInfo pipeline_info = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = std::size(stages),
		.pStages = stages,
		.pVertexInputState = &vertex_input,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport,
		.pRasterizationState = &rasterizer,
		.pMultisampleState = &multisampling,
		.pDepthStencilState = &depth_stencil,
		.pColorBlendState = &color_blending,
		.pDynamicState = &dynamic_state,
		.layout = scene.pipeline_layout,
		.renderPass = context.render_pass,
	};
	const VkResult result = vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &scene.pipeline);
	vkDestroyShaderModule(context.device, fragment_shader, nullptr);
	vkDestroyShaderModule(context.device, vertex_shader, nullptr);
	return result == VK_SUCCESS;
}

bool create_resources() {
	if (!create_buffer(scene.vertex_buffer, scene.vertex_buffer_allocation, scene.vertex_buffer_mapped,sizeof(cube_vertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) || !create_buffer(scene.index_buffer, scene.index_buffer_allocation, scene.index_buffer_mapped,sizeof(cube_indices), VK_BUFFER_USAGE_INDEX_BUFFER_BIT)) return false;
	if (!create_buffer(scene.first_uniform_buffer, scene.first_uniform_buffer_allocation,scene.first_uniform_buffer_mapped, sizeof(GlobalUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) || !create_buffer(scene.second_uniform_buffer, scene.second_uniform_buffer_allocation,scene.second_uniform_buffer_mapped, sizeof(GlobalUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)) return false;

	std::memcpy(scene.vertex_buffer_mapped, cube_vertices, sizeof(cube_vertices));
	vmaFlushAllocation(graphics::internal::context.allocator, scene.vertex_buffer_allocation,0, sizeof(cube_vertices));
	std::memcpy(scene.index_buffer_mapped, cube_indices, sizeof(cube_indices));
	vmaFlushAllocation(graphics::internal::context.allocator, scene.index_buffer_allocation,0, sizeof(cube_indices));
	return create_descriptors() && create_pipeline();
}

glm::mat4 model_matrix(const glm::vec3& position, const glm::vec3& rotation, const glm::vec3& scale, const glm::vec3 offset = {}, const glm::vec3 animation_rotation = {}) {
	constexpr glm::mat4 identity(1.0);
	const glm::vec3 angles = rotation + animation_rotation;
	return glm::translate(identity, position + offset) * glm::rotate(identity, glm::radians(angles.z), glm::vec3(0.0, 0.0, 1.0)) * glm::rotate(identity, glm::radians(angles.y), glm::vec3(0.0, 1.0, 0.0)) * glm::rotate(identity, glm::radians(angles.x), glm::vec3(1.0, 0.0, 0.0)) * glm::scale(identity, scale);
}

void update_uniforms() {
	const auto [width, height] = graphics::internal::context.swapchain_extent;
	if (!height) return;

	const float aspect = static_cast<float>(width) / static_cast<float>(height);
	glm::mat4 projection = controls.perspective ? glm::perspective(glm::radians(60.0), static_cast<double>(aspect), 0.1, 100.0) : glm::ortho(-3.0 * aspect, 3.0 * aspect, -3.0, 3.0, 0.1, 100.0);
	projection[1][1] *= -1.0;

	const float phase = controls.phase;
	const glm::vec3 trajectory(
		controls.radius * std::cos(phase),
		controls.height * std::sin(phase),
		controls.depth * std::sin(phase)
		);
	const glm::vec3 animated_rotation(
		30* std::sin(phase),
		glm::degrees(phase),
		std::cos(phase)
		);
	const glm::mat4 view = glm::lookAt(glm::vec3(0.0, 0.0, 7), glm::vec3(0.0,0.0,0.0), glm::vec3(0.0, 1.0, 0.0));

	const GlobalUniforms first_uniforms{
		.model = model_matrix(controls.first_position, controls.first_rotation, controls.first_scale, trajectory, animated_rotation),
		.view = view,
		.projection = projection,
		.color_multiplier = controls.first_color,
	};
	const GlobalUniforms second_uniforms{
		.model = model_matrix(controls.second_position, controls.second_rotation,controls.second_scale),
		.view = view,
		.projection = projection,
		.color_multiplier = controls.second_color,
	};

	std::memcpy(scene.first_uniform_buffer_mapped, &first_uniforms, sizeof(first_uniforms));
	vmaFlushAllocation(graphics::internal::context.allocator, scene.first_uniform_buffer_allocation, 0, sizeof(first_uniforms));
	std::memcpy(scene.second_uniform_buffer_mapped, &second_uniforms, sizeof(second_uniforms));
	vmaFlushAllocation(graphics::internal::context.allocator, scene.second_uniform_buffer_allocation, 0, sizeof(second_uniforms));
}

}

namespace application {

bool initialize() {
	if (!create_resources()) {
		destroy_resources();
		return false;
	}
	update_uniforms();
	return true;
}

void shutdown() {
	vkQueueWaitIdle(graphics::internal::context.graphics_queue);
	destroy_resources();
}

void update(const double time) {
	const float delta_time = previous_time == 0.0 ? 0.0 : static_cast<float>(time - previous_time);
	previous_time = time;
	if (controls.animation_playing) {
		controls.phase = std::fmod(controls.phase + delta_time * controls.animation_speed,2.0f * std::numbers::pi_v<float>);
	}

	ImGui::Begin("Lab 1");
	if (ImGui::RadioButton("Orthographic", !controls.perspective)) controls.perspective = false;
	ImGui::SameLine();
	if (ImGui::RadioButton("Perspective", controls.perspective)) controls.perspective = true;

	ImGui::Separator();
	ImGui::TextUnformatted("Primary cube");
	ImGui::PushID("Primary cube");
	ImGui::DragFloat3("Position", glm::value_ptr(controls.first_position), 0.02);
	ImGui::DragFloat3("Rotation", glm::value_ptr(controls.first_rotation), 0.5);
	ImGui::DragFloat3("Scale", glm::value_ptr(controls.first_scale), 0.01, 0.1, 2.0);
	ImGui::ColorEdit4("Color multiplier", glm::value_ptr(controls.first_color));
	ImGui::PopID();

	ImGui::Separator();
	ImGui::Checkbox("Play animation", &controls.animation_playing);
	ImGui::SameLine();
	if (ImGui::Button("Restart animation")) controls.phase = 0.0;
	ImGui::SliderFloat("Animation speed", &controls.animation_speed, 0.0, 4.0, "%.2fx");
	ImGui::SliderFloat("Trajectory radius", &controls.radius, 0.0, 2.0);
	ImGui::SliderFloat("Trajectory height", &controls.height, 0.0, 2.0);
	ImGui::SliderFloat("Trajectory depth", &controls.depth, 0.0, 2.0);

	ImGui::Separator();
	ImGui::TextUnformatted("Second cube");
	ImGui::PushID("Second cube");
	ImGui::DragFloat3("Position", glm::value_ptr(controls.second_position), 0.02);
	ImGui::DragFloat3("Rotation", glm::value_ptr(controls.second_rotation), 0.5);
	ImGui::DragFloat3("Scale", glm::value_ptr(controls.second_scale), 0.01, 0.1, 2.0);
	ImGui::ColorEdit4("Colour multiplier", glm::value_ptr(controls.second_color));
	ImGui::PopID();
	ImGui::End();
}

void render(const graphics::internal::FrameData& fd) {
	const auto& context = graphics::internal::context;
	update_uniforms();
	vkResetCommandBuffer(fd.command_buffer, 0);

	constexpr VkCommandBufferBeginInfo begin_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
	};

	vkBeginCommandBuffer(fd.command_buffer, &begin_info);

	constexpr VkClearValue clear_values[] = {
		{ .color = {{0.1, 0.1, 0.1, 1.0}} },
		{ .depthStencil = {.depth = 1.0, .stencil = 0} },
	};
	const VkRenderPassBeginInfo render_pass = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = context.render_pass,
		.framebuffer = fd.framebuffer,
		.renderArea = { .extent = context.swapchain_extent },
		.clearValueCount = std::size(clear_values),
		.pClearValues = clear_values,
	};
	vkCmdBeginRenderPass(fd.command_buffer, &render_pass, VK_SUBPASS_CONTENTS_INLINE);

	const VkViewport viewport = {
		.x = 0,.y =0,
		.width = static_cast<float>(context.swapchain_extent.width),
		.height = static_cast<float>(context.swapchain_extent.height),
		.minDepth = 0 ,.maxDepth = 1,
	};
	const VkRect2D scissor = { .extent = context.swapchain_extent };
	vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
	vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);
	vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, scene.pipeline);
	constexpr VkDeviceSize offset = 0;
	vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &scene.vertex_buffer, &offset);
	vkCmdBindIndexBuffer(fd.command_buffer, scene.index_buffer, 0, VK_INDEX_TYPE_UINT32);

	vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, scene.pipeline_layout, 0, 1, &scene.first_descriptor_set, 0, nullptr);
	vkCmdDrawIndexed(fd.command_buffer, std::size(cube_indices), 1, 0, 0, 0);

	vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, scene.pipeline_layout, 0, 1, &scene.second_descriptor_set, 0, nullptr);
	vkCmdDrawIndexed(fd.command_buffer, std::size(cube_indices), 1, 0, 0, 0);

	vkCmdEndRenderPass(fd.command_buffer);
	vkEndCommandBuffer(fd.command_buffer);
}

}
