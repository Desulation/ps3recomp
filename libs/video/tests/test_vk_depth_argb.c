/* The Vulkan backend's depth-as-A8R8G8B8 resolve (vulkan/rsx_vk_depth_argb.comp)
 * against the guest-memory route it replaces, for every 32-bit pattern a
 * depth snapshot texel can hold: zero, denormals, every float in (0, 1), 1,
 * above 1, the infinities, every NaN, every negative.
 *
 * The expected texel is the draw engine's own route, written out: the clamp
 * and the double-precision quantization of eng_zdepth_to_guest, the Z24S8
 * guest bytes, then rsx_texture_decode as a linear A8R8G8B8 texture (both
 * byte orders, rsx_texture_argb_is_rgba). The GPU side is the shader the
 * backend embeds, on images made as the backend makes them: an R32_SFLOAT
 * snapshot and a MUTABLE R8G8B8A8_UNORM result, both read and written through
 * R32_UINT storage views, the result copied out as the bytes of its texels.
 * Needs a Vulkan device (lavapipe will do; VK_ICD_FILENAMES picks one).
 *
 *   cc -std=gnu17 -O2 -I include -I libs/video libs/video/tests/test_vk_depth_argb.c \
 *      libs/video/rsx_texture_layout.c -lvulkan -o /tmp/test_vk_depth_argb
 *   /tmp/test_vk_depth_argb            # all 2^32 patterns, both byte orders
 *   /tmp/test_vk_depth_argb --quick    # every 257th pattern, one dispatch
 */
#include <vulkan/vulkan.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rsx_texture_layout.h"
#include "vulkan/rsx_vk_depth_argb_comp.spv.h"

#define SIDE   4096u                    /* one dispatch: 2^24 patterns */
#define TEXELS (SIDE * SIDE)

#define CHECK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { \
    fprintf(stderr, "%s:%d: %s -> %d\n", __FILE__, __LINE__, #x, (int)r_); exit(2); } } while (0)

static VkDevice dev;
static VkPhysicalDevice phys;

static uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags want)
{
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    fprintf(stderr, "no memory type %x\n", (unsigned)want);
    exit(2);
}

static VkImage image(VkFormat fmt, VkImageUsageFlags usage, VkDeviceMemory* mem)
{
    VkImageCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT,
        .imageType = VK_IMAGE_TYPE_2D, .format = fmt, .extent = { SIDE, SIDE, 1 },
        .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = usage,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED };
    VkImage img;
    CHECK(vkCreateImage(dev, &ici, NULL, &img));
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(dev, img, &req);
    VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size,
                                 .memoryTypeIndex = memory_type(req.memoryTypeBits, 0) };
    CHECK(vkAllocateMemory(dev, &mai, NULL, mem));
    CHECK(vkBindImageMemory(dev, img, *mem, 0));
    return img;
}

static VkImageView uint_view(VkImage img)
{
    VkImageViewCreateInfo vci = { .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img,
                                  .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_R32_UINT,
                                  .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
    VkImageView v;
    CHECK(vkCreateImageView(dev, &vci, NULL, &v));
    return v;
}

static VkBuffer host_buffer(VkBufferUsageFlags usage, VkDeviceMemory* mem, void** ptr)
{
    VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = TEXELS * 4ull,
                               .usage = usage };
    VkBuffer b;
    CHECK(vkCreateBuffer(dev, &bci, NULL, &b));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, b, &req);
    VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size,
                                 .memoryTypeIndex = memory_type(req.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) };
    CHECK(vkAllocateMemory(dev, &mai, NULL, mem));
    CHECK(vkBindBufferMemory(dev, b, *mem, 0));
    CHECK(vkMapMemory(dev, *mem, 0, VK_WHOLE_SIZE, 0, ptr));
    return b;
}

static void barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from)
{
    VkImageMemoryBarrier b = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        .oldLayout = from, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = img, .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         0, 0, NULL, 0, NULL, 1, &b);
}

/* eng_zdepth_to_guest's Z24S8 bytes for one depth value. */
static void guest_z24s8(uint32_t bits, uint8_t out[4])
{
    float d;
    memcpy(&d, &bits, 4);
    if (!(d > 0.0f)) d = 0.0f;
    if (d > 1.0f) d = 1.0f;
    const uint32_t v = (uint32_t)((double)d * 16777215.0 + 0.5);
    out[0] = (uint8_t)(v >> 16); out[1] = (uint8_t)(v >> 8); out[2] = (uint8_t)v; out[3] = 0;
}

int main(int argc, char** argv)
{
    const int quick = argc > 1 && !strcmp(argv[1], "--quick");
    /* Patterns between two tested ones, and dispatches of TEXELS of them. */
    const uint32_t step = quick ? 257u : 1u;
    const uint32_t chunks = quick ? 1u : (uint32_t)((1ull << 32) / TEXELS);

    VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_0 };
    VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app };
    VkInstance inst;
    CHECK(vkCreateInstance(&ici, NULL, &inst));
    uint32_t n = 8;
    VkPhysicalDevice pds[8];
    vkEnumeratePhysicalDevices(inst, &n, pds);
    if (!n) { fprintf(stderr, "no Vulkan device\n"); return 2; }
    const char* pick = getenv("TEST_VK_DEVICE");
    phys = pds[pick ? (uint32_t)atoi(pick) % n : 0];
    VkPhysicalDeviceProperties pp;
    vkGetPhysicalDeviceProperties(phys, &pp);
    uint32_t nq = 16, qf = UINT32_MAX;
    VkQueueFamilyProperties qp[16];
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, qp);
    for (uint32_t i = 0; i < nq && qf == UINT32_MAX; i++)
        if ((qp[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) ==
            (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) qf = i;
    if (qf == UINT32_MAX) { fprintf(stderr, "no graphics+compute queue\n"); return 2; }
    printf("device: %s\n", pp.deviceName);

    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                    .queueFamilyIndex = qf, .queueCount = 1, .pQueuePriorities = &prio };
    VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                               .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci };
    CHECK(vkCreateDevice(phys, &dci, NULL, &dev));
    VkQueue q;
    vkGetDeviceQueue(dev, qf, 0, &q);

    VkDeviceMemory m_src, m_dst, m_up, m_down;
    VkImage src = image(VK_FORMAT_R32_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                              VK_IMAGE_USAGE_STORAGE_BIT, &m_src);
    VkImage dst = image(VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                                                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT, &m_dst);
    VkImageView v_src = uint_view(src), v_dst = uint_view(dst);
    uint32_t* up;
    uint8_t* down;
    VkBuffer b_up = host_buffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &m_up, (void**)&up);
    VkBuffer b_down = host_buffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT, &m_down, (void**)&down);

    /* As vk_dargb_init builds them. */
    const VkDescriptorSetLayoutBinding lb[2] = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL } };
    VkDescriptorSetLayoutCreateInfo lci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                            .bindingCount = 2, .pBindings = lb };
    VkDescriptorSetLayout sl;
    CHECK(vkCreateDescriptorSetLayout(dev, &lci, NULL, &sl));
    const VkPushConstantRange pr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 12 };
    VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                        .setLayoutCount = 1, .pSetLayouts = &sl,
                                        .pushConstantRangeCount = 1, .pPushConstantRanges = &pr };
    VkPipelineLayout pl;
    CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));
    VkShaderModuleCreateInfo smci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                      .codeSize = sizeof rsx_vk_depth_argb_comp_spv,
                                      .pCode = rsx_vk_depth_argb_comp_spv };
    VkShaderModule sm;
    CHECK(vkCreateShaderModule(dev, &smci, NULL, &sm));
    VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                   .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main" },
        .layout = pl };
    VkPipeline pipe;
    CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));
    const VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 };
    VkDescriptorPoolCreateInfo dpci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                        .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps };
    VkDescriptorPool dp;
    CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &dp));
    VkDescriptorSetAllocateInfo dsai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                         .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &sl };
    VkDescriptorSet set;
    CHECK(vkAllocateDescriptorSets(dev, &dsai, &set));
    const VkDescriptorImageInfo ii[2] = { { VK_NULL_HANDLE, v_src, VK_IMAGE_LAYOUT_GENERAL },
                                          { VK_NULL_HANDLE, v_dst, VK_IMAGE_LAYOUT_GENERAL } };
    VkWriteDescriptorSet wr = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set,
                                .descriptorCount = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                .pImageInfo = ii };
    vkUpdateDescriptorSets(dev, 1, &wr, 0, NULL);

    VkCommandPoolCreateInfo cpi = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                    .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                    .queueFamilyIndex = qf };
    VkCommandPool cp;
    CHECK(vkCreateCommandPool(dev, &cpi, NULL, &cp));
    VkCommandBufferAllocateInfo cbai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                         .commandPool = cp, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                         .commandBufferCount = 1 };
    VkCommandBuffer cb;
    CHECK(vkAllocateCommandBuffers(dev, &cbai, &cb));
    VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence fence;
    CHECK(vkCreateFence(dev, &fci, NULL, &fence));

    /* The route being replaced, as the engine runs it: Z24S8 guest bytes at a
     * pitch of SIDE * 4, decoded as a linear A8R8G8B8 texture. */
    rsx_tex_level lv[RSX_MAX_TEXTURE_LEVELS];
    if (!rsx_texture_mip_chain(0x85u | 0x20u, SIDE, SIDE, 1, SIDE * 4u, lv) ||
        lv[0].tl.dst_row_bytes != SIDE * 4u) { fprintf(stderr, "unexpected layout\n"); return 2; }
    uint8_t* guest = (uint8_t*)malloc(TEXELS * 4ull);
    uint8_t* want = (uint8_t*)malloc(TEXELS * 4ull);
    if (!guest || !want) return 2;

    unsigned long long tested = 0, bad = 0;
    int first_layout = 1;
    for (int rgba = 0; rgba < 2; rgba++) {
        /* The guest-order variant (TEX_RGBA) is a different packing of the
         * same value: a sample of it is enough. */
        const uint32_t chunk_step = rgba ? 16u : 1u;
        for (uint32_t chunk = 0; chunk < chunks; chunk += chunk_step) {
            const uint64_t base = (uint64_t)chunk * TEXELS * step;
            for (uint32_t i = 0; i < TEXELS; i++) {
                const uint32_t bits = (uint32_t)(base + (uint64_t)i * step);
                up[i] = bits;
                guest_z24s8(bits, guest + (size_t)i * 4u);
            }
            CHECK(vkResetCommandBuffer(cb, 0));
            VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
            CHECK(vkBeginCommandBuffer(cb, &bi));
            barrier(cb, src, first_layout ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL);
            barrier(cb, dst, first_layout ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL);
            first_layout = 0;
            VkBufferImageCopy region = { .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                                         .imageExtent = { SIDE, SIDE, 1 } };
            vkCmdCopyBufferToImage(cb, b_up, src, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
            barrier(cb, src, VK_IMAGE_LAYOUT_GENERAL);
            const uint32_t pc[3] = { SIDE, SIDE, (uint32_t)rgba };
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &set, 0, NULL);
            vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
            vkCmdDispatch(cb, SIDE / 8u, SIDE / 8u, 1);
            barrier(cb, dst, VK_IMAGE_LAYOUT_GENERAL);
            vkCmdCopyImageToBuffer(cb, dst, VK_IMAGE_LAYOUT_GENERAL, b_down, 1, &region);
            VkMemoryBarrier hb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                   .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                                   .dstAccessMask = VK_ACCESS_HOST_READ_BIT };
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                                 0, 1, &hb, 0, NULL, 0, NULL);
            CHECK(vkEndCommandBuffer(cb));
            VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1,
                                .pCommandBuffers = &cb };
            CHECK(vkQueueSubmit(q, 1, &si, fence));
            /* The expected texels while the GPU works. */
            rsx_texture_decode(want, SIDE * 4u, guest, SIDE, SIDE, &lv[0].tl, rgba);
            CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
            CHECK(vkResetFences(dev, 1, &fence));
            for (uint32_t i = 0; i < TEXELS; i++) {
                if (memcmp(down + (size_t)i * 4u, want + (size_t)i * 4u, 4) == 0) continue;
                if (bad++ < 10) {
                    const uint8_t* g = down + (size_t)i * 4u;
                    const uint8_t* w = want + (size_t)i * 4u;
                    printf("FAIL %s bits 0x%08X: gpu %02X %02X %02X %02X, want %02X %02X %02X %02X\n",
                           rgba ? "rgba" : "argb", up[i], g[0], g[1], g[2], g[3], w[0], w[1], w[2], w[3]);
                }
            }
            tested += TEXELS;
        }
    }
    printf("%llu patterns checked, %llu differ\n", tested, bad);
    return bad ? 1 : 0;
}
