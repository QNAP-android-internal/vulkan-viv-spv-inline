#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <spirv-tools/libspirv.h>
#include <vulkan/vk_layer.h>
#include <vulkan/vulkan.h>

#define EXPORT __attribute__((visibility("default")))

#define OP_ENTRY_POINT 15
#define OP_TYPE_IMAGE 25
#define OP_TYPE_SAMPLED_IMAGE 27
#define OP_TYPE_ARRAY 28
#define OP_TYPE_STRUCT 30
#define OP_TYPE_POINTER 32
#define OP_CONSTANT 43
#define OP_FUNCTION 54
#define OP_FUNCTION_PARAMETER 55
#define OP_FUNCTION_END 56
#define OP_FUNCTION_CALL 57
#define OP_VARIABLE 59
#define OP_COMPOSITE_CONSTRUCT 80
#define OP_COMPOSITE_EXTRACT 81

#define SC_UNIFORM_CONSTANT 0
#define FC_INLINE 0x1
#define FC_DONT_INLINE 0x2

#define K_PARAM 1
#define K_FROM_PARAM 2
#define K_CONSTRUCT 4
#define K_STRUCT_TYPE 8

static PFN_vkGetInstanceProcAddr next_gipa;
static PFN_vkGetDeviceProcAddr next_gdpa;
static VkInstance g_instance;
static uint32_t g_max_sampled_images = 16;

static int debug(void) { return getenv("VIV_SPV_INLINE_DEBUG") != NULL; }
static int enabled(const char *off_var) { return getenv(off_var) == NULL; }

struct func {
    uint32_t id;
    size_t control_at;
    int needed;
    int entry;
};

struct call {
    int caller;
    uint32_t callee;
};

struct plan {
    uint32_t *code;
    int inline_needed;
    int simplify_needed;
    int strip_needed;
};

static int func_index(struct func *f, int n, uint32_t id) {
    for (int i = 0; i < n; i++)
        if (f[i].id == id) return i;
    return -1;
}

static uint32_t image_count(const uint32_t *code, size_t words, uint32_t bound) {
    uint32_t *weight = calloc(bound, sizeof *weight);
    uint32_t *constant = calloc(bound, sizeof *constant);
    uint32_t *pointee = calloc(bound, sizeof *pointee);
    uint32_t total = 0;
    if (!weight || !constant || !pointee) goto out;
    for (size_t i = 5; i < words;) {
        uint32_t op = code[i] & 0xffff, len = code[i] >> 16;
        if (len == 0 || i + len > words) break;
        const uint32_t *w = &code[i];
        if (op == OP_TYPE_IMAGE && len > 7 && w[1] < bound && w[7] == 1) weight[w[1]] = 1;
        else if (op == OP_TYPE_SAMPLED_IMAGE && len > 2 && w[1] < bound) weight[w[1]] = 1;
        else if (op == OP_CONSTANT && len > 3 && w[2] < bound) constant[w[2]] = w[3];
        else if (op == OP_TYPE_ARRAY && len > 3 && w[1] < bound && w[2] < bound && w[3] < bound)
            weight[w[1]] = weight[w[2]] * constant[w[3]];
        else if (op == OP_TYPE_POINTER && len > 3 && w[1] < bound && w[3] < bound && w[2] == SC_UNIFORM_CONSTANT)
            pointee[w[1]] = w[3];
        else if (op == OP_VARIABLE && len > 3 && w[1] < bound && w[3] == SC_UNIFORM_CONSTANT && pointee[w[1]] < bound)
            total += weight[pointee[w[1]]];
        i += len;
    }
out:
    free(weight);
    free(constant);
    free(pointee);
    return total;
}

static int analyze(const uint32_t *code, size_t words, struct plan *p) {
    memset(p, 0, sizeof *p);
    if (words < 5) return 0;
    uint32_t bound = code[3];
    uint8_t *kind = calloc(bound, 1);
    struct func *f = calloc(words / 4 + 1, sizeof *f);
    struct call *c = calloc(words / 4 + 1, sizeof *c);
    uint32_t *entries = calloc(words / 4 + 1, sizeof *entries);
    int nf = 0, nc = 0, ne = 0, cur = -1, ok = 0;
    if (!kind || !f || !c || !entries) goto out;

    for (size_t i = 5; i < words;) {
        uint32_t op = code[i] & 0xffff, len = code[i] >> 16;
        if (len == 0 || i + len > words) goto out;
        const uint32_t *w = &code[i];
        if (op == OP_ENTRY_POINT && len > 2) {
            entries[ne++] = w[2];
        } else if (op == OP_FUNCTION && len > 3) {
            f[nf].id = w[2];
            f[nf].control_at = i + 3;
            cur = nf++;
        } else if (op == OP_FUNCTION_END) {
            cur = -1;
        } else if (op == OP_TYPE_STRUCT && len > 1 && w[1] < bound) {
            kind[w[1]] |= K_STRUCT_TYPE;
        } else if (op == OP_FUNCTION_PARAMETER && len > 2 && w[2] < bound) {
            kind[w[2]] |= K_PARAM;
        } else if (op == OP_COMPOSITE_EXTRACT && len > 3 && w[2] < bound && w[3] < bound) {
            if (kind[w[3]] & (K_PARAM | K_FROM_PARAM)) kind[w[2]] |= K_FROM_PARAM;
            if (kind[w[3]] & K_CONSTRUCT) p->simplify_needed = 1;
        } else if (op == OP_COMPOSITE_CONSTRUCT && len > 2 && w[2] < bound) {
            if (w[1] < bound && (kind[w[1]] & K_STRUCT_TYPE)) kind[w[2]] |= K_CONSTRUCT;
            for (uint32_t k = 3; k < len; k++)
                if (w[k] < bound && (kind[w[k]] & (K_PARAM | K_FROM_PARAM)) && cur >= 0) f[cur].needed = 1;
        } else if (op == OP_FUNCTION_CALL && len > 3 && cur >= 0) {
            c[nc].caller = cur;
            c[nc].callee = w[3];
            nc++;
        }
        i += len;
    }

    for (int i = 0; i < ne; i++) {
        int k = func_index(f, nf, entries[i]);
        if (k >= 0) f[k].entry = 1;
    }
    for (int changed = 1; changed;) {
        changed = 0;
        for (int i = 0; i < nc; i++) {
            int callee = func_index(f, nf, c[i].callee);
            int caller = c[i].caller;
            if (callee >= 0 && f[callee].needed && !f[caller].entry && !f[caller].needed) {
                f[caller].needed = 1;
                changed = 1;
            }
        }
    }
    for (int i = 0; i < nc; i++) {
        int callee = func_index(f, nf, c[i].callee);
        if (callee >= 0 && f[callee].needed) p->inline_needed = 1;
    }

    if (enabled("VIV_SPV_NO_STRIP") && image_count(code, words, bound) > g_max_sampled_images)
        p->strip_needed = 1;
    if (!enabled("VIV_SPV_NO_SIMPLIFY")) p->simplify_needed = 0;
    if (p->inline_needed && enabled("VIV_SPV_NO_SIMPLIFY")) p->simplify_needed = 1;

    if (p->inline_needed) {
        p->code = malloc(words * 4);
        if (!p->code) goto out;
        memcpy(p->code, code, words * 4);
        for (int i = 0; i < nf; i++) {
            uint32_t *ctl = &p->code[f[i].control_at];
            if (f[i].needed) *ctl &= ~FC_DONT_INLINE;
            else *ctl = (*ctl & ~FC_INLINE) | FC_DONT_INLINE;
        }
    }
    ok = 1;
out:
    free(kind);
    free(f);
    free(c);
    free(entries);
    return ok;
}

static spv_binary rewrite(const uint32_t *code, size_t words, const struct plan *p) {
    const char *flags[16];
    size_t n = 0;
    if (p->inline_needed) {
        flags[n++] = "--eliminate-dead-branches";
        flags[n++] = "--eliminate-dead-code-aggressive";
        flags[n++] = "--merge-return";
        flags[n++] = "--inline-entry-points-exhaustive";
        flags[n++] = "--eliminate-dead-functions";
    }
    if (p->simplify_needed) flags[n++] = "--simplify-instructions";
    if (p->strip_needed) {
        flags[n++] = "--remove-unused-interface-variables";
        flags[n++] = "--eliminate-dead-variables";
    }
    spv_binary out = NULL;
    spv_optimizer_t *opt = spvOptimizerCreate(SPV_ENV_VULKAN_1_3);
    spv_optimizer_options o = spvOptimizerOptionsCreate();
    spvOptimizerOptionsSetRunValidator(o, false);
    if (!spvOptimizerRegisterPassesFromFlags(opt, flags, n) ||
        spvOptimizerRun(opt, p->code ? p->code : code, words, &out, o) != SPV_SUCCESS)
        out = NULL;
    spvOptimizerOptionsDestroy(o);
    spvOptimizerDestroy(opt);
    return out;
}

static VKAPI_ATTR VkResult VKAPI_CALL
layer_CreateShaderModule(VkDevice device, const VkShaderModuleCreateInfo *info,
                         const VkAllocationCallbacks *alloc, VkShaderModule *module) {
    PFN_vkCreateShaderModule next =
        (PFN_vkCreateShaderModule)next_gdpa(device, "vkCreateShaderModule");
    size_t words = info->codeSize / 4;
    struct plan p;
    if (!analyze(info->pCode, words, &p) || !(p.inline_needed || p.simplify_needed || p.strip_needed)) {
        free(p.code);
        return next(device, info, alloc, module);
    }

    spv_binary rewritten = rewrite(info->pCode, words, &p);
    free(p.code);
    if (!rewritten) {
        fprintf(stderr, "viv_spv_inline: rewrite failed, passing module through (%zu words)\n", words);
        return next(device, info, alloc, module);
    }
    if (debug())
        fprintf(stderr, "viv_spv_inline: rewrote module %zu -> %zu words (inline=%d simplify=%d strip=%d)\n",
                words, rewritten->wordCount, p.inline_needed, p.simplify_needed, p.strip_needed);
    VkShaderModuleCreateInfo patched = *info;
    patched.codeSize = rewritten->wordCount * 4;
    patched.pCode = rewritten->code;
    VkResult r = next(device, &patched, alloc, module);
    spvBinaryDestroy(rewritten);
    return r;
}

struct image_info {
    VkImage image;
    uint32_t mip_levels;
    uint32_t array_layers;
};

static pthread_mutex_t images_lock = PTHREAD_MUTEX_INITIALIZER;
static struct image_info *images;
static size_t n_images, cap_images;

static VKAPI_ATTR VkResult VKAPI_CALL
layer_CreateImage(VkDevice device, const VkImageCreateInfo *info, const VkAllocationCallbacks *alloc, VkImage *image) {
    VkResult r = ((PFN_vkCreateImage)next_gdpa(device, "vkCreateImage"))(device, info, alloc, image);
    if (r != VK_SUCCESS) return r;
    pthread_mutex_lock(&images_lock);
    if (n_images == cap_images) {
        size_t cap = cap_images ? cap_images * 2 : 256;
        struct image_info *grown = realloc(images, cap * sizeof *grown);
        if (grown) {
            images = grown;
            cap_images = cap;
        }
    }
    if (n_images < cap_images)
        images[n_images++] = (struct image_info){ *image, info->mipLevels, info->arrayLayers };
    pthread_mutex_unlock(&images_lock);
    return r;
}

static VKAPI_ATTR void VKAPI_CALL
layer_DestroyImage(VkDevice device, VkImage image, const VkAllocationCallbacks *alloc) {
    pthread_mutex_lock(&images_lock);
    for (size_t i = 0; i < n_images; i++)
        if (images[i].image == image) {
            images[i] = images[--n_images];
            break;
        }
    pthread_mutex_unlock(&images_lock);
    ((PFN_vkDestroyImage)next_gdpa(device, "vkDestroyImage"))(device, image, alloc);
}

static VKAPI_ATTR VkResult VKAPI_CALL
layer_CreateImageView(VkDevice device, const VkImageViewCreateInfo *info, const VkAllocationCallbacks *alloc,
                      VkImageView *view) {
    PFN_vkCreateImageView next = (PFN_vkCreateImageView)next_gdpa(device, "vkCreateImageView");
    const VkImageSubresourceRange *r = &info->subresourceRange;
    if (!enabled("VIV_NO_MIP_FIX") ||
        (r->levelCount != VK_REMAINING_MIP_LEVELS && r->layerCount != VK_REMAINING_ARRAY_LAYERS))
        return next(device, info, alloc, view);

    VkImageViewCreateInfo patched = *info;
    pthread_mutex_lock(&images_lock);
    for (size_t i = 0; i < n_images; i++)
        if (images[i].image == info->image) {
            if (r->levelCount == VK_REMAINING_MIP_LEVELS && images[i].mip_levels > r->baseMipLevel)
                patched.subresourceRange.levelCount = images[i].mip_levels - r->baseMipLevel;
            if (r->layerCount == VK_REMAINING_ARRAY_LAYERS && images[i].array_layers > r->baseArrayLayer)
                patched.subresourceRange.layerCount = images[i].array_layers - r->baseArrayLayer;
            break;
        }
    pthread_mutex_unlock(&images_lock);
    return next(device, &patched, alloc, view);
}

static void fake_d24s8(VkFormat f, VkFormatProperties *p) {
    if (f == VK_FORMAT_D24_UNORM_S8_UINT && enabled("VIV_NO_D24S8_SAMPLED"))
        p->optimalTilingFeatures |= VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
}

static VKAPI_ATTR void VKAPI_CALL
layer_GetPhysicalDeviceFormatProperties(VkPhysicalDevice pd, VkFormat f, VkFormatProperties *p) {
    ((PFN_vkGetPhysicalDeviceFormatProperties)next_gipa(g_instance, "vkGetPhysicalDeviceFormatProperties"))(pd, f, p);
    fake_d24s8(f, p);
}

static VKAPI_ATTR void VKAPI_CALL
layer_GetPhysicalDeviceFormatProperties2(VkPhysicalDevice pd, VkFormat f, VkFormatProperties2 *p) {
    ((PFN_vkGetPhysicalDeviceFormatProperties2)next_gipa(g_instance, "vkGetPhysicalDeviceFormatProperties2"))(pd, f, p);
    fake_d24s8(f, &p->formatProperties);
}

static VKAPI_ATTR VkResult VKAPI_CALL
layer_GetPhysicalDeviceImageFormatProperties(VkPhysicalDevice pd, VkFormat f, VkImageType t, VkImageTiling tl,
                                             VkImageUsageFlags u, VkImageCreateFlags fl, VkImageFormatProperties *p) {
    PFN_vkGetPhysicalDeviceImageFormatProperties next =
        (PFN_vkGetPhysicalDeviceImageFormatProperties)next_gipa(g_instance, "vkGetPhysicalDeviceImageFormatProperties");
    VkResult r = next(pd, f, t, tl, u, fl, p);
    if (r == VK_ERROR_FORMAT_NOT_SUPPORTED && f == VK_FORMAT_D24_UNORM_S8_UINT &&
        (u & VK_IMAGE_USAGE_SAMPLED_BIT) && enabled("VIV_NO_D24S8_SAMPLED"))
        r = next(pd, f, t, tl, u & ~VK_IMAGE_USAGE_SAMPLED_BIT, fl, p);
    return r;
}

static VKAPI_ATTR VkResult VKAPI_CALL
layer_CreateInstance(const VkInstanceCreateInfo *info, const VkAllocationCallbacks *alloc,
                     VkInstance *instance) {
    VkLayerInstanceCreateInfo *chain = (VkLayerInstanceCreateInfo *)info->pNext;
    while (chain && !(chain->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO &&
                      chain->function == VK_LAYER_LINK_INFO))
        chain = (VkLayerInstanceCreateInfo *)chain->pNext;
    if (!chain) return VK_ERROR_INITIALIZATION_FAILED;
    next_gipa = chain->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    chain->u.pLayerInfo = chain->u.pLayerInfo->pNext;
    PFN_vkCreateInstance create = (PFN_vkCreateInstance)next_gipa(NULL, "vkCreateInstance");
    VkResult r = create(info, alloc, instance);
    if (r == VK_SUCCESS) g_instance = *instance;
    return r;
}

static VKAPI_ATTR VkResult VKAPI_CALL
layer_CreateDevice(VkPhysicalDevice phys, const VkDeviceCreateInfo *info,
                   const VkAllocationCallbacks *alloc, VkDevice *device) {
    VkLayerDeviceCreateInfo *chain = (VkLayerDeviceCreateInfo *)info->pNext;
    while (chain && !(chain->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO &&
                      chain->function == VK_LAYER_LINK_INFO))
        chain = (VkLayerDeviceCreateInfo *)chain->pNext;
    if (!chain) return VK_ERROR_INITIALIZATION_FAILED;
    PFN_vkGetInstanceProcAddr gipa = chain->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    next_gdpa = chain->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    chain->u.pLayerInfo = chain->u.pLayerInfo->pNext;
    VkPhysicalDeviceProperties props;
    ((PFN_vkGetPhysicalDeviceProperties)gipa(g_instance, "vkGetPhysicalDeviceProperties"))(phys, &props);
    g_max_sampled_images = props.limits.maxPerStageDescriptorSampledImages;
    PFN_vkCreateDevice create = (PFN_vkCreateDevice)gipa(NULL, "vkCreateDevice");
    return create(phys, info, alloc, device);
}

static PFN_vkVoidFunction device_hook(const char *name) {
    if (!strcmp(name, "vkCreateShaderModule")) return (PFN_vkVoidFunction)layer_CreateShaderModule;
    if (!strcmp(name, "vkCreateImage")) return (PFN_vkVoidFunction)layer_CreateImage;
    if (!strcmp(name, "vkDestroyImage")) return (PFN_vkVoidFunction)layer_DestroyImage;
    if (!strcmp(name, "vkCreateImageView")) return (PFN_vkVoidFunction)layer_CreateImageView;
    return NULL;
}

EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char *name);

EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vkGetInstanceProcAddr(VkInstance instance, const char *name) {
    if (!strcmp(name, "vkGetInstanceProcAddr")) return (PFN_vkVoidFunction)vkGetInstanceProcAddr;
    if (!strcmp(name, "vkGetDeviceProcAddr")) return (PFN_vkVoidFunction)vkGetDeviceProcAddr;
    if (!strcmp(name, "vkCreateInstance")) return (PFN_vkVoidFunction)layer_CreateInstance;
    if (!strcmp(name, "vkCreateDevice")) return (PFN_vkVoidFunction)layer_CreateDevice;
    if (!strcmp(name, "vkGetPhysicalDeviceFormatProperties"))
        return (PFN_vkVoidFunction)layer_GetPhysicalDeviceFormatProperties;
    if (!strcmp(name, "vkGetPhysicalDeviceFormatProperties2") || !strcmp(name, "vkGetPhysicalDeviceFormatProperties2KHR"))
        return (PFN_vkVoidFunction)layer_GetPhysicalDeviceFormatProperties2;
    if (!strcmp(name, "vkGetPhysicalDeviceImageFormatProperties"))
        return (PFN_vkVoidFunction)layer_GetPhysicalDeviceImageFormatProperties;
    PFN_vkVoidFunction hook = device_hook(name);
    if (hook) return hook;
    return next_gipa ? next_gipa(instance, name) : NULL;
}

EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vkGetDeviceProcAddr(VkDevice device, const char *name) {
    if (!strcmp(name, "vkGetDeviceProcAddr")) return (PFN_vkVoidFunction)vkGetDeviceProcAddr;
    PFN_vkVoidFunction hook = device_hook(name);
    if (hook) return hook;
    return next_gdpa(device, name);
}

EXPORT VKAPI_ATTR VkResult VKAPI_CALL
vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface *v) {
    if (v->loaderLayerInterfaceVersion > 2) v->loaderLayerInterfaceVersion = 2;
    v->pfnGetInstanceProcAddr = vkGetInstanceProcAddr;
    v->pfnGetDeviceProcAddr = vkGetDeviceProcAddr;
    v->pfnGetPhysicalDeviceProcAddr = NULL;
    return VK_SUCCESS;
}
