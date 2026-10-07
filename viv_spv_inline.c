#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <spirv-tools/libspirv.h>
#include <vulkan/vk_layer.h>
#include <vulkan/vulkan.h>

#define EXPORT __attribute__((visibility("default")))
#define SPV_OP_FUNCTION_PARAMETER 55

static PFN_vkGetInstanceProcAddr next_gipa;
static PFN_vkGetDeviceProcAddr next_gdpa;

static int debug(void) { return getenv("VIV_SPV_INLINE_DEBUG") != NULL; }

static int has_function_parameter(const uint32_t *code, size_t words) {
    for (size_t i = 5; i < words;) {
        uint32_t op = code[i] & 0xffff, len = code[i] >> 16;
        if (op == SPV_OP_FUNCTION_PARAMETER) return 1;
        if (len == 0) return 0;
        i += len;
    }
    return 0;
}

static spv_binary inline_all(const uint32_t *code, size_t words) {
    static const char *passes[] = {
        "--eliminate-dead-branches", "--eliminate-dead-code-aggressive",
        "--merge-return", "--inline-entry-points-exhaustive",
        "--eliminate-dead-functions" };
    spv_binary out = NULL;
    spv_optimizer_t *opt = spvOptimizerCreate(SPV_ENV_VULKAN_1_3);
    spv_optimizer_options o = spvOptimizerOptionsCreate();
    spvOptimizerOptionsSetRunValidator(o, false);
    if (!spvOptimizerRegisterPassesFromFlags(opt, passes, sizeof passes / sizeof *passes) ||
        spvOptimizerRun(opt, code, words, &out, o) != SPV_SUCCESS)
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
    if (!has_function_parameter(info->pCode, words))
        return next(device, info, alloc, module);

    spv_binary inlined = inline_all(info->pCode, words);
    if (!inlined) {
        fprintf(stderr, "viv_spv_inline: rewrite failed, passing module through (%zu words)\n", words);
        return next(device, info, alloc, module);
    }
    if (debug())
        fprintf(stderr, "viv_spv_inline: inlined module %zu -> %zu words\n", words, inlined->wordCount);
    VkShaderModuleCreateInfo patched = *info;
    patched.codeSize = inlined->wordCount * 4;
    patched.pCode = inlined->code;
    VkResult r = next(device, &patched, alloc, module);
    spvBinaryDestroy(inlined);
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
    return create(info, alloc, instance);
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
    PFN_vkCreateDevice create = (PFN_vkCreateDevice)gipa(NULL, "vkCreateDevice");
    return create(phys, info, alloc, device);
}

EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char *name);

EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vkGetInstanceProcAddr(VkInstance instance, const char *name) {
    if (!strcmp(name, "vkGetInstanceProcAddr")) return (PFN_vkVoidFunction)vkGetInstanceProcAddr;
    if (!strcmp(name, "vkGetDeviceProcAddr")) return (PFN_vkVoidFunction)vkGetDeviceProcAddr;
    if (!strcmp(name, "vkCreateInstance")) return (PFN_vkVoidFunction)layer_CreateInstance;
    if (!strcmp(name, "vkCreateDevice")) return (PFN_vkVoidFunction)layer_CreateDevice;
    if (!strcmp(name, "vkCreateShaderModule")) return (PFN_vkVoidFunction)layer_CreateShaderModule;
    return next_gipa ? next_gipa(instance, name) : NULL;
}

EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vkGetDeviceProcAddr(VkDevice device, const char *name) {
    if (!strcmp(name, "vkGetDeviceProcAddr")) return (PFN_vkVoidFunction)vkGetDeviceProcAddr;
    if (!strcmp(name, "vkCreateShaderModule")) return (PFN_vkVoidFunction)layer_CreateShaderModule;
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
