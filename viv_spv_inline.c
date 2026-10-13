#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <spirv-tools/libspirv.h>
#include <vulkan/vk_layer.h>
#include <vulkan/vulkan.h>

#define EXPORT __attribute__((visibility("default")))

#define OP_EXT_INST_IMPORT 11
#define OP_EXT_INST 12
#define OP_ENTRY_POINT 15
#define OP_TYPE_VECTOR 23
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
#define OP_DECORATE 71
#define OP_ACCESS_CHAIN 65
#define OP_IN_BOUNDS_ACCESS_CHAIN 66
#define OP_COMPOSITE_CONSTRUCT 80
#define OP_COMPOSITE_EXTRACT 81
#define OP_F_ADD 129
#define OP_F_SUB 131
#define OP_VECTOR_TIMES_SCALAR 142
#define GLSL_FMIX 46

#define SC_UNIFORM_CONSTANT 0
#define DECORATION_BINDING 33
#define DECORATION_DESCRIPTOR_SET 34
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
    int mix_needed;
    int chain_needed;
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


static int has_type_and_id(uint32_t op) {
    if (op >= 109 && op <= 148) return 1;
    switch (op) {
    case 1: case OP_EXT_INST: case 41: case 42: case OP_CONSTANT: case 44: case 45:
    case OP_FUNCTION_PARAMETER: case OP_VARIABLE: case 61: case 65: case 66: case 79:
    case OP_COMPOSITE_CONSTRUCT: case OP_COMPOSITE_EXTRACT: case 82: case 83: case 169: case 245:
        return 1;
    }
    return 0;
}

static uint32_t *premix(const uint32_t *code, size_t *words_io, int *count) {
    size_t words = *words_io;
    *count = 0;
    if (words < 5) return NULL;
    uint32_t bound = code[3], glsl = 0, *out = NULL;
    uint32_t *rtype = calloc(bound, sizeof *rtype), *vcomp = calloc(bound, sizeof *vcomp);
    size_t *def_at = calloc(bound, sizeof *def_at), *at = calloc(words / 8 + 1, sizeof *at);
    uint8_t *isvec = calloc(bound, 1);
    int n = 0;
    if (!rtype || !vcomp || !def_at || !at || !isvec) goto out;
    for (size_t i = 5; i < words;) {
        uint32_t op = code[i] & 0xffff, len = code[i] >> 16;
        if (len == 0 || i + len > words) goto out;
        const uint32_t *w = &code[i];
        if (op == OP_EXT_INST_IMPORT && len > 5 && w[1] < bound && !memcmp(&w[2], "GLSL.std.450", 13)) glsl = w[1];
        else if (op == OP_TYPE_VECTOR && len > 3 && w[1] < bound) { isvec[w[1]] = 1; vcomp[w[1]] = w[2]; }
        else if (has_type_and_id(op) && len > 2 && w[2] < bound) { rtype[w[2]] = w[1]; def_at[w[2]] = i; }
        i += len;
    }
    if (!glsl) goto out;
    for (size_t i = 5; i < words;) {
        uint32_t op = code[i] & 0xffff, len = code[i] >> 16;
        const uint32_t *w = &code[i];
        if (op == OP_EXT_INST && len == 8 && w[3] == glsl && w[4] == GLSL_FMIX && w[1] < bound && isvec[w[1]] &&
            w[7] < bound && def_at[w[7]]) {
            const uint32_t *c = &code[def_at[w[7]]];
            uint32_t cop = c[0] & 0xffff, clen = c[0] >> 16;
            if (cop == OP_COMPOSITE_CONSTRUCT && clen >= 4 && c[1] == w[1] && c[3] < bound && rtype[c[3]] == vcomp[w[1]]) {
                int splat = 1;
                for (uint32_t k = 4; k < clen; k++) if (c[k] != c[3]) splat = 0;
                if (splat) at[n++] = i;
            }
        }
        i += len;
    }
    if (!n) goto out;
    out = malloc((words + 7 * n) * 4);
    if (!out) goto out;
    memcpy(out, code, 5 * 4);
    out[3] = bound + 2 * n;
    size_t o = 5;
    int k = 0;
    uint32_t next = bound;
    for (size_t i = 5; i < words;) {
        uint32_t len = code[i] >> 16;
        const uint32_t *w = &code[i];
        if (k < n && at[k] == i) {
            uint32_t T = w[1], Y = w[2], a = w[5], b = w[6], s = code[def_at[w[7]] + 3], t1 = next++, t2 = next++;
            uint32_t ins[15] = { (5u << 16) | OP_F_SUB, T, t1, b, a,
                                 (5u << 16) | OP_VECTOR_TIMES_SCALAR, T, t2, t1, s,
                                 (5u << 16) | OP_F_ADD, T, Y, a, t2 };
            memcpy(&out[o], ins, sizeof ins);
            o += 15;
            k++;
        } else {
            memcpy(&out[o], w, len * 4);
            o += len;
        }
        i += len;
    }
    *words_io = o;
    *count = n;
out:
    free(rtype); free(vcomp); free(def_at); free(at); free(isvec);
    return out;
}

static int analyze(const uint32_t *code, size_t words, struct plan *p) {
    memset(p, 0, sizeof *p);
    if (words < 5) return 0;
    uint32_t bound = code[3];
    uint8_t *kind = calloc(bound, 1), *isconst = calloc(bound, 1), *chain = calloc(bound, 1), *isptr = calloc(bound, 1);
    struct func *f = calloc(words / 4 + 1, sizeof *f);
    struct call *c = calloc(words / 4 + 1, sizeof *c);
    uint32_t *entries = calloc(words / 4 + 1, sizeof *entries);
    int nf = 0, nc = 0, ne = 0, cur = -1, ok = 0;
    if (!kind || !isconst || !chain || !isptr || !f || !c || !entries) goto out;

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
        } else if (op == OP_TYPE_POINTER && len > 1 && w[1] < bound) {
            isptr[w[1]] = 1;
        } else if (op == OP_TYPE_STRUCT && len > 1 && w[1] < bound) {
            kind[w[1]] |= K_STRUCT_TYPE;
        } else if (op == OP_FUNCTION_PARAMETER && len > 2 && w[2] < bound) {
            kind[w[2]] |= K_PARAM;
            if (cur >= 0 && !(w[1] < bound && isptr[w[1]])) f[cur].needed = 1;
        } else if (op == OP_COMPOSITE_EXTRACT && len > 3 && w[2] < bound && w[3] < bound) {
            if (kind[w[3]] & (K_PARAM | K_FROM_PARAM)) kind[w[2]] |= K_FROM_PARAM;
            if (kind[w[3]] & K_CONSTRUCT) p->simplify_needed = 1;
        } else if (op == OP_COMPOSITE_CONSTRUCT && len > 2 && w[2] < bound) {
            if (w[1] < bound && (kind[w[1]] & K_STRUCT_TYPE)) kind[w[2]] |= K_CONSTRUCT;
            for (uint32_t k = 3; k < len; k++)
                if (w[k] < bound && (kind[w[k]] & (K_PARAM | K_FROM_PARAM)) && cur >= 0) f[cur].needed = 1;
        } else if (((op >= 41 && op <= 46) || (op >= 48 && op <= 52)) && len > 2 && w[2] < bound) {
            isconst[w[2]] = 1;
        } else if ((op == OP_ACCESS_CHAIN || op == OP_IN_BOUNDS_ACCESS_CHAIN) && len > 3 && w[2] < bound && w[3] < bound) {
            uint8_t dyn = chain[w[3]] & 2;
            for (uint32_t k = 4; k < len; k++)
                if (w[k] >= bound || !isconst[w[k]]) dyn = 2;
            if ((chain[w[3]] & 1) && dyn) p->chain_needed = 1;
            chain[w[2]] = 1 | dyn;
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
    if (!enabled("VIV_SPV_NO_COMBINE")) p->chain_needed = 0;
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
    free(isconst);
    free(chain);
    free(isptr);
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
    if (p->mix_needed && !p->inline_needed) flags[n++] = "--eliminate-dead-code-aggressive";
    if (p->chain_needed) flags[n++] = "--combine-access-chains";
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

struct binding_ref {
    uint32_t set;
    uint32_t binding;
};

struct module_info {
    VkShaderModule module;
    struct binding_ref *bindings;
    uint32_t n;
};

struct set_info {
    VkDescriptorSetLayout layout;
    VkDescriptorSetLayoutCreateFlags flags;
    VkDescriptorSetLayoutBinding *bindings;
    VkSampler *samplers;
    uint32_t n;
};

struct pipeline_layout_info {
    VkPipelineLayout layout;
    VkPipelineLayoutCreateFlags flags;
    VkDescriptorSetLayout *sets;
    uint32_t n_sets;
    VkPushConstantRange *push;
    uint32_t n_push;
};

static pthread_mutex_t layouts_lock = PTHREAD_MUTEX_INITIALIZER;
static struct module_info *modules;
static size_t n_modules, cap_modules;
static struct set_info *sets;
static size_t n_sets, cap_sets;
static struct pipeline_layout_info *pipeline_layouts;
static size_t n_pipeline_layouts, cap_pipeline_layouts;

#define PUSH(arr, n, cap, item) do { \
    if (n == cap) { \
        size_t new_cap = cap ? cap * 2 : 64; \
        void *grown = realloc(arr, new_cap * sizeof *arr); \
        if (grown) { arr = grown; cap = new_cap; } \
    } \
    if (n < cap) arr[n++] = item; \
} while (0)

static void push_set(struct set_info s) { PUSH(sets, n_sets, cap_sets, s); }
static void push_pipeline_layout(struct pipeline_layout_info p) { PUSH(pipeline_layouts, n_pipeline_layouts, cap_pipeline_layouts, p); }

static void record_module(VkShaderModule module, const uint32_t *code, size_t words) {
    if (words < 5) return;
    uint32_t bound = code[3];
    uint32_t *set = malloc(bound * sizeof *set), *binding = malloc(bound * sizeof *binding);
    if (!set || !binding) {
        free(set);
        free(binding);
        return;
    }
    memset(set, 0xff, bound * sizeof *set);
    memset(binding, 0xff, bound * sizeof *binding);
    for (size_t i = 5; i < words;) {
        const uint32_t *w = code + i;
        uint16_t op = w[0] & 0xffff, len = w[0] >> 16;
        if (len == 0 || i + len > words) break;
        if (op == OP_DECORATE && len == 4 && w[1] < bound) {
            if (w[2] == DECORATION_BINDING) binding[w[1]] = w[3];
            else if (w[2] == DECORATION_DESCRIPTOR_SET) set[w[1]] = w[3];
        }
        i += len;
    }
    struct module_info m = { module, calloc(bound, sizeof *m.bindings), 0 };
    for (uint32_t id = 0; id < bound && m.bindings; id++)
        if (set[id] != 0xffffffff && binding[id] != 0xffffffff)
            m.bindings[m.n++] = (struct binding_ref){ set[id], binding[id] };
    free(set);
    free(binding);
    pthread_mutex_lock(&layouts_lock);
    PUSH(modules, n_modules, cap_modules, m);
    pthread_mutex_unlock(&layouts_lock);
}

static VKAPI_ATTR VkResult VKAPI_CALL
layer_CreateShaderModule(VkDevice device, const VkShaderModuleCreateInfo *info,
                         const VkAllocationCallbacks *alloc, VkShaderModule *module) {
    PFN_vkCreateShaderModule next =
        (PFN_vkCreateShaderModule)next_gdpa(device, "vkCreateShaderModule");
    size_t words = info->codeSize / 4, orig_words = words;
    int nmix = 0;
    uint32_t *pre = enabled("VIV_SPV_NO_MIX") ? premix(info->pCode, &words, &nmix) : NULL;
    const uint32_t *cur = pre ? pre : info->pCode;
    struct plan p;
    int ok = analyze(cur, words, &p);
    p.mix_needed = nmix > 0;
    if (!ok || !(p.inline_needed || p.simplify_needed || p.strip_needed || p.mix_needed || p.chain_needed)) {
        free(p.code);
        free(pre);
        VkResult r = next(device, info, alloc, module);
        if (r == VK_SUCCESS) record_module(*module, info->pCode, orig_words);
        return r;
    }

    spv_binary rewritten = rewrite(cur, words, &p);
    free(p.code);
    free(pre);
    if (!rewritten) {
        fprintf(stderr, "viv_spv_inline: rewrite failed, passing module through (%zu words)\n", orig_words);
        VkResult r = next(device, info, alloc, module);
        if (r == VK_SUCCESS) record_module(*module, info->pCode, orig_words);
        return r;
    }
    if (debug())
        fprintf(stderr, "viv_spv_inline: rewrote module %zu -> %zu words (inline=%d simplify=%d strip=%d mix=%d chain=%d)\n",
                orig_words, rewritten->wordCount, p.inline_needed, p.simplify_needed, p.strip_needed, nmix, p.chain_needed);
    VkShaderModuleCreateInfo patched = *info;
    patched.codeSize = rewritten->wordCount * 4;
    patched.pCode = rewritten->code;
    VkResult r = next(device, &patched, alloc, module);
    if (r == VK_SUCCESS) record_module(*module, rewritten->code, rewritten->wordCount);
    spvBinaryDestroy(rewritten);
    return r;
}

static VKAPI_ATTR void VKAPI_CALL
layer_DestroyShaderModule(VkDevice device, VkShaderModule module, const VkAllocationCallbacks *alloc) {
    pthread_mutex_lock(&layouts_lock);
    for (size_t i = 0; i < n_modules; i++)
        if (modules[i].module == module) {
            free(modules[i].bindings);
            modules[i] = modules[--n_modules];
            break;
        }
    pthread_mutex_unlock(&layouts_lock);
    ((PFN_vkDestroyShaderModule)next_gdpa(device, "vkDestroyShaderModule"))(device, module, alloc);
}

static VKAPI_ATTR VkResult VKAPI_CALL
layer_CreateDescriptorSetLayout(VkDevice device, const VkDescriptorSetLayoutCreateInfo *info,
                                const VkAllocationCallbacks *alloc, VkDescriptorSetLayout *layout) {
    VkResult r = ((PFN_vkCreateDescriptorSetLayout)next_gdpa(device, "vkCreateDescriptorSetLayout"))(device, info, alloc, layout);
    if (r != VK_SUCCESS) return r;
    struct set_info s = { *layout, info->flags, NULL, NULL, info->bindingCount };
    s.bindings = calloc(s.n ? s.n : 1, sizeof *s.bindings);
    size_t n_samplers = 0;
    for (uint32_t i = 0; i < s.n; i++)
        if (info->pBindings[i].pImmutableSamplers) n_samplers += info->pBindings[i].descriptorCount;
    s.samplers = calloc(n_samplers ? n_samplers : 1, sizeof *s.samplers);
    n_samplers = 0;
    for (uint32_t i = 0; i < s.n; i++) {
        s.bindings[i] = info->pBindings[i];
        if (s.bindings[i].pImmutableSamplers) {
            memcpy(s.samplers + n_samplers, s.bindings[i].pImmutableSamplers, s.bindings[i].descriptorCount * sizeof *s.samplers);
            s.bindings[i].pImmutableSamplers = s.samplers + n_samplers;
            n_samplers += s.bindings[i].descriptorCount;
        }
    }
    pthread_mutex_lock(&layouts_lock);
    push_set(s);
    pthread_mutex_unlock(&layouts_lock);
    return r;
}

static VKAPI_ATTR void VKAPI_CALL
layer_DestroyDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout layout, const VkAllocationCallbacks *alloc) {
    pthread_mutex_lock(&layouts_lock);
    for (size_t i = 0; i < n_sets; i++)
        if (sets[i].layout == layout) {
            free(sets[i].bindings);
            free(sets[i].samplers);
            sets[i] = sets[--n_sets];
            break;
        }
    pthread_mutex_unlock(&layouts_lock);
    ((PFN_vkDestroyDescriptorSetLayout)next_gdpa(device, "vkDestroyDescriptorSetLayout"))(device, layout, alloc);
}

static VKAPI_ATTR VkResult VKAPI_CALL
layer_CreatePipelineLayout(VkDevice device, const VkPipelineLayoutCreateInfo *info,
                           const VkAllocationCallbacks *alloc, VkPipelineLayout *layout) {
    VkResult r = ((PFN_vkCreatePipelineLayout)next_gdpa(device, "vkCreatePipelineLayout"))(device, info, alloc, layout);
    if (r != VK_SUCCESS) return r;
    struct pipeline_layout_info p = { *layout, info->flags, NULL, info->setLayoutCount, NULL, info->pushConstantRangeCount };
    p.sets = calloc(p.n_sets ? p.n_sets : 1, sizeof *p.sets);
    p.push = calloc(p.n_push ? p.n_push : 1, sizeof *p.push);
    memcpy(p.sets, info->pSetLayouts, p.n_sets * sizeof *p.sets);
    memcpy(p.push, info->pPushConstantRanges, p.n_push * sizeof *p.push);
    pthread_mutex_lock(&layouts_lock);
    push_pipeline_layout(p);
    pthread_mutex_unlock(&layouts_lock);
    return r;
}

static VKAPI_ATTR void VKAPI_CALL
layer_DestroyPipelineLayout(VkDevice device, VkPipelineLayout layout, const VkAllocationCallbacks *alloc) {
    pthread_mutex_lock(&layouts_lock);
    for (size_t i = 0; i < n_pipeline_layouts; i++)
        if (pipeline_layouts[i].layout == layout) {
            free(pipeline_layouts[i].sets);
            free(pipeline_layouts[i].push);
            pipeline_layouts[i] = pipeline_layouts[--n_pipeline_layouts];
            break;
        }
    pthread_mutex_unlock(&layouts_lock);
    ((PFN_vkDestroyPipelineLayout)next_gdpa(device, "vkDestroyPipelineLayout"))(device, layout, alloc);
}

static int module_uses(const struct module_info *m, uint32_t set, uint32_t binding) {
    for (uint32_t i = 0; i < m->n; i++)
        if (m->bindings[i].set == set && m->bindings[i].binding == binding) return 1;
    return 0;
}

static VkPipelineLayout trimmed_layout(VkDevice device, const VkGraphicsPipelineCreateInfo *info) {
    const struct module_info *stage[8];
    uint32_t n_stage = 0;
    VkPipelineLayout result = VK_NULL_HANDLE;
    pthread_mutex_lock(&layouts_lock);
    for (uint32_t i = 0; i < info->stageCount && n_stage < 8; i++)
        for (size_t j = 0; j < n_modules; j++)
            if (modules[j].module == info->pStages[i].module) stage[n_stage++] = &modules[j];
    const struct pipeline_layout_info *pl = NULL;
    for (size_t i = 0; i < n_pipeline_layouts; i++)
        if (pipeline_layouts[i].layout == info->layout) pl = &pipeline_layouts[i];
    if (!pl || n_stage != info->stageCount) goto out;

    VkDescriptorSetLayout new_sets[16];
    int changed = 0;
    for (uint32_t s = 0; s < pl->n_sets && s < 16; s++) {
        const struct set_info *set = NULL;
        for (size_t i = 0; i < n_sets; i++)
            if (sets[i].layout == pl->sets[s]) set = &sets[i];
        new_sets[s] = pl->sets[s];
        if (!set) continue;
        VkDescriptorSetLayoutBinding *b = malloc((set->n ? set->n : 1) * sizeof *b);
        int unused = 0;
        for (uint32_t i = 0; i < set->n; i++) {
            b[i] = set->bindings[i];
            int used = 0;
            for (uint32_t k = 0; k < n_stage; k++) used |= module_uses(stage[k], s, b[i].binding);
            if (!used && (b[i].stageFlags & VK_SHADER_STAGE_ALL_GRAPHICS)) {
                b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
                unused++;
            }
        }
        if (unused) {
            VkDescriptorSetLayoutCreateInfo ci = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, set->flags, set->n, b };
            if (((PFN_vkCreateDescriptorSetLayout)next_gdpa(device, "vkCreateDescriptorSetLayout"))(device, &ci, NULL, &new_sets[s]) == VK_SUCCESS)
                changed++;
            else
                new_sets[s] = pl->sets[s];
        }
        free(b);
    }
    if (changed) {
        VkPipelineLayoutCreateInfo ci = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, pl->flags,
                                          pl->n_sets, new_sets, pl->n_push, pl->push };
        if (((PFN_vkCreatePipelineLayout)next_gdpa(device, "vkCreatePipelineLayout"))(device, &ci, NULL, &result) != VK_SUCCESS)
            result = VK_NULL_HANDLE;
    }
out:
    pthread_mutex_unlock(&layouts_lock);
    return result;
}

static VKAPI_ATTR VkResult VKAPI_CALL
layer_CreateGraphicsPipelines(VkDevice device, VkPipelineCache cache, uint32_t count,
                              const VkGraphicsPipelineCreateInfo *infos, const VkAllocationCallbacks *alloc,
                              VkPipeline *pipelines) {
    PFN_vkCreateGraphicsPipelines next = (PFN_vkCreateGraphicsPipelines)next_gdpa(device, "vkCreateGraphicsPipelines");
    VkResult r = next(device, cache, count, infos, alloc, pipelines);
    if (r != VK_ERROR_INVALID_SHADER_NV || !enabled("VIV_NO_LAYOUT_TRIM")) return r;
    r = VK_SUCCESS;
    for (uint32_t i = 0; i < count; i++) {
        if (pipelines[i] != VK_NULL_HANDLE) continue;
        VkGraphicsPipelineCreateInfo patched = infos[i];
        patched.layout = trimmed_layout(device, &infos[i]);
        VkResult one = patched.layout ? next(device, cache, 1, &patched, alloc, &pipelines[i]) : VK_ERROR_INVALID_SHADER_NV;
        if (debug())
            fprintf(stderr, "viv_spv_inline: pipeline retry with trimmed layout -> %d\n", one);
        if (one != VK_SUCCESS) r = one;
    }
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
    if (!strcmp(name, "vkDestroyShaderModule")) return (PFN_vkVoidFunction)layer_DestroyShaderModule;
    if (!strcmp(name, "vkCreateDescriptorSetLayout")) return (PFN_vkVoidFunction)layer_CreateDescriptorSetLayout;
    if (!strcmp(name, "vkDestroyDescriptorSetLayout")) return (PFN_vkVoidFunction)layer_DestroyDescriptorSetLayout;
    if (!strcmp(name, "vkCreatePipelineLayout")) return (PFN_vkVoidFunction)layer_CreatePipelineLayout;
    if (!strcmp(name, "vkDestroyPipelineLayout")) return (PFN_vkVoidFunction)layer_DestroyPipelineLayout;
    if (!strcmp(name, "vkCreateGraphicsPipelines")) return (PFN_vkVoidFunction)layer_CreateGraphicsPipelines;
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
