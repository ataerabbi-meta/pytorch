# Owner(s): ["module: unknown"]

import difflib

import yaml

from torch._inductor import config as inductor_config
from torch.testing._internal.common_utils import run_tests, TestCase


class TestConfigModule(TestCase):
    def test_inductor_config_hash_portable_fixture(self):
        torch_config = inductor_config.save_config_portable()
        torch_config_yaml = yaml.dump(
            torch_config,
            sort_keys=True,
        )
        self.assertExpectedInline(
            torch_config_yaml,
            """\
TYPE_CHECKING: false
add_pre_grad_passes: null
aggressive_fusion: false
alignment_asserts: true
allow_buffer_reuse: true
always_keep_tensor_constants: false
annotate_training: false
aot_inductor.allow_stack_allocation: false
aot_inductor.aoti_shim_library: null
aot_inductor.aoti_shim_library_path: null
aot_inductor.check_lowerbound: true
aot_inductor.compile_wrapper_opt_level: O1
aot_inductor.cross_target_platform: null
aot_inductor.custom_op_libs: null
aot_inductor.custom_ops_to_c_shims: {}
aot_inductor.debug_compile: false
aot_inductor.debug_intermediate_value_printer: '0'
aot_inductor.debug_symbols: false
aot_inductor.dump_aoti_minifier: false
aot_inductor.dynamic_linkage: true
aot_inductor.embed_kernel_binary: null
aot_inductor.emit_multi_arch_kernel: null
aot_inductor.enable_lto: false
aot_inductor.filtered_kernel_names: null
aot_inductor.force_mmap_weights: false
aot_inductor.link_libtorch: true
aot_inductor.metadata: {}
aot_inductor.model_name_for_generated_files: null
aot_inductor.output_path: ''
aot_inductor.package: false
aot_inductor.package_constants_in_so: true
aot_inductor.package_constants_on_disk_format: null
aot_inductor.package_cpp_only: null
aot_inductor.precompile_headers: true
aot_inductor.presets: {}
aot_inductor.raise_error_on_ignored_optimization: true
aot_inductor.repro_level: 2
aot_inductor.serialized_in_spec: ''
aot_inductor.serialized_out_spec: ''
aot_inductor.use_consts_asm_build: true
aot_inductor.use_minimal_arrayref_interface: false
aot_inductor.use_runtime_constant_folding: false
aot_inductor.weight_use_caching_allocator: false
aot_inductor_mode.compile_standalone: false
apply_gumbel_max_trick: true
assert_indirect_indexing: true
assume_32bit_indexing: false
assume_aligned_inputs: false
assume_unaligned_fallback_output: false
aten_distributed_optimizations.bucket_exposed_first: true
aten_distributed_optimizations.collective_bucketing: null
aten_distributed_optimizations.collective_estimator: analytical
aten_distributed_optimizations.compute_overlap_multipler: null
aten_distributed_optimizations.custom_runtime_estimation: null
aten_distributed_optimizations.enable_fusion_regions: null
aten_distributed_optimizations.enable_overlap_scheduling: false
aten_distributed_optimizations.insert_overlap_deps: null
aten_distributed_optimizations.log_final_collectives_estimations: false
aten_distributed_optimizations.max_coll_distance: null
aten_distributed_optimizations.max_compute_pre_fetch: null
aten_distributed_optimizations.max_in_flight_gb: null
aten_distributed_optimizations.max_memory_increase_gb: null
aten_distributed_optimizations.max_memory_increase_ratio: null
autoheuristic_collect: ''
autoheuristic_log_path: DEFAULT
autoheuristic_use: mixed_mm
autotune_fallback_to_aten: false
autotune_in_subproc: false
autotune_lookup_table: {}
autotune_multi_device: false
autotune_num_choices_displayed: 10
b2b_gemm_pass: false
batch_fusion: true
benchmark_combo_kernel: false
benchmark_epilogue_fusion: true
benchmark_fusion: false
benchmark_harness: true
benchmark_kernel: false
bucket_all_gathers_fx: none
bucket_all_gathers_fx_bucket_size_determinator: null
bucket_all_reduces_fx: none
bucket_all_reduces_fx_bucket_size_determinator: null
bucket_reduce_scatters_fx: none
bucket_reduce_scatters_fx_bucket_size_determinator: null
bundle_triton_into_fx_graph_cache: true
bundled_autotune_remote_cache: null
bw_outputs_user_visible: true
can_inplace_pad_graph_input: false
check_stack_no_cycles_TESTING_ONLY: false
collective_benchmark_nruns: 50
collective_benchmark_timeout: 30.0
combo_kernel_allow_mixed_sizes: 1
combo_kernel_foreach_dynamic_shapes: true
combo_kernel_max_num_args: 250
combo_kernels: false
combo_kernels_autotune: 1
comment_origin: false
comprehensive_padding: true
compute_all_bounds: false
constant_and_index_propagation: true
conv_1x1_as_mm: false
coordinate_descent_check_all_directions: false
coordinate_descent_search_radius: 1
coordinate_descent_tuning: false
cpp.cxx: !!python/tuple
- null
- g++
cpp.descriptive_names: original_aten
cpp.dynamic_threads: false
cpp.enable_concat_linear: false
cpp.enable_floating_point_contract_flag: 'off'
cpp.enable_grouped_gemm_template: false
cpp.enable_kernel_profile: false
cpp.enable_loop_tail_vec: true
cpp.enable_tiling_heuristics: true
cpp.enable_unsafe_math_opt_flag: false
cpp.fallback_scatter_reduce_sum: true
cpp.force_inline_kernel: false
cpp.gemm_cache_blocking: null
cpp.gemm_max_k_slices: 1
cpp.gemm_thread_factors: null
cpp.inject_log1p_bug_TESTING_ONLY: null
cpp.inject_relu_bug_TESTING_ONLY: null
cpp.max_horizontal_fusion_size: 16
cpp.min_chunk_size: 512
cpp.no_redundant_loops: true
cpp.simdlen: null
cpp.threads: -1
cpp.use_constexpr_for_int_array: true
cpp.use_decompose_tanh: false
cpp.use_small_dequant_buffer: false
cpp.vec_isa_ok: null
cpp.weight_prepack: true
cpp_cache_precompile_headers: true
cpp_wrapper: false
cpp_wrapper_build_separate: false
cpu_backend: cpp
cpu_gpu_bw: 50.0
cuda.arch: null
cuda.binary_remote_cache_force_write: false
cuda.compile_opt_level: -O1
cuda.cuda_cxx: null
cuda.cutlass_backend_min_gemm_size: 1
cuda.cutlass_enabled_ops: all
cuda.cutlass_epilogue_fusion_enabled: false
cuda.cutlass_hash_with_compile_cmd: false
cuda.cutlass_instantiation_level: '0'
cuda.cutlass_max_profiling_configs: null
cuda.cutlass_max_profiling_swizzle_options:
- 1
- 2
- 4
- 8
cuda.cutlass_op_allowlist_regex: null
cuda.cutlass_op_denylist_regex: null
cuda.cutlass_prescreening: true
cuda.cutlass_tma_only: false
cuda.enable_caching_codegen: true
cuda.enable_cuda_lto: false
cuda.enable_debug_info: false
cuda.enable_ptxas_info: false
cuda.generate_test_runner: false
cuda.nvgemm_max_profiling_configs: 5
cuda.upload_to_binary_remote_cache: false
cuda.use_binary_remote_cache: true
cuda.use_fast_math: false
cuda.version: null
cuda_backend: triton
custom_partitioner_fn: null
custom_should_partition_ops: []
cutedsl_enable_autotuning: false
dce: false
debug: false
debug_fusion: false
debug_index_asserts: false
debug_ir_traceback: false
decompose_mem_bound_mm: false
deterministic: false
developer_warnings: true
disable_cpp_codegen: false
disable_padding_cpu: true
disable_progress: true
distributed_max_autotune_gemm: false
dynamic_scale_rblock: true
efficient_conv_bn_eval_fx_passes: false
emulate_divison_rounding: false
emulate_precision_casts: false
enable_auto_functionalized_v2: true
enable_autograd_for_aot: false
enable_caching_generated_triton_templates: true
enable_linear_binary_folding: false
enabled_metric_tables: ''
epilogue_fusion: true
epilogue_fusion_first: false
estimate_op_runtime: default
expand_dimension_for_pointwise_nodes: false
external_matmul: []
fallback_by_default: false
fallback_embedding_bag_byte_unpack: false
fallback_random: false
file_lock_timeout: 600
force_fuse_int_mm_with_mul: false
force_layout_optimization: false
force_pointwise_cat: false
force_same_precision: false
force_shape_pad: false
freezing: false
freezing_discard_parameters: false
fx_passes_numeric_check:
  num_iterations: 1
  pre_grad: false
  precision: 0.0001
  requires_optimizer: true
fx_wrapper: false
generate_intermediate_hooks: false
global_cache_dir: null
graph_partition: true
group_fusion: false
halide.asserts: false
halide.cpu_target: host
halide.debug: false
halide.gpu_target: host-cuda
halide.scan_kernels: false
halide.scheduler_cpu: Adams2019
halide.scheduler_cuda: Anderson2021
implicit_fallbacks: true
inductor_choices_class: null
inductor_default_autotune_rep: 100
inductor_default_autotune_warmup: 25
inplace_buffers: true
inplace_padding: true
inter_node_bw: 25
intra_node_bw: 300
is_nightly_or_source: true
is_predispatch: false
joint_graph_constant_folding: true
keep_output_stride: true
kernel_name_max_ops: 10
layout_opt_default: '1'
layout_optimization: true
log_tlparse: false
lookup_table.check_src_hash: true
lookup_table.table: null
loop_index_inversion_in_fusion: true
loop_ordering_after_fusion: true
max_autotune: false
max_autotune_allow_flexible_layouts: false
max_autotune_conv_backends: ATEN,TRITON
max_autotune_flex_search_space: DEFAULT
max_autotune_gemm: false
max_autotune_gemm_backends: ATEN,TRITON,CPP
max_autotune_gemm_search_space: DEFAULT
max_autotune_pointwise: false
max_autotune_prune_choices_based_on_shared_mem: true
max_autotune_report_choices_stats: true
max_autotune_subproc_graceful_timeout_seconds: 0.0
max_autotune_subproc_result_timeout_seconds: 60.0
max_autotune_subproc_terminate_timeout_seconds: 0.0
max_epilogue_benchmarked_choices: 1
max_fusion_buffer_group_pairwise_attempts: 64
max_fusion_size: 64
max_fusion_unique_io_buffers: null
max_pointwise_cat_inputs: 8
memory_planning: false
memory_pool: intermediates
min_num_split: 0
mixed_mm_choice: heuristic
multi_kernel_hints: []
nan_asserts: false
non_blocking_remote_cache_write: true
online_softmax: true
optimize_scatter_upon_const_tensor: true
pad_channels_last: false
pad_dynamic_shapes: false
pad_outputs: false
padding_alignment_bytes: 128
padding_stride_threshold: 1024
pallas_take_first_jax_device_only: true
pattern_matcher: true
permute_fusion: false
pick_loop_orders: true
pipeline_max_autotune_gemm: false
post_grad_fusion_options: {}
pre_grad_custom_pass: null
pre_grad_fusion_options: {}
precompilation_timeout_seconds: 3600
profile_bandwidth: false
profile_bandwidth_output: null
profile_bandwidth_regex: ''
profile_bandwidth_with_do_bench_using_profiling: false
profiler_mark_wrapper_call: false
prologue_fusion: true
quiesce_async_compile_pool: true
quiesce_async_compile_time: 60
realize_acc_reads_size_threshold: null
realize_acc_reads_threshold: 8
realize_opcount_threshold: 30
realize_reads_threshold: 4
remote_gemm_autotune_cache: false
remove_pre_grad_passes: null
reorder_for_compute_comm_overlap: false
reorder_for_compute_comm_overlap_passes: []
reorder_for_locality: true
reorder_for_peak_memory: true
reorder_for_peak_memory_debug: false
reorder_prefetch_limit: null
rocm.arch: []
rocm.ck_dir: null
rocm.ck_max_profiling_configs: null
rocm.ck_supported_arch:
- gfx90a
- gfx942
- gfx950
rocm.ck_tile_max_profiling_configs: null
rocm.compile_opt_level: -O2
rocm.contiguous_threshold: 16
rocm.flush_denormals: true
rocm.generate_test_runner: false
rocm.is_debug: false
rocm.kBatch_sweep: null
rocm.n_max_profiling_configs: null
rocm.print_kernel_resource_usage: false
rocm.rocm_home: null
rocm.save_temps: false
rocm.split_k_threshold: 16
rocm.use_fast_math: true
rocm.use_preselected_instances: false
run_jit_post_compile_hook: false
runtime_estimations_mms_benchmark: false
runtime_triton_nan_asserts: false
save_args: false
scalar_asserts: true
score_fusion_memory_threshold: 10
search_autotune_cache: false
selective_decompose: false
shape_padding: true
size_asserts: true
size_threshold_for_succ_based_strategy: 0
sleep_sec_TESTING_ONLY: null
small_memory_access_threshold: 16777216
split_cat_fx_passes: true
split_reductions: true
static_launch_user_defined_triton_kernels: false
static_weight_shapes: true
strict_static_cuda_launcher: false
test_configs.assume_bucketing_reduces_latency: true
test_configs.autotune_choice_desc_regex: null
test_configs.autotune_choice_name_regex: null
test_configs.bisect_keep_custom_backend_for_inductor: false
test_configs.bisect_pre_grad_graph: false
test_configs.distort_benchmarking_result: ''
test_configs.force_extern_kernel_in_multi_template: false
test_configs.force_filter_reduction_configs: false
test_configs.graphsafe_rng_func_ignores_fallback_random: false
test_configs.max_mm_configs: null
test_configs.runtime_triton_dtype_assert: false
test_configs.runtime_triton_shape_assert: false
test_configs.static_cpp_dtype_assert: false
test_configs.track_memory_lifecycle: null
test_configs.use_libtorch: false
torchinductor_worker_logpath: ''
triton.autotune_at_compile_time: null
triton.autotune_cublasLt: true
triton.autotune_pointwise: true
triton.autotune_with_sample_inputs: false
triton.coalesce_tiling_analysis: true
triton.codegen_upcast_to_fp32: true
triton.cooperative_reductions: false
triton.cudagraph_capture_sizes: null
triton.cudagraph_dynamic_shape_warn_limit: 8
triton.cudagraph_or_error: false
triton.cudagraph_skip_dynamic_graphs: false
triton.cudagraph_support_input_mutation: true
triton.cudagraph_trees: true
triton.cudagraph_trees_history_recording: false
triton.cudagraph_trees_objgraph: false
triton.cudagraph_unexpected_rerecord_limit: 128
triton.cudagraphs: false
triton.debug_sync_graph: false
triton.debug_sync_kernel: false
triton.decompose_k_threshold: 32
triton.dense_indexing: false
triton.descriptive_names: original_aten
triton.disallow_failing_autotune_kernels_TESTING_ONLY: false
triton.divisible_by_16: true
triton.enable_epilogue_subtiling: true
triton.enable_pdl: false
triton.enable_persistent_tma_matmul: false
triton.enable_template_tma_store: false
triton.enable_tlx_templates: false
triton.fast_path_cudagraph_asserts: false
triton.force_cooperative_reductions: false
triton.force_cudagraph_sync: false
triton.force_cudagraphs_warmup: false
triton.inject_relu_bug_TESTING_ONLY: null
triton.max_tiles: null
triton.min_split_scan_rblock: 256
triton.mix_order_reduction: true
triton.mix_order_reduction_autotune_split_size: false
triton.mix_order_reduction_initial_xblock: 1
triton.mix_order_reduction_split_size: null
triton.multi_kernel: 0
triton.native_matmul: false
triton.num_decompose_k_splits: 10
triton.persistent_reductions: true
triton.prefer_nd_tiling: false
triton.reorder_for_reducing_graph_partitions: true
triton.skip_cudagraph_warmup: false
triton.skip_l1_cache: false
triton.slow_path_cudagraph_asserts: true
triton.spill_threshold: 16
triton.store_cubin: false
triton.tile_reductions: false
triton.tiling_prevents_pointwise_fusion: true
triton.tiling_prevents_reduction_fusion: true
triton.transpose_discontiguous_tensor_descriptor: true
triton.unique_kernel_names: true
triton.unique_user_kernel_names: false
triton.use_block_ptr: false
triton.use_tensor_descriptor: false
triton_disable_device_detection: false
triton_kernel_default_layout_constraint: needs_fixed_stride_order
unbacked_symint_fallback: 8192
unroll_reductions_threshold: 8
unsafe_ignore_unsupported_triton_autotune_args: false
unsafe_marked_cacheable_functions: {}
unsafe_skip_cache_dynamic_shape_guards: false
use_dce: true
use_experimental_benchmarker: true
use_fast_math: false
use_joint_graph_passes: true
use_mixed_mm: true
use_post_grad_passes: true
use_pre_grad_passes: true
use_static_cuda_launcher: true
verbose_progress: false
warn_mix_layout: false
worker_log_path: null
worker_suppress_logging: true
wrap_inductor_compiled_regions: false
write_are_deterministic_algorithms_enabled: true
xpu_backend: triton
""",
        )

    def test_inductor_config_hash_portable_without_ignore(self):
        """
        Detect the inductor config hash will change if we forgot to ignore cuda.cutlass_dir.
        """
        expected_torch_config = inductor_config.save_config_portable()
        expected_torch_config_yaml = yaml.dump(
            expected_torch_config,
            sort_keys=True,
        )

        idx = inductor_config._cache_config_ignore_prefix.index("cuda.cutlass_dir")
        inductor_config._cache_config_ignore_prefix.remove("cuda.cutlass_dir")
        try:
            changed_torch_config = inductor_config.save_config_portable()
            changed_torch_config_yaml = yaml.dump(
                changed_torch_config,
                sort_keys=True,
            )
            self.assertNotEqual(changed_torch_config_yaml, expected_torch_config_yaml)
            diff = difflib.ndiff(
                expected_torch_config_yaml.splitlines(keepends=True),
                changed_torch_config_yaml.splitlines(keepends=True),
            )
            diff_lines = [line for line in diff if line.startswith(("+ ", "- "))]
            self.assertEqual(len(diff_lines), 1)
            self.assertTrue(diff_lines[0].startswith("+ cuda.cutlass_dir: "))
        finally:
            inductor_config._cache_config_ignore_prefix.insert(idx, "cuda.cutlass_dir")


if __name__ == "__main__":
    run_tests()
