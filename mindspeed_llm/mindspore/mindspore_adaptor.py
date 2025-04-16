from mindspeed_llm.tasks.megatron_adaptor import MegatronAdaptation, MegatronAdaptationABC


class MindSporeAdaptation(MegatronAdaptationABC):
    """
    Adaptations for models in Megatron-LM Core structure.
    """
    def execute(self):
        args = MegatronAdaptation.get_args()
        if not hasattr(args, "ai_framework") or args.ai_framework != "mindspore":
            return
        from ..core.models.gpt.gpt_model import GPTModel
        from ..mindspore.core.transformer.moe.moe_layer import moe_layer_init_wrapper, moe_layer_forward
        from mindspeed.mindspore.core.data_parallel.distributed_data_parallel import distributed_data_parallel_init_with_cp
        from mindspeed.mindspore.core.transformer.moe.experts import groupedmlp_init_wrapper, groupedmlp_forward

        MegatronAdaptation.register('megatron.core.models.gpt.gpt_model.GPTModel', GPTModel)
        MegatronAdaptation.register('megatron.core.distributed.distributed_data_parallel.DistributedDataParallel.__init__',
                    distributed_data_parallel_init_with_cp, force_patch=True)
        MegatronAdaptation.register('megatron.core.transformer.moe.moe_layer.MoELayer.__init__',
                                moe_layer_init_wrapper)
        MegatronAdaptation.register('megatron.core.transformer.moe.experts.GroupedMLP.__init__',
                                groupedmlp_init_wrapper)
        MegatronAdaptation.register('megatron.core.transformer.moe.moe_layer.MoELayer.forward', moe_layer_forward, force_patch=True)

        if args.moe_permutation_async_comm:
            if args.moe_token_dispatcher_type == 'alltoall':
                if args.moe_alltoall_overlap_comm:
                    from mindspeed.mindspore.core.transformer.moe.legacy_a2a_token_dispatcher import alltoall_token_permutation_new, \
                            alltoall_token_unpermutation_new
                    from mindspeed.mindspore.core.transformer.moe.experts import group_mlp_forward
                    MegatronAdaptation.register('megatron.core.transformer.moe.experts.GroupedMLP.forward', group_mlp_forward, force_patch=True)
                    MegatronAdaptation.register(
                        'megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_permutation',
                        alltoall_token_permutation_new, force_patch=True)
                    MegatronAdaptation.register(
                        'megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_unpermutation',
                        alltoall_token_unpermutation_new, force_patch=True)
                
                if hasattr(args, 'use_fused_moe_token_permute_and_unpermute') and args.use_fused_moe_token_permute_and_unpermute and not args.moe_expert_capacity_factor:
                    from mindspeed.mindspore.core.fusions.npu_moe_token_permute import permute_wrapper
                    from mindspeed.mindspore.core.fusions.npu_moe_token_unpermute import unpermute_wrapper
                    MegatronAdaptation.register('megatron.core.transformer.moe.moe_utils.permute', permute_wrapper)
                    MegatronAdaptation.register('megatron.core.transformer.moe.moe_utils.unpermute', unpermute_wrapper)

        if not args.moe_alltoall_overlap_comm:
            MegatronAdaptation.register('megatron.core.transformer.moe.experts.GroupedMLP.forward',
                                    groupedmlp_forward, force_patch=True)

        from mindspeed.mindspore.core.distributed.distributed_data_parallel import distributed_data_parallel_init, local_make_param_hook
        MegatronAdaptation.register('megatron.core.distributed.distributed_data_parallel.DistributedDataParallel.__init__', distributed_data_parallel_init, force_patch=True)
        MegatronAdaptation.register('megatron.core.distributed.distributed_data_parallel.DistributedDataParallel._make_param_hook', local_make_param_hook)

        from mindspeed.mindspore.core.distributed.param_and_grad_buffer import register_grad_ready 
        MegatronAdaptation.register('megatron.core.distributed.param_and_grad_buffer.register_grad_ready', register_grad_ready)

        from mindspeed.mindspore.core.models.common.embeddings.rotary_pos_embedding import get_rotary_seq_len, local_rotate_half
        MegatronAdaptation.register('megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding.get_rotary_seq_len', get_rotary_seq_len)
        MegatronAdaptation.register('megatron.core.models.common.embeddings._rotate_half', local_rotate_half)

        from mindspeed.mindspore.core.optimizer import get_megatron_optimizer
        MegatronAdaptation.register('megatron.core.optimizer.get_megatron_optimizer', get_megatron_optimizer)
        from mindspeed.mindspore.core.optimizer.optimizer import megatron_optimizer_init
        MegatronAdaptation.register('megatron.core.optimizer.optimizer.MegatronOptimizer.__init__', megatron_optimizer_init)

        from mindspeed.mindspore.core.pipeline_parallel.schedules import forward_step, backward_step, forward_backward_no_pipelining
        MegatronAdaptation.register('megatron.core.pipeline_parallel.schedules.forward_step', forward_step)
        MegatronAdaptation.register('megatron.core.pipeline_parallel.schedules.backward_step', backward_step)
        MegatronAdaptation.register('megatron.core.pipeline_parallel.schedules.forward_backward_no_pipelining', forward_backward_no_pipelining)
        from mindspeed.mindspore.core.pipeline_parallel.schedules import forward_backward_pipelining_with_interleaving, forward_backward_pipelining_without_interleaving
        MegatronAdaptation.register('megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_with_interleaving', forward_backward_pipelining_with_interleaving)
        MegatronAdaptation.register('megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_without_interleaving', forward_backward_pipelining_without_interleaving)

        from mindspeed.mindspore.core.tensor_parallel.data import local_build_key_size_numel_dictionaries
        MegatronAdaptation.register('megatron.core.tensor_parallel.data._build_key_size_numel_dictionaries', local_build_key_size_numel_dictionaries) # 1097

        from mindspeed.mindspore.core.tensor_parallel.mappings import all_to_all_forward 
        MegatronAdaptation.register('megatron.core.tensor_parallel.mappings._AllToAll.forward', all_to_all_forward)

        from mindspeed.mindspore.core.tensor_parallel.random import local_set_cuda_rng_state, checkpoint_function_forward, checkpoint_function_backward
        MegatronAdaptation.register('megatron.core.tensor_parallel.random._set_cuda_rng_state', local_set_cuda_rng_state, force_patch=True)
        MegatronAdaptation.register('megatron.core.tensor_parallel.random.CheckpointFunction.forward', checkpoint_function_forward)
        MegatronAdaptation.register('megatron.core.tensor_parallel.random.CheckpointFunction.backward', checkpoint_function_backward)
