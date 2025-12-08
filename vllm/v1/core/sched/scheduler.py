# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import itertools
import time
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Optional, Union

from vllm.config import VllmConfig
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.factory import (
    KVConnectorFactory)
from vllm.distributed.kv_transfer.kv_connector.v1 import (KVConnectorBase_V1,
                                                          KVConnectorRole)
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.encoder_cache_manager import (EncoderCacheManager,
                                                compute_encoder_budget)
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import (CachedRequestData, NewRequestData,
                                       SchedulerOutput)
from vllm.v1.core.sched.request_queue import (SchedulingPolicy,
                                              create_request_queue)
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.engine import (EngineCoreEventType, EngineCoreOutput,
                            EngineCoreOutputs)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager
from copy import deepcopy
logger = init_logger(__name__)
from vllm.v1.core.block_pool import KVCacheBlock

from collections import Counter
class Scheduler(SchedulerInterface):

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.kv_cache_config = kv_cache_config
        self.kv_events_config = vllm_config.kv_events_config
        self.parallel_config = vllm_config.parallel_config
        self.log_stats = log_stats
        self.structured_output_manager = structured_output_manager

        # include_finished_set controls whether a separate set of finished
        # request ids should be included in the EngineCoreOutputs returned
        # by update_from_outputs(). This is currently used in the multi-engine
        # case to track request lifetimes efficiently.
        self.finished_req_ids_dict: Optional[dict[int, set[str]]] = (
            defaultdict(set) if include_finished_set else None)

        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = \
            self.scheduler_config.max_num_batched_tokens
        self.max_model_len = self.scheduler_config.max_model_len
        self.enable_kv_cache_events = (
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events)
        
        # req_id -> Request
        self.requests: dict[str, Request] = {}        
        
        # Create KVConnector for the Scheduler. Note that each Worker
        # will have a corresponding KVConnector with Role=WORKER.
        # KV Connector pushes/pull of remote KVs for P/D and offloading.
        self.connector = None
        if self.vllm_config.kv_transfer_config is not None:
            assert len(self.kv_cache_config.kv_cache_groups) >= 1, ( #sefi - changed from == 1
                "Multiple KV cache groups are not currently supported "
                "with KV connectors")
            self.connector = KVConnectorFactory.create_connector_v1(
                config=self.vllm_config, role=KVConnectorRole.SCHEDULER)
            # self.connector.set_request_dict(self.requests) #sefi 
           

        self.kv_event_publisher = EventPublisherFactory.create(
            self.kv_events_config,
            self.parallel_config.data_parallel_rank,
        )

        num_gpu_blocks = self.cache_config.num_gpu_blocks
        assert num_gpu_blocks is not None and num_gpu_blocks > 0

        self.block_size = self.cache_config.block_size

        

        # Scheduling policy
        if self.scheduler_config.policy == "priority":
            self.policy = SchedulingPolicy.PRIORITY
        elif self.scheduler_config.policy == "fcfs":
            self.policy = SchedulingPolicy.FCFS
        else:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}")
        # Priority queues for requests.
        self.waiting = create_request_queue(self.policy)
        self.running: list[Request] = []

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: set[str] = set()

        # KV Connector: requests in process of async KV loading or recving
        self.finished_recving_kv_req_ids: set[str] = set()

        # Encoder-related.
        # Calculate encoder cache size if applicable
        # NOTE: For now we use the same budget for both compute and space.
        # This can be changed when we make encoder cache for embedding caching
        # across requests.
        encoder_compute_budget, encoder_cache_size = compute_encoder_budget(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            mm_registry=mm_registry,
        )

        # NOTE(woosuk): Here, "encoder" includes the vision encoder (and
        # projector if needed). Currently, we assume that the encoder also
        # has the Transformer architecture (e.g., ViT).
        self.max_num_encoder_input_tokens = encoder_compute_budget
        # NOTE: For the models without encoder (e.g., text-only models),
        # the encoder cache will not be initialized because cache size is 0
        # for these models.
        self.encoder_cache_manager = EncoderCacheManager(
            cache_size=encoder_cache_size)

        speculative_config = vllm_config.speculative_config

        self.use_eagle = False
        self.num_spec_tokens = self.num_lookahead_tokens = 0
        if speculative_config:
            self.num_spec_tokens = speculative_config.num_speculative_tokens
            if speculative_config.use_eagle():
                self.use_eagle = True
                self.num_lookahead_tokens = self.num_spec_tokens

        # Create the KV cache manager.
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            enable_caching=self.cache_config.enable_prefix_caching,
            caching_hash_algo=self.cache_config.prefix_caching_hash_algo,
            use_eagle=self.use_eagle,
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,
        )
        self.use_pp = self.parallel_config.pipeline_parallel_size > 1

    def schedule(self) -> SchedulerOutput:
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        # NOTE: structured_output_request_ids maps
        # a request's (request that uses structured output)
        # request_id to the running request index.
        # This will helps us determine to slice the grammar bitmask
        # and only applies valid mask for requests that
        # uses structured decoding.
        structured_output_request_ids: dict[str, int] = {}

        req_to_new_block_ids: dict[str, tuple[list[int], ...]] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # For logging.
        scheduled_timestamp = time.monotonic()

        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            num_new_tokens = (request.num_tokens_with_spec +
                              request.num_output_placeholders -
                              request.num_computed_tokens)
            if (0 < self.scheduler_config.long_prefill_token_threshold <
                    num_new_tokens):
                num_new_tokens = (
                    self.scheduler_config.long_prefill_token_threshold)
            num_new_tokens = min(num_new_tokens, token_budget)

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len - 1 - request.num_computed_tokens)

            # Schedule encoder inputs.
            encoder_inputs_to_schedule = None
            new_encoder_budget = encoder_budget
            if request.has_encoder_inputs:
                (encoder_inputs_to_schedule, num_new_tokens,
                 new_encoder_budget) = self._try_schedule_encoder_inputs(
                     request, request.num_computed_tokens, num_new_tokens,
                     encoder_budget)

            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when
                #    (1) PP>1 and we have already scheduled all prompt tokens
                #    but they are not finished yet.
                #    (2) Async scheduling and the request has reached to either
                #    its max_total_tokens or max_model_len.
                # 2. The encoder budget is exhausted.
                # 3. The encoder cache is exhausted.
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue

            while True:
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_lookahead_tokens=self.num_lookahead_tokens)
                if new_blocks is None:
                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        self.running.remove(preempted_req)
                    else:
                        preempted_req = self.running.pop()

                    self.kv_cache_manager.free(preempted_req)
                    preempted_req.status = RequestStatus.PREEMPTED
                    preempted_req.num_computed_tokens = 0
                    if self.log_stats:
                        preempted_req.record_event(
                            EngineCoreEventType.PREEMPTED, scheduled_timestamp)

                    self.waiting.prepend_request(preempted_req)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # No more request to preempt.
                        can_schedule = False
                        break
                else:
                    # The request can be scheduled.
                    can_schedule = True
                    break
            if not can_schedule:
                break
            assert new_blocks is not None

            # Schedule the request.
            scheduled_running_reqs.append(request)
            if request.use_structured_output:
                # PERF: in case of chunked prefill,
                # request might not include any new tokens.
                # Therefore, we might introduce some additional
                # cycle to fill in the bitmask, which could be a big no-op.
                structured_output_request_ids[request.request_id] = req_index
            req_to_new_block_ids[request.request_id] = (
                new_blocks.get_block_ids())
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

            # Speculative decode related.
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (num_new_tokens +
                                             request.num_computed_tokens -
                                             request.num_tokens)
                if num_scheduled_spec_tokens > 0:
                    # Trim spec_token_ids list to num_scheduled_spec_tokens.
                    del request.spec_token_ids[num_scheduled_spec_tokens:]
                    scheduled_spec_decode_tokens[request.request_id] = (
                        request.spec_token_ids)

            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request.request_id] = (
                    encoder_inputs_to_schedule)
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                encoder_budget = new_encoder_budget

        # Record the LoRAs in scheduled_running_reqs
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0)
            assert len(scheduled_loras) <= self.lora_config.max_loras

        # Use a temporary RequestQueue to collect requests that need to be
        # skipped and put back at the head of the waiting queue later
        skipped_waiting_requests = create_request_queue(self.policy)

        # Next, schedule the WAITING requests.
        if not preempted_reqs:
            while self.waiting and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break

                request = self.waiting.peek_request()

                # KVTransfer: skip request if still waiting for remote kvs.
                if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                    is_ready = self._update_waiting_for_remote_kv(request)
                    if is_ready:
                        request.status = RequestStatus.WAITING
                    else:
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",
                            request.request_id)
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                # Skip request if the structured output request is still waiting
                # for FSM compilation.
                if request.status == RequestStatus.WAITING_FOR_FSM:
                    structured_output_req = request.structured_output_request
                    if structured_output_req and structured_output_req.grammar:
                        request.status = RequestStatus.WAITING
                    else:
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                # Check that adding the request still respects the max_loras
                # constraint.
                if (self.lora_config and request.lora_request and
                    (len(scheduled_loras) == self.lora_config.max_loras and
                     request.lora_request.lora_int_id not in scheduled_loras)):
                    # Scheduling would exceed max_loras, skip.
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    continue

                num_external_computed_tokens = 0
                load_kv_async = False

                # Get already-cached tokens.
                if request.num_computed_tokens == 0:
                    # Get locally-cached tokens.
                    new_computed_blocks, num_new_local_computed_tokens = \
                        self.kv_cache_manager.get_computed_blocks(
                            request)

                    # Get externally-cached tokens if using a KVConnector.
                    if self.connector is not None:
                        num_external_computed_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request, num_new_local_computed_tokens))

                    # Total computed tokens (local + external).
                    num_computed_tokens = (num_new_local_computed_tokens +
                                           num_external_computed_tokens)
                # KVTransfer: WAITING reqs have num_computed_tokens > 0
                # after async KV recvs are completed.
                else:
                    new_computed_blocks = (
                        self.kv_cache_manager.create_empty_block_list())
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                encoder_inputs_to_schedule = None
                new_encoder_budget = encoder_budget

                # KVTransfer: loading remote KV, do not allocate for new work.
                if load_kv_async:
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                # Number of tokens to be scheduled.
                else:
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    num_new_tokens = request.num_tokens - num_computed_tokens
                    if (0 < self.scheduler_config.long_prefill_token_threshold
                            < num_new_tokens):
                        num_new_tokens = (
                            self.scheduler_config.long_prefill_token_threshold)

                    # chunked prefill has to be enabled explicitly to allow
                    # pooling requests to be chunked
                    if not self.scheduler_config.chunked_prefill_enabled and \
                        num_new_tokens > token_budget:
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                    num_new_tokens = min(num_new_tokens, token_budget)
                    assert num_new_tokens > 0

                    # Schedule encoder inputs.
                    if request.has_encoder_inputs:
                        (encoder_inputs_to_schedule, num_new_tokens,
                         new_encoder_budget
                         ) = self._try_schedule_encoder_inputs(
                             request, num_computed_tokens, num_new_tokens,
                             encoder_budget)
                        if num_new_tokens == 0:
                            # The request cannot be scheduled.
                            break

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens + num_external_computed_tokens,
                    num_new_local_computed_tokens,
                    new_computed_blocks,
                    num_lookahead_tokens=self.num_lookahead_tokens,
                    delay_cache_blocks=load_kv_async,
                )
                if new_blocks is None:
                    # The request cannot be scheduled.
                    break

                # KVTransfer: the connector uses this info to determine
                # if a load is needed. Note that
                # This information is used to determine if a load is
                # needed for this request.
                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        new_computed_blocks + new_blocks,
                        num_external_computed_tokens,
                    )

                # Request was already popped from self.waiting
                # unless it was re-added above due to new_blocks being None.
                request = self.waiting.pop_request()
                if load_kv_async:
                    # If loading async, allocate memory and put request
                    # into the WAITING_FOR_REMOTE_KV state.
                    skipped_waiting_requests.prepend_request(request)
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    continue

                if request.use_structured_output:
                    structured_output_request_ids[request.request_id] = (
                        req_index)
                req_index += 1
                self.running.append(request)
                if self.log_stats:
                    request.record_event(EngineCoreEventType.SCHEDULED,
                                         scheduled_timestamp)
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(
                        f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_block_ids[request.request_id] = (
                    self.kv_cache_manager.get_block_ids(request.request_id))
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                # Count the number of prefix cached tokens.
                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = num_computed_tokens
                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request.request_id] = (
                        encoder_inputs_to_schedule)
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                    encoder_budget = new_encoder_budget

        # Put back any skipped requests at the head of the waiting queue
        if skipped_waiting_requests:
            self.waiting.prepend_requests(skipped_waiting_requests)

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert (len(scheduled_new_reqs) + len(scheduled_resumed_reqs) +
                len(scheduled_running_reqs) <= len(self.running))

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = [0] * len(
            self.kv_cache_config.kv_cache_groups)
        if self.running:
            any_request = self.running[0]
            num_common_prefix_blocks = (
                self.kv_cache_manager.get_num_common_prefix_blocks(
                    any_request, len(self.running)))

        grammar_bitmask = self.structured_output_manager.grammar_bitmask(
            self.requests,
            structured_output_request_ids,
            scheduled_spec_decode_tokens,
        )
        
        # Construct the scheduler output.
        new_reqs_data = [
            NewRequestData.from_request(req,
                                        req_to_new_block_ids[req.request_id])
            for req in scheduled_new_reqs
        ]
        cached_reqs_data = self._make_cached_request_data(
            scheduled_running_reqs,
            scheduled_resumed_reqs,
            num_scheduled_tokens,
            scheduled_spec_decode_tokens,
            req_to_new_block_ids,
        )
        
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_input_ids=self.encoder_cache_manager.get_freed_ids(),
            structured_output_request_ids=structured_output_request_ids,
            grammar_bitmask=grammar_bitmask,
        )

        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta = self.connector.build_connector_meta(scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        events = self.kv_cache_manager.take_events()
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        self._update_after_schedule(scheduler_output)
        # self._validate_queue_consistency("line 591 in schedule()")
        return scheduler_output

    def _update_after_schedule(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token

            # NOTE: _free_encoder_inputs relies on num_computed_tokens, which
            # may be updated again in _update_from_output for speculative
            # decoding. However, it is safe to call the method here because
            # encoder inputs are always part of the prompt, not the output,
            # and thus are unaffected by speculative decoding.
            if request.has_encoder_inputs:
                self._free_encoder_inputs(request)

        # Clear the finished request IDs.
        # NOTE: We shouldn't do self.finished_req_ids.clear() here because
        # it will also affect the scheduler output.
        self.finished_req_ids = set()

    def _make_cached_request_data(
        self,
        running_reqs: list[Request],
        resumed_reqs: list[Request],
        num_scheduled_tokens: dict[str, int],
        spec_decode_tokens: dict[str, list[int]],
        req_to_new_block_ids: dict[str, tuple[list[int], ...]],
    ) -> CachedRequestData:
        req_ids: list[str] = []
        new_token_ids: list[list[int]] = []
        new_block_ids: list[tuple[list[int], ...]] = []
        num_computed_tokens: list[int] = []

        use_connector = self.connector is not None
        for req in itertools.chain(running_reqs, resumed_reqs):
            req_id = req.request_id
            req_ids.append(req_id)
            num_tokens = (num_scheduled_tokens[req_id] -
                          len(spec_decode_tokens.get(req_id, ())))
            if self.use_pp:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker. Otherwise, we don't
                # need to send the sampled tokens back because the model runner
                # will cache them.
                token_ids = req.all_token_ids[req.num_computed_tokens:req.
                                              num_computed_tokens + num_tokens]
                new_token_ids.append(token_ids)
            elif use_connector:
                # When using a KVConnector, we add a placeholder to avoid index
                # out of bounds errors. TODO: Remove this once the KVConnector
                # is updated to handle token IDs properly.
                new_token_ids.append([])
            new_block_ids.append(req_to_new_block_ids[req_id])
            num_computed_tokens.append(req.num_computed_tokens)
        # Because resumed_reqs is usually empty, it is more efficient to do
        # in-place appending so that we don't need to allocate a new list.
        resumed_from_preemption = [False] * len(running_reqs)
        resumed_from_preemption += [True] * len(resumed_reqs)

        return CachedRequestData(
            req_ids=req_ids,
            resumed_from_preemption=resumed_from_preemption,
            new_token_ids=new_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
        )

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_budget: int,
    ) -> tuple[list[int], int, int]:
        """
        Determine which encoder inputs need to be scheduled in the current step,
        and update `num_new_tokens` and encoder token budget accordingly.

        An encoder input will be scheduled if:
        - Its output tokens overlap with the range of tokens being computed
        in this step, i.e.,
        [num_computed_tokens, num_computed_tokens + num_new_tokens).
        - It is not already computed and stored in the encoder cache.
        - There is sufficient encoder token budget to process it.
        - The encoder cache has space to store it.

        If an encoder input cannot be scheduled due to cache or budget
        limitations, the method adjusts `num_new_tokens` to schedule only the
        decoder tokens up to just before the unschedulable encoder input.

        Note that num_computed_tokens includes both locally cached
        blocks and externally cached blocks (via KVConnector).
        """
        if num_new_tokens == 0 or not request.has_encoder_inputs:
            return [], num_new_tokens, encoder_budget
        encoder_inputs_to_schedule: list[int] = []
        mm_positions = request.mm_positions
        assert mm_positions is not None
        assert len(mm_positions) > 0
        for i, pos_info in enumerate(mm_positions):
            start_pos = pos_info.offset
            num_encoder_tokens = pos_info.length

            # The encoder output is needed if the two ranges overlap:
            # [num_computed_tokens, num_computed_tokens + num_new_tokens) and
            # [start_pos, start_pos + num_encoder_tokens)
            if start_pos >= num_computed_tokens + num_new_tokens:
                # The encoder input is not needed in this step.
                break
            if start_pos + num_encoder_tokens <= num_computed_tokens:
                # The encoder input is already computed and stored
                # in the decoder's KV cache.
                continue

            if self.encoder_cache_manager.has_cache(request, i):
                # The encoder input is already computed and cached.
                continue

            # If no encoder input chunking is allowed, we do not want to
            # partially schedule a multimodal item. If the scheduled range would
            # only cover part of the mm input, roll back to before the mm item.
            if (self.scheduler_config.disable_chunked_mm_input
                    and num_computed_tokens < start_pos
                    and (num_computed_tokens + num_new_tokens)
                    < (start_pos + num_encoder_tokens)):
                num_new_tokens = start_pos - num_computed_tokens
                break

            if (not self.encoder_cache_manager.can_allocate(request, i)
                    or num_encoder_tokens > encoder_budget):
                # The encoder cache is full or the encoder budget is exhausted.
                # NOTE(woosuk): We assume that the encoder input tokens should
                # be processed altogether, as the encoder usually uses
                # bidirectional attention.
                if num_computed_tokens < start_pos:
                    # We only schedule the decoder tokens just before the
                    # encoder input.
                    num_new_tokens = start_pos - num_computed_tokens
                else:
                    # Because of prefix caching, num_computed_tokens is greater
                    # than start_pos even though its encoder input is not
                    # available. In this case, we can't schedule any token for
                    # the request in this step.
                    num_new_tokens = 0
                break

            encoder_budget -= num_encoder_tokens
            encoder_inputs_to_schedule.append(i)
        return encoder_inputs_to_schedule, num_new_tokens, encoder_budget

    def _validate_queue_consistency(self, location: str):  
        queue = self.kv_cache_manager.coordinator.block_pool.free_block_queue  
        actual_free = queue.get_all_free_blocks()  
        if len(actual_free) != queue.num_free_blocks:  
            logger.error(  
                f"Queue corruption detected at {location}: "  
                f"counter={queue.num_free_blocks}, actual={len(actual_free)}"  
            )  
            logger.error(f"Queue corrupted at {location}")
            # Log which blocks are in the queue  
            # logger.error(f"Blocks in queue: {[b.block_id for b in actual_free]}")  
            # raise AssertionError(f"Queue corrupted at {location}")
        
    
 
    def _validate_free_queue_integrity(self):  
        """Validate the free queue linked list is not corrupted."""  
        queue = self.kv_cache_manager.coordinator.block_pool.free_block_queue  
        
        if queue.num_free_blocks == 0:  
            # Empty queue should have head pointing to tail  
            if (queue.fake_free_list_head.next_free_block is not queue.fake_free_list_tail or  
                queue.fake_free_list_tail.prev_free_block is not queue.fake_free_list_head):  
                raise RuntimeError("Empty free queue has inconsistent pointers")  
            return  
        
        # Walk the list and check for broken links  
        curr = queue.fake_free_list_head.next_free_block  
        actual_count = 0  
        seen_blocks = set()  
        
        while curr is not queue.fake_free_list_tail:  
            if curr is None:  
                # Found broken link - this is the corruption  
                # raise RuntimeError(  
                #     f"Free queue corrupted: found None link at position {actual_count}"  
                # )  
                logger.error(  
                f"Free queue corrupted: found None link at position {actual_count}"  
            )  
            elif curr.block_id in seen_blocks:  
                logger.error(  
                    f"Free queue corrupted: block {curr.block_id} appears twice"  
                )  
            else:
                seen_blocks.add(curr.block_id)  
                actual_count += 1  
                curr = curr.next_free_block  
            
        if actual_count != queue.num_free_blocks:  
            # Log details for debugging  
            logger.error(  
                f"Free queue count mismatch: expected {queue.num_free_blocks}, "  
                f"found {actual_count}. Missing blocks: "  
            )  
            # raise RuntimeError(  
            #     f"Free queue count mismatch: expected {queue.num_free_blocks}, "  
            #     f"found {actual_count}"  
            # )
    
    # def _handle_block_merging_with_counts(  
    #     self,   
    #     request_blocks: dict[str, dict[int, list[int]]]   ,      
    # ) -> None:  
    #     """Apply reference changes using pre-calculated counts."""  
    #     if not request_blocks:  
    #         return  
            
    #     before_free = self.kv_cache_manager.coordinator.block_pool.get_num_free_blocks()  
        
    #     old_block_ids_cnt = []
    #     new_block_ids_cnt = []
            
    #     for req_id, group_blocks in request_blocks.items():  
    #         if req_id not in self.requests:  
    #             continue
    #         if not group_blocks:
    #             continue
            
    #         for group_idx, new_block_ids in group_blocks.items():  
    #             if group_idx == 0:  
    #                 continue  
                    
    #             manager = self.kv_cache_manager.coordinator.single_type_managers[group_idx]  
    #             req_blocks = manager.req_to_blocks.get(req_id, [])  
                
    #             old_block_ids_cnt += list(b.block_id for b in req_blocks if not b.is_null and b.block_id != -1)
    #             # Build new block list  
    #             new_req_blocks = []  
    #             for  block_id in new_block_ids:    
                   
    #                 block = self.kv_cache_manager.coordinator.block_pool.blocks[block_id]    
    #                 new_req_blocks.append(block)
                   
    #             manager.req_to_blocks.update({req_id: new_req_blocks}) #new_req_blocks
                
    #             new_block_ids_cnt+= list(b.block_id for b in new_req_blocks if not b.is_null and b.block_id != -1)
        
    #     # Apply reference count changes 
    #     old_cnt = Counter(old_block_ids_cnt)
    #     new_cnt = Counter(new_block_ids_cnt)

    #     block_ref_changes = {}
    #     all_keys = set(old_cnt.keys()).union(set(new_cnt.keys()))

    #     for key in all_keys:
    #         block_ref_changes[key] = new_cnt[key] - old_cnt[key]

    #     blocks_to_touch = []  
    #     blocks_to_free = []  
        
    #     for block_id, ref_change in block_ref_changes.items():  
    #         if ref_change > 0:  
    #             # Touch block ref_change times  
    #             block = self.kv_cache_manager.coordinator.block_pool.blocks[block_id]  
                
    #             blocks_to_touch.extend([block] * ref_change)  
                
    #         elif ref_change < 0:  
    #             # Free block |ref_change| times  
    #             block = self.kv_cache_manager.coordinator.block_pool.blocks[block_id]  
                
    #             if (block.prev_free_block is not None):  
    #                 # Block is already in free queue, don't free again  
    #                     logger.warning(f"Block {block_id} already in free queue, skipping free")  
    #                     continue  

    #             blocks_to_free.extend([block] * abs(ref_change)) 
                
        
    #     # Apply pool operations  
    #     if blocks_to_touch:  
    #         self.kv_cache_manager.coordinator.block_pool.touch((blocks_to_touch,))  
    #     if blocks_to_free:
    #         self.kv_cache_manager.coordinator.block_pool.free_blocks(blocks_to_free) 

    #     after_free = self.kv_cache_manager.coordinator.block_pool.get_num_free_blocks()  
    #     logger.info(f"Block merging freed {after_free - before_free} blocks")  
        
    def _handle_block_merging_with_counts(  
        self,   
        request_blocks: dict[str, dict[int, list[int]]],      
    ) -> None:  
        """Apply reference changes using pre-calculated counts."""  
        if not request_blocks:
            return
            
        before_free = self.kv_cache_manager.coordinator.block_pool.get_num_free_blocks()  
        
        # Using defaultdict for counting block IDs
        old_ref_counts = defaultdict(int)
        new_ref_counts = defaultdict(int)

        for req_id, group_blocks in request_blocks.items():  
            if req_id not in self.requests or not group_blocks:
                continue
            
            for group_idx, new_block_ids in group_blocks.items():  
                if group_idx == 0:  
                    continue  
                    
                manager = self.kv_cache_manager.coordinator.single_type_managers[group_idx]  
                req_blocks = manager.req_to_blocks.get(req_id, [])  
                
                # Count old references directly
                for block in req_blocks:
                    if not block.is_null and block.block_id != -1:
                        old_ref_counts[block.block_id] += 1

                # Build new blocks while filtering valid IDs
                new_req_blocks = []
                block_pool = self.kv_cache_manager.coordinator.block_pool
                
                for block_id in new_block_ids:    
                    block = block_pool.blocks[block_id]    
                    new_req_blocks.append(block)
                    if not block.is_null and block.block_id != -1:
                        new_ref_counts[block.block_id] += 1  # Count new references

                manager.req_to_blocks[req_id] = new_req_blocks

        # Calculate reference changes
        block_ref_changes = {}
        all_keys = set(new_ref_counts) | set(old_ref_counts)  # Combine keys efficiently

        for key in all_keys:
            block_ref_changes[key] = new_ref_counts[key] - old_ref_counts[key]

        blocks_to_touch = []  
        blocks_to_free = []  

        for block_id, ref_change in block_ref_changes.items():  
            block = self.kv_cache_manager.coordinator.block_pool.blocks[block_id]  # Access block once
            
            if ref_change > 0:  
                blocks_to_touch.extend([block] * ref_change)  
            elif ref_change < 0:  
                # if block.prev_free_block is not None:  # Check if already in free queue
                #     logger.warning(f"Block {block_id} already in free queue, skipping free")  
                # else:
                blocks_to_free.extend([block] * abs(ref_change)) 

        # Apply batch operations if necessary
        if blocks_to_touch:
            self.kv_cache_manager.coordinator.block_pool.touch((blocks_to_touch,))
        if blocks_to_free:
            self.kv_cache_manager.coordinator.block_pool.free_blocks(blocks_to_free)
        
        after_free = self.kv_cache_manager.coordinator.block_pool.get_num_free_blocks()  
        logger.info(f"Block merging freed {after_free - before_free} blocks")  
 
    def _handle_block_merging_with_counts_skipped(    
    self,     
    request_blocks: dict[str, dict[int, list[int]]],        
    req_id,  
    old_ref_counts: defaultdict[int],  
    new_ref_counts: defaultdict[int]  
) -> None:    
        """Apply reference changes using pre-calculated counts."""    
        group_blocks = request_blocks.get(req_id)
        if not group_blocks:  
            return  

        for group_idx, new_block_ids in group_blocks.items():    
            if group_idx == 0:  
                continue    

            manager = self.kv_cache_manager.coordinator.single_type_managers[group_idx]    
            req_blocks = manager.req_to_blocks.get(req_id, [])  

            # Count old references directly  
            for block in req_blocks:  
                if block.block_id != -1 and not block.is_null:  
                    old_ref_counts[block.block_id] += 1  
            
            # Build new blocks and count new references  
            new_req_blocks = []    
            block_pool = self.kv_cache_manager.coordinator.block_pool
            
            for block_id in new_block_ids:      
                if block_id == -1:      
                    new_req_blocks.append(manager.block_pool.null_block)      
                else:      
                    block = block_pool.blocks[block_id]      
                    if block.is_null:      
                        raise ValueError(f"Cannot map to null block {block_id}")      
                    new_req_blocks.append(block)  
                    new_ref_counts[block_id] += 1  

            manager.req_to_blocks[req_id] = new_req_blocks

    def _handle_counts(self, block_ref_changes: defaultdict[int]):  
        """Apply reference count changes efficiently."""  
        if not block_ref_changes:  
            return  
        
        blocks_to_touch = []  
        blocks_to_free = [] 
        before_free = self.kv_cache_manager.coordinator.block_pool.get_num_free_blocks()    
        for block_id, ref_change in block_ref_changes.items():  
            block = self.kv_cache_manager.coordinator.block_pool.blocks[block_id]  # Access block once
            
            if ref_change > 0:  
                blocks_to_touch.extend([block] * ref_change)  
            elif ref_change < 0:  
                if block.prev_free_block is not None:  # Check if already in free queue
                    logger.warning(f"Block {block_id} already in free queue, skipping free")  
                else:
                    blocks_to_free.extend([block] * abs(ref_change)) 

        # Apply batch operations if necessary
        if blocks_to_touch:
            self.kv_cache_manager.coordinator.block_pool.touch((blocks_to_touch,))
        if blocks_to_free:
            self.kv_cache_manager.coordinator.block_pool.free_blocks(blocks_to_free)
        
        after_free = self.kv_cache_manager.coordinator.block_pool.get_num_free_blocks()  
        logger.info(f"Block merging freed {after_free - before_free} blocks")  
        
    
    def _validate_block_merge_mapping(  
        self,  
        block_merge_mapping: dict[str, dict[int, dict[int, int]]]  
    ) -> None:  
        """Validate the block merge mapping is consistent."""  
        for req_id, group_mappings in block_merge_mapping.items():  
            if req_id not in self.requests:  
                continue  
                
            for group_idx, mappings in group_mappings.items():  
                if group_idx >= self.kv_cache_manager.num_kv_cache_groups:  
                    raise ValueError(f"Invalid group_idx {group_idx}")  
                    
                manager = self.kv_cache_manager.coordinator.single_type_managers[group_idx]  
                req_blocks = manager.req_to_blocks.get(req_id, [])  
                req_block_ids = {b.block_id for b in req_blocks if not b.is_null}  
                
                for old_block_id, new_block_id in mappings.items():  
                    if old_block_id not in req_block_ids:  
                        logger.warning(  
                            f"Block {old_block_id} not found in request {req_id} group {group_idx}"  
                        )  
                    
                    # Validate new block ID range  
                    if new_block_id < 0 or new_block_id >= len(self.kv_cache_manager.coordinator.block_pool.blocks):  
                        raise ValueError(f"Invalid new_block_id {new_block_id} in mapping")
    
    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        sampled_token_ids = model_runner_output.sampled_token_ids
        spec_token_ids = model_runner_output.spec_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        num_nans_in_logits = model_runner_output.num_nans_in_logits
        ####### sefi
        # Extract your new fields  
        # blocks_count = model_runner_output.blocks_count  
        block_merge_mapping = model_runner_output.updated_block_table  

        # ref_cnt_change = model_runner_output.ref_cnt_change  # Your similarity merging creates this  
        # old_block_ids_cnt = []
        # new_block_ids_cnt = []
        if block_merge_mapping:
            self._handle_block_merging_with_counts(block_merge_mapping)
            # self._handle_counts(ref_cnt_change)

        if False:
            before_free = self.kv_cache_manager.coordinator.block_pool.get_num_free_blocks()  
            blocks_already_freed = set()
            if blocks_count:  
                for block_id, count in blocks_count.items():  
                    block = self.kv_cache_manager.coordinator.block_pool.blocks[block_id]

                    if (block.ref_cnt == 0 and   
                                block.prev_free_block is not None and  
                                block.block_id not in blocks_already_freed):  
                                self.kv_cache_manager.coordinator.block_pool.free_block_queue.remove(block)  
                                blocks_already_freed.add(block.block_id)  

                    block.ref_cnt = count 
            
            if blocks_to_free:
                
                blocks_to_free_objs = []  # Track what we've removed 
                for block_id in blocks_to_free: 
                    block = self.kv_cache_manager.coordinator.block_pool.blocks[block_id]  
                    block.ref_cnt = 1 
                    if block.ref_cnt == 1 and block.block_id not in blocks_already_freed:  
                        blocks_to_free_objs.append(block) 
              
                if blocks_to_free_objs:  
                    self.kv_cache_manager.coordinator.block_pool.free_blocks(blocks_to_free_objs)
            
            after_free = self.kv_cache_manager.coordinator.block_pool.get_num_free_blocks()  
            if blocks_count:
                logger.info(f"Block merging freed {after_free - before_free} blocks")
                
        if False:
            assert self.kv_cache_manager.coordinator.block_pool.free_block_queue.fake_free_list_head.next_free_block is not None    
            before_free = self.kv_cache_manager.coordinator.block_pool.get_num_free_blocks()   
                
            blocks_to_free_objs = []    
            blocks_already_freed = set()  
            for req_id, request in self.requests.items(): 
                if req_id not in block_merge_mapping.keys():  
                    continue
                for group_idx in range(1, self.kv_cache_manager.num_kv_cache_groups):  
                    if group_idx not in block_merge_mapping[req_id]:  
                        continue
                    manager = self.kv_cache_manager.coordinator.single_type_managers[group_idx]  
                    req_blocks = manager.req_to_blocks.get(req_id, [])  
                    
                    # Track which blocks to remove  
                    blocks_to_remove_indices = []  
                    
                    for i, block in enumerate(set(req_blocks)):  
                        if block.block_id in block_merge_mapping[req_id][group_idx].keys():  
                            new_block_id = block_merge_mapping[req_id][group_idx][block.block_id]  
                            new_block = self.kv_cache_manager.coordinator.block_pool.blocks[new_block_id]  
                            
                            # Handle new block  
                            if new_block.prev_free_block is not None:  
                                self.kv_cache_manager.coordinator.block_pool.free_block_queue.remove(new_block)  
                            new_block.ref_cnt += 1  
                            
                            # Handle old block
                            if block.ref_cnt > 1:  
                                block.ref_cnt -= 1  

                            if block.ref_cnt == 1:  
                                blocks_to_free_objs.append(block)  
                            
                            # Replace in the list  
                            req_blocks[i] = new_block  
                            
                            # If the old block was cached, we need to update the cached count  
                            if block._block_hash is not None:  
                                blocks_to_remove_indices.append(i)  
                    
                    # Update num_cached_block if we removed cached blocks  
                    if blocks_to_remove_indices and req_id in manager.num_cached_block:  
                        # Recalculate the number of cached blocks  
                        num_cached = sum(1 for b in req_blocks if b._block_hash is not None)  
                        manager.num_cached_block[req_id] = num_cached
            
             # Free all old blocks at once  
            if blocks_to_free_objs:  
                blocks_to_free_objs.reverse()  # Free in reverse order for proper eviction  
                self.kv_cache_manager.coordinator.block_pool.free_blocks(blocks_to_free_objs)  
                
                after_free = self.kv_cache_manager.coordinator.block_pool.get_num_free_blocks()  
                logger.info(f"Block merging freed {after_free - before_free} blocks")  
            
            self._validate_queue_consistency("line 945")          

        if False:  
            assert self.kv_cache_manager.coordinator.block_pool.free_block_queue.fake_free_list_head.next_free_block is not None  
            before_free = self.kv_cache_manager.coordinator.block_pool.get_num_free_blocks()  
            
            blocks_to_free_objs = []  
            
            blocks_removed_from_queue = set()  # Track blocks removed from free queue  
            
            self._validate_queue_consistency("line 867")  
            
            for req_id, group_mappings in block_merge_mapping.items():  
                if req_id not in self.requests:  
                    continue 

                blocks_already_processed = set()  # Track blocks we've already handled   
                
                for group_idx in range(1, self.kv_cache_manager.num_kv_cache_groups): 
                    if group_idx not in group_mappings:  
                        continue   
                    manager = self.kv_cache_manager.coordinator.single_type_managers[group_idx]  
                    req_blocks = manager.req_to_blocks.get(req_id, [])  
                    
                    new_req_blocks = []  
                    for block in req_blocks:
                        if block.is_null or block.block_id == -1:  
                            new_req_blocks.append(block)  
                            continue    
                        if block.block_id in group_mappings[group_idx]:  
                            new_block_id = group_mappings[group_idx][block.block_id]  
                            new_block = self.kv_cache_manager.coordinator.block_pool.blocks[new_block_id]  
                            
                            # Handle the NEW block (cached block being reused)  
                            # if new_block.block_id not in blocks_already_processed:  
                                # Remove from free queue if it's there (check pointers, not ref_cnt)  
                            if new_block.prev_free_block is not None or new_block.next_free_block is not None:  
                                if new_block.block_id not in blocks_removed_from_queue:  
                                    self.kv_cache_manager.coordinator.block_pool.free_block_queue.remove(new_block)  
                                    blocks_removed_from_queue.add(new_block.block_id)  
                            # Increment ref_cnt for the new block  
                            if new_block.block_id not in blocks_already_processed:
                                new_block.ref_cnt += 1  
                                blocks_already_processed.add(new_block.block_id)  
                            
                            # Handle the OLD block (being replaced)  
                            if block.block_id not in blocks_already_processed:  
                                if block.ref_cnt > 1:  
                                    block.ref_cnt -= 1
                                elif block.ref_cnt == 1:
                                    blocks_to_free_objs.append(block)  
                                blocks_already_processed.add(block.block_id)  
                            
                            new_req_blocks.append(new_block)  
                        else:  
                            new_req_blocks.append(block)  
                    
                    # Replace the entire block list  
                    manager.req_to_blocks[req_id] = new_req_blocks  
            
            # Free all old blocks at once  
            if blocks_to_free_objs:  
                blocks_to_free_objs.reverse()  # Free in reverse order for proper eviction  
                self.kv_cache_manager.coordinator.block_pool.free_blocks(blocks_to_free_objs)  
                
                after_free = self.kv_cache_manager.coordinator.block_pool.get_num_free_blocks()  
                logger.info(f"Block merging freed {after_free - before_free} blocks")  
            
            self._validate_queue_consistency("line 945")          
       
        if False: ### main working version
                assert self.kv_cache_manager.coordinator.block_pool.free_block_queue.fake_free_list_head.next_free_block is not None    
                before_free = self.kv_cache_manager.coordinator.block_pool.get_num_free_blocks()   
                
                blocks_to_free_objs = []    
                blocks_already_freed = set()  

                self._validate_queue_consistency("line 867")
                
                for req_id, request in self.requests.items():  
                    for group_idx in range(1, self.kv_cache_manager.num_kv_cache_groups):  
                        manager = self.kv_cache_manager.coordinator.single_type_managers[group_idx]  
                        req_blocks = manager.req_to_blocks.get(req_id, [])  
                        
                        # Build a NEW list with updated block references  
                        new_req_blocks = []  
                        for block in req_blocks:  
                            if block.block_id in block_merge_mapping[req_id][group_idx]:  
                                # Use the merged block  
                                new_block_id = block_merge_mapping[req_id][group_idx][block.block_id]  
                                new_block = self.kv_cache_manager.coordinator.block_pool.blocks[new_block_id]  
                                new_req_blocks.append(new_block)  
                                
                                # Handle reference counting  
                                if new_block.prev_free_block is not None:  
                                    self.kv_cache_manager.coordinator.block_pool.free_block_queue.remove(new_block)  
                                new_block.ref_cnt += 1  
                                
                                # if block.ref_cnt > 1:  
                                #     block.ref_cnt -= 1
                                # Old block to be freed  
                                if block.ref_cnt == 1:  
                                    blocks_to_free_objs.append(block)  
                            else:  
                                # Keep the original block  
                                new_req_blocks.append(block)  
                        
                        # Replace the entire list with the new one  
                        manager.req_to_blocks[req_id] = new_req_blocks
                       
                # for i, block in enumerate(req_blocks):    
                        #     # Skip fake blocks  
                        #     if block.block_id == -1 or block.is_null:  
                        #         continue  
                        #     if block.block_id not in group_mappings[group_idx]:  
                        #         continue 
                              
                                
                        #     new_block_id = group_mappings[group_idx][block.block_id]    
                        #     new_block = self.kv_cache_manager.coordinator.block_pool.blocks[new_block_id]  

                        #     if new_block_id == block.block_id:  
                        #         continue
                            
                        #     # Remove new_block from free queue if it's there  
                        #     if (new_block.ref_cnt == 0 and not block.is_null and 
                        #         new_block.prev_free_block is not None and    
                        #         new_block.block_id not in blocks_already_freed):    
                        #         self.kv_cache_manager.coordinator.block_pool.free_block_queue.remove(new_block)    
                        #         blocks_already_freed.add(new_block.block_id)    
                           
                        #     # Increment new block's ref_cnt  
                        #     new_block.ref_cnt += 1  
                        #     # new_block.incr_ref()
                                                        
                        #     # Decrement old block - but only if not already processed  
                        #     if block.block_id not in blocks_already_freed:  
                        #         if block.ref_cnt > 1:    
                        #             # block.decr_ref()
                        #             block.ref_cnt -= 1    
                        #         elif block.ref_cnt == 1:  
                        #             # block.ref_cnt = 1  # just to be explicit
                        #             blocks_to_free_objs.append(block)    
                        #             blocks_already_freed.add(block.block_id) 
                                

                        #     if block._block_hash is not None:  
                        #                 blocks_to_remove_indices.append(i)

                        #     # Update the request's block list
                        #     # new_req_blocks = list(req_blocks[:])   
                        #     req_blocks[i] = new_block
                        #     manager.req_to_blocks[req_id] = deepcopy(req_blocks) # req_blocks[:]  #
                                            
                
                        #     # DON'T update req_blocks[i] here - worker already updated its state  
                    
                        # Update num_cached_block if we removed cached blocks    
                        # if blocks_to_remove_indices and req_id in manager.num_cached_block:    
                        # # Use new_req_blocks, not req_blocks  
                        #     num_cached = sum(1 for b in new_req_blocks if b._block_hash is not None)    
                        #     manager.num_cached_block[req_id] = num_cached
                
                # Free all blocks at once    
                if blocks_to_free_objs:  
                    blocks_to_free_objs.reverse()  
                    self.kv_cache_manager.coordinator.block_pool.free_blocks(blocks_to_free_objs)  
                    
                    after_free = self.kv_cache_manager.coordinator.block_pool.get_num_free_blocks()    
                    logger.info(f"Block merging freed {after_free - before_free} blocks")
                
                self._validate_queue_consistency("line 945")

        if False: ##  
            assert self.kv_cache_manager.coordinator.block_pool.free_block_queue.fake_free_list_head.next_free_block is not None      
            before_free = self.kv_cache_manager.coordinator.block_pool.get_num_free_blocks()     
           
            # blocks_already_incremented = set()  # NEW: Track blocks whose ref_cnt we've incremented  
        
            self._validate_queue_consistency("line 867")  
            
            for req_id, group_mappings in block_merge_mapping.items():  
                if req_id not in self.requests:  
                    continue  

                blocks_to_free_all = {}      
                blocks_to_touch_all = {}  # NEW: Track blocks to touch    

                for group_idx in range(1, self.kv_cache_manager.num_kv_cache_groups):  
                    if group_idx not in group_mappings:  
                        continue  
                        
                    manager = self.kv_cache_manager.coordinator.single_type_managers[group_idx]  
                    req_blocks = manager.req_to_blocks.get(req_id, [])  
                    
                    # Track all old blocks before updating  
                    current_block_ids = {block.block_id for block in req_blocks   
                                    if not block.is_null and block.block_id != -1}  
                    
                    new_req_blocks = []  
                    # new_block_ids = set()  
                    
                    for block in req_blocks:  
                        if block.is_null or block.block_id == -1:  
                            new_req_blocks.append(block)  
                            continue  
                            
                        if block.block_id in group_mappings[group_idx]:  
                            new_block_id = group_mappings[group_idx][block.block_id]  
                            new_block = self.kv_cache_manager.coordinator.block_pool.blocks[new_block_id]  
                            new_req_blocks.append(new_block)  
                            # new_block_ids.add(new_block_id)  
                            
                            # Remove from free queue if needed  
                            # if new_block.prev_free_block is not None:  
                            #     self.kv_cache_manager.coordinator.block_pool.free_block_queue.remove(new_block)  
                            # new_block.ref_cnt += 1  
                            if new_block.block_id not in current_block_ids:
                                blocks_to_touch_all[new_block.block_id] = new_block

                            blocks_to_free_all[block.block_id] = block
                        
                        else:  
                            new_req_blocks.append(block)  
                            # new_block_ids.add(block.block_id)  
                    
                    # Update the request's block list  
                    manager.req_to_blocks[req_id] = new_req_blocks

                if blocks_to_touch_all:    
                    self.kv_cache_manager.coordinator.block_pool.touch((list(blocks_to_touch_all.values()),)) 
                if blocks_to_free_all:    
                        # Consider reversing if blocks are from same request chain  
                    blocks_to_free_objs = list(blocks_to_free_all.values())
                        # blocks_to_free_objs.reverse()    
                    self.kv_cache_manager.coordinator.block_pool.free_blocks(blocks_to_free_objs) 
                    
                    # Free blocks that are no longer referenced  
                    # blocks_to_check = old_block_ids - new_block_ids  
                    # for block_id in blocks_to_check:  
                    #     block = self.kv_cache_manager.coordinator.block_pool.blocks[block_id]  
                    #     if block.ref_cnt > 1:  
                    #         block.ref_cnt -= 1  
                    #     elif block.ref_cnt == 1:  
                    #         blocks_to_free_objs.append(block)
                        
            
              

            # Free all blocks at once      
               
                
                after_free = self.kv_cache_manager.coordinator.block_pool.get_num_free_blocks()      
                logger.info(f"Block merging freed {after_free - before_free} blocks")  
            
            self._validate_queue_consistency("line 945")

        if False:    
            assert self.kv_cache_manager.coordinator.block_pool.free_block_queue.fake_free_list_head.next_free_block is not None          
            before_free = self.kv_cache_manager.coordinator.block_pool.get_num_free_blocks()         
            
            blocks_to_free_objs = []         
            
            # Track all blocks that will lose references across ALL requests  
            blocks_losing_refs = {}  # block_id -> count of how many requests stop using it  
            
            for req_id, group_mappings in block_merge_mapping.items():  
                if req_id not in self.requests:      
                    continue  
                
                request = self.requests[req_id]  
                
                # Skip preempted requests  
                if request.status == RequestStatus.PREEMPTED:    
                    continue  
                
                # Skip stopping requests (as before)  
                is_stopping = False  
                
                # Determine if request will stop after this batch  
                if hasattr(request, 'output_token_ids') and request.output_token_ids:  
                    last_token_id = request.output_token_ids[-1]  
                    sampling_params = request.sampling_params  
                    if sampling_params:  
                        if not sampling_params.ignore_eos and last_token_id == request.eos_token_id:  
                            is_stopping = True  
                        elif last_token_id in (sampling_params.stop_token_ids or ()):  
                            is_stopping = True  
                        elif request.num_output_tokens >= request.max_tokens:  
                            is_stopping = True  
                        elif request.num_tokens >= self.max_model_len:  
                            is_stopping = True  
                            
                if is_stopping:    
                    continue  
                    
                for group_idx in range(1, self.kv_cache_manager.num_kv_cache_groups):  
                    if group_idx not in group_mappings:      
                        continue  
                        
                    manager = self.kv_cache_manager.coordinator.single_type_managers[group_idx]      
                    req_blocks = manager.req_to_blocks.get(req_id, [])      
                    
                    new_req_blocks = []      
                    unique_blocks_for_req = set()  
                    
                    for block in req_blocks:      
                        if block.is_null or block.block_id == -1:      
                            new_req_blocks.append(block)      
                            continue  
                            
                        if block.block_id in group_mappings[group_idx]:      
                            new_block_id = group_mappings[group_idx][block.block_id]      
                            new_block = self.kv_cache_manager.coordinator.block_pool.blocks[new_block_id]      
                            new_req_blocks.append(new_block)      
                            
                            # Increment ref_cnt for new blocks (once per request)  
                            if new_block.block_id not in unique_blocks_for_req:  
                                if new_block.prev_free_block is not None:      
                                    self.kv_cache_manager.coordinator.block_pool.free_block_queue.remove(new_block)      
                                new_block.ref_cnt += 1      
                                unique_blocks_for_req.add(new_block.block_id)  
                            
                            # Track old blocks that will lose this request's reference  
                            if block.block_id not in unique_blocks_for_req:  
                                blocks_losing_refs[block.block_id] = blocks_losing_refs.get(block.block_id, 0) + 1  
                                unique_blocks_for_req.add(block.block_id)  
                        else:      
                            new_req_blocks.append(block)      
                    
                    manager.req_to_blocks[req_id] = new_req_blocks      
            
            # Now free blocks that lost ALL their references  
            for block_id, lost_refs in blocks_losing_refs.items():  
                # Only free if this block was actually referenced by the requests being processed  
                # AND we're tracking it as losing references  
                block = self.kv_cache_manager.coordinator.block_pool.blocks[block_id]  
                blocks_to_free_objs.append(block)  
            
            # Free all blocks at once          
            if blocks_to_free_objs:        
                self.kv_cache_manager.coordinator.block_pool.free_blocks(blocks_to_free_objs)        
                
                after_free = self.kv_cache_manager.coordinator.block_pool.get_num_free_blocks()          
                logger.info(f"Block merging freed {after_free - before_free} blocks")

        if False:  
            before_free = self.kv_cache_manager.coordinator.block_pool.get_num_free_blocks()  
            
            # Track blocks to touch and blocks that might need freeing  
            blocks_to_touch_all = []  
            blocks_to_free_all = []  
            
            for req_id, group_mappings in  block_merge_mapping.items():  
                if req_id not in self.requests:  
                    continue
               
                blocks_to_touch = {}
                blocks_to_free = {} 
                    
                request = self.requests[req_id]  

                # Skip preempted requests  
                if request.status == RequestStatus.PREEMPTED:    
                    continue  
                
                # Skip stopping requests (as before)  
                is_stopping = False  
                
                # Determine if request will stop after this batch  
                if hasattr(request, 'output_token_ids') and request.output_token_ids:  
                    last_token_id = request.output_token_ids[-1]  
                    sampling_params = request.sampling_params  
                    if sampling_params:  
                        if not sampling_params.ignore_eos and last_token_id == request.eos_token_id:  
                            is_stopping = True  
                        elif last_token_id in (sampling_params.stop_token_ids or ()):  
                            is_stopping = True  
                        elif request.num_output_tokens >= request.max_tokens:  
                            is_stopping = True  
                        elif request.num_tokens >= self.max_model_len:  
                            is_stopping = True  
                            
                if is_stopping:    
                    continue   
                
                

                for group_idx in range(1, self.kv_cache_manager.num_kv_cache_groups):  
                    if group_idx not in group_mappings:  
                        continue  
                        
                    manager = self.kv_cache_manager.coordinator.single_type_managers[group_idx]  
                    req_blocks = manager.req_to_blocks.get(req_id, [])  
                    current_block_ids = {block.block_id for block in req_blocks}  

                    new_req_blocks = []  
                    # old_blocks_used = set()  
                    
                    for block in req_blocks:  
                        if block.is_null or block.block_id == -1:  
                            new_req_blocks.append(block)  
                            continue  
                            
                        if block.block_id in group_mappings[group_idx]:  
                            new_block_id = group_mappings[group_idx][block.block_id]  
                            new_block = self.kv_cache_manager.coordinator.block_pool.blocks[new_block_id]  
                            new_req_blocks.append(new_block) 

                            
                            # Track new blocks to touch 
                            if new_block.block_id not in current_block_ids: 
                                blocks_to_touch[new_block_id]=new_block  
                            
                            blocks_to_free[block.block_id]=block
                            
                        else:  
                            new_req_blocks.append(block)  
                    
                    manager.req_to_blocks[req_id] = new_req_blocks

                blocks_to_touch_all += list(blocks_to_touch.values())  #[b for b in blocks_to_touch]
                blocks_to_free_all +=  list(blocks_to_free.values()) #list(b for b in blocks_to_free.values() if b) #[b for b in blocks_to_free]
            
            # Touch all new blocks at once (more efficient)  
            if blocks_to_touch_all:  
                self.kv_cache_manager.coordinator.block_pool.touch((blocks_to_touch_all,))  
          
            # Free blocks through the block pool (handles ref_cnt properly)  
            if blocks_to_free_all:
                final_free = []
                for b in blocks_to_free_all:
                    if b.ref_cnt < 1 or b.is_null:
                        continue
                    if b in blocks_to_touch_all:
                        continue

                    final_free.append(b)   

                if not all(b.ref_cnt ==1 for b in final_free):
                    logger.error("Blocks to free have ref_cnt > 1") 
                self.kv_cache_manager.coordinator.block_pool.free_blocks(final_free)  
                
                after_free = self.kv_cache_manager.coordinator.block_pool.get_num_free_blocks()  
                logger.info(f"Block merging freed {after_free - before_free} blocks")        
        
        if False:    
            before_free = self.kv_cache_manager.coordinator.block_pool.get_num_free_blocks()    
            
            # Use dicts to track unique blocks by block_id  
            blocks_to_touch_all = {}  # {block_id: KVCacheBlock}  
            blocks_to_free_all = {}   # {block_id: KVCacheBlock}  
            
            for req_id, group_mappings in block_merge_mapping.items():    
                if req_id not in self.requests:    
                    continue  
                
                request = self.requests[req_id]    
        
                # Skip preempted or stopping requests  
                if request.status == RequestStatus.PREEMPTED:  
                    continue  
                    
                is_stopping = False  
                if hasattr(request, 'output_token_ids') and request.output_token_ids:  
                    last_token_id = request.output_token_ids[-1]  
                    sampling_params = request.sampling_params  
                    if sampling_params:  
                        if not sampling_params.ignore_eos and last_token_id == request.eos_token_id:  
                            is_stopping = True  
                        elif last_token_id in (sampling_params.stop_token_ids or ()):  
                            is_stopping = True  
                        elif request.num_output_tokens >= request.max_tokens:  
                            is_stopping = True  
                        elif request.num_tokens >= self.max_model_len:  
                            is_stopping = True  
                            
                if is_stopping:  
                    continue     
                
                for group_idx in range(1, self.kv_cache_manager.num_kv_cache_groups):    
                    if group_idx not in group_mappings:    
                        continue    
                        
                    manager = self.kv_cache_manager.coordinator.single_type_managers[group_idx]    
                    req_blocks = manager.req_to_blocks.get(req_id, [])    
                    current_block_ids = {block.block_id for block in req_blocks}    
        
                    new_req_blocks = []    
                    
                    for block in req_blocks:    
                        if block.is_null or block.block_id == -1:    
                            new_req_blocks.append(block)    
                            continue    
                            
                        if block.block_id in group_mappings[group_idx]:    
                            new_block_id = group_mappings[group_idx][block.block_id]    
                            new_block = self.kv_cache_manager.coordinator.block_pool.blocks[new_block_id]    
                            new_req_blocks.append(new_block)    
        
                            # Track new blocks to touch (only if not already in request)    
                            if new_block.block_id not in current_block_ids:    
                                blocks_to_touch_all[new_block.block_id] = new_block    
                            
                            # ALWAYS track old blocks for freeing when they're replaced  
                            blocks_to_free_all[block.block_id] = block  
                            
                        else:    
                            new_req_blocks.append(block)    
                    
                    manager.req_to_blocks[req_id] = new_req_blocks  
            
            # Touch all new blocks at once    
            if blocks_to_touch_all:    
                self.kv_cache_manager.coordinator.block_pool.touch((list(blocks_to_touch_all.values()),))    
            
            # Free blocks through the block pool    
            if blocks_to_free_all:  
                # Convert to list and filter out blocks that were just touched  
                final_free = [b for b in blocks_to_free_all.values()   
                            if b.block_id not in blocks_to_touch_all]  
                
                # Additional safety check: only free blocks with ref_cnt > 0  
                final_free = [b for b in final_free if b.ref_cnt > 0 and not b.is_null]  
                
                if final_free:  
                    self.kv_cache_manager.coordinator.block_pool.free_blocks(final_free)    
                    
                    after_free = self.kv_cache_manager.coordinator.block_pool.get_num_free_blocks()    
                    logger.info(f"Block merging freed {after_free - before_free} blocks")
       
        if False:    
            before_free = self.kv_cache_manager.coordinator.block_pool.get_num_free_blocks()    
            
            blocks_to_touch_all = {}  # {block_id: KVCacheBlock}  
            blocks_to_free_all = {}   # {block_id: KVCacheBlock}  
            
            for req_id, group_mappings in block_merge_mapping.items():    
                if req_id not in self.requests:    
                    continue  
                
                request = self.requests[req_id]    
        
                # Skip all special request states  
                if request.status in [  
                    RequestStatus.PREEMPTED,  
                    RequestStatus.FINISHED_ABORTED,  
                    # RequestStatus.WAITING_FOR_REMOTE_KV  
                ]:  
                    continue  
                    
                is_stopping = False  
                if hasattr(request, 'output_token_ids') and request.output_token_ids:  
                    last_token_id = request.output_token_ids[-1]  
                    sampling_params = request.sampling_params  
                    if sampling_params:  
                        if not sampling_params.ignore_eos and last_token_id == request.eos_token_id:  
                            is_stopping = True  
                        elif last_token_id in (sampling_params.stop_token_ids or ()):  
                            is_stopping = True  
                        elif request.num_output_tokens >= request.max_tokens:  
                            is_stopping = True  
                        elif request.num_tokens >= self.max_model_len:  
                            is_stopping = True  
                            
                            
                if is_stopping:  
                    continue     
                
                # Track ALL block IDs in the final list for this request  
                final_block_ids = set()  
                
                for group_idx in range(1, self.kv_cache_manager.num_kv_cache_groups):    
                    if group_idx not in group_mappings:    
                        continue    
                        
                    manager = self.kv_cache_manager.coordinator.single_type_managers[group_idx]    
                    req_blocks = manager.req_to_blocks.get(req_id, [])    
                    current_block_ids = {block.block_id for block in req_blocks}    
        
                    new_req_blocks = []    
                    
                    for block in req_blocks:    
                        if block.is_null or block.block_id == -1:    
                            new_req_blocks.append(block)    
                            continue  
                            
                        if block.block_id in group_mappings[group_idx]:    
                            new_block_id = group_mappings[group_idx][block.block_id]    
                            new_block = self.kv_cache_manager.coordinator.block_pool.blocks[new_block_id]    
                            new_req_blocks.append(new_block)    
        
                            # Track new blocks to touch (only if not already in request)    
                            if new_block_id not in current_block_ids:    
                                blocks_to_touch_all[new_block_id] = new_block    
                            
                            # Track old blocks for potential freeing  
                            blocks_to_free_all[block.block_id] = block  
                            
                        else:    
                            new_req_blocks.append(block)  
                    
                    # Collect all block IDs that remain in the final list  
                    for block in new_req_blocks:  
                        if not block.is_null and block.block_id != -1:  
                            final_block_ids.add(block.block_id)  
                    
                    manager.req_to_blocks[req_id] = new_req_blocks  
                
                # Remove blocks that are still in the final list from freeing  
                blocks_to_free_final = {  
                    block_id: block for block_id, block in blocks_to_free_all.items()  
                    if block_id not in final_block_ids  
                }  
            
            # Touch all new blocks at once    
            if blocks_to_touch_all:    
                self.kv_cache_manager.coordinator.block_pool.touch((list(blocks_to_touch_all.values()),))    
            
            # Free blocks through the block pool    
            if blocks_to_free_final:  
                final_free = list(blocks_to_free_final.values())  
                
                # Additional safety check: only free blocks with ref_cnt > 0  
                final_free = [b for b in final_free if b.ref_cnt > 0 and not b.is_null]  
                
                if final_free:  
                    self.kv_cache_manager.coordinator.block_pool.free_blocks(final_free)    
                    
                    after_free = self.kv_cache_manager.coordinator.block_pool.get_num_free_blocks()    
                    logger.info(f"Block merging freed {after_free - before_free} blocks")       
            ################## end sefi #######

        if False:    
            assert self.kv_cache_manager.coordinator.block_pool.free_block_queue.fake_free_list_head.next_free_block is not None        
            before_free = self.kv_cache_manager.coordinator.block_pool.get_num_free_blocks()       
            
            self._validate_queue_consistency("line 867")    
            
            for req_id, group_mappings in block_merge_mapping.items():    
                if req_id not in self.requests:    
                    continue    
                       
                
                for group_idx in range(1, self.kv_cache_manager.num_kv_cache_groups):    
                    if group_idx not in group_mappings:    
                        continue

                    blocks_to_free_all = {}        
                    blocks_to_touch_all = {}      
                    final_block_ids = set()  # Track blocks that remain in final list      
                        
                    manager = self.kv_cache_manager.coordinator.single_type_managers[group_idx]    
                    req_blocks = manager.req_to_blocks.get(req_id, [])    
                    
                    current_block_ids = {block.block_id for block in req_blocks     
                                    if not block.is_null and block.block_id != -1}    
                    
                    new_req_blocks = []    
                    
                    for block in req_blocks:    
                        if block.is_null or block.block_id == -1:    
                            new_req_blocks.append(block)    
                            continue    
                            
                        if block.block_id in group_mappings[group_idx]:    
                            new_block_id = group_mappings[group_idx][block.block_id]    
                            new_block = self.kv_cache_manager.coordinator.block_pool.blocks[new_block_id]    
                            new_req_blocks.append(new_block)    
                            
                            if new_block.block_id not in current_block_ids:  
                                blocks_to_touch_all[new_block.block_id] = new_block  
                            
                            blocks_to_free_all[block.block_id] = block  
                            
                        else:    
                            new_req_blocks.append(block)    
                    
                    # Collect block IDs that remain in this group's final list  
                    for block in new_req_blocks:  
                        if not block.is_null and block.block_id != -1:  
                            final_block_ids.add(block.block_id)  
                    
                    manager.req_to_blocks[req_id] = new_req_blocks  
                    
                # Only free blocks that are NOT in the final list  
                    blocks_to_free_final = {  
                        block_id: block for block_id, block in blocks_to_free_all.items()  
                        if block_id not in final_block_ids  
                    }  
                    
                    if blocks_to_touch_all:      
                        self.kv_cache_manager.coordinator.block_pool.touch((list(blocks_to_touch_all.values()),))   
                    if blocks_to_free_final:      
                        blocks_to_free_objs = list(blocks_to_free_final.values())  
                        self.kv_cache_manager.coordinator.block_pool.free_blocks(blocks_to_free_objs)   
            
            after_free = self.kv_cache_manager.coordinator.block_pool.get_num_free_blocks()        
            logger.info(f"Block merging freed {after_free - before_free} blocks")    
            
            self._validate_queue_consistency("line 945")

        # old_block_ids_cnt = defaultdict(int) 
        # new_block_ids_cnt = defaultdict(int) 
        
        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: Optional[SpecDecodingStats] = None

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            request = self.requests.get(req_id)
            if request is None:
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism).
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = sampled_token_ids[
                req_index] if sampled_token_ids else []

            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id))
            if scheduled_spec_token_ids:
                # num_computed_tokens represents the number of tokens
                # processed in the current step, considering scheduled
                # tokens and rejections. If some tokens are rejected,
                # num_computed_tokens is decreased by the number of rejected
                # tokens, where is given by:
                # len(scheduled_spec_token_ids) + 1 - len(generated_token_ids).
                num_tokens_rejected = (len(scheduled_spec_token_ids) + 1 -
                                       len(generated_token_ids))
                request.num_computed_tokens -= num_tokens_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=len(scheduled_spec_token_ids),
                    num_accepted_tokens=len(generated_token_ids) - 1)

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            kv_transfer_params = None
            status_before_stop = request.status

            # Check for stop and update request status.
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids)

            # Stop checking for pooler models.
            pooler_output = None
            if pooler_outputs:
                pooler_output = pooler_outputs[req_index]
                stopped = check_stop(request, self.max_model_len,
                                     pooler_output)

            if stopped:
                kv_transfer_params = self._free_request(request)
                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)
            # else:
            #     if block_merge_mapping:
                    
            #         self._handle_block_merging_with_counts_skipped(block_merge_mapping, req_id, old_block_ids_cnt, new_block_ids_cnt)

            
            # Extract sample logprobs if needed.
            if request.sampling_params is not None \
                and request.sampling_params.logprobs is not None and logprobs:
                # NOTE: once we support N tokens per step (spec decode),
                # the outer lists can be of length > 1.
                new_logprobs = logprobs.slice(req_index, req_index + 1)

            if new_token_ids and self.structured_output_manager.should_advance(
                    request):
                # NOTE: structured_output_request
                # should not be None if use_structured_output, we have
                # check above, so safe to ignore type warning
                request.structured_output_request.grammar.accept_tokens(  # type: ignore[union-attr]
                    req_id, new_token_ids)

            # spec_token_ids comes from the model runner output
            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # Add newly generated spec token ids to the request.
            if spec_token_ids is not None:
                if self.structured_output_manager.should_advance(request):
                    metadata = request.structured_output_request
                    # Needs to happen after new_token_ids are accepted.
                    request.spec_token_ids = metadata.grammar.validate_tokens(  # type: ignore[union-attr]
                        spec_token_ids[req_index])
                else:
                    request.spec_token_ids = spec_token_ids[req_index]

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if new_token_ids or pooler_output is not None \
                or kv_transfer_params:

                # Add EngineCoreOutput for this Request.
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=request.get_finished_reason(),
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        kv_transfer_params=kv_transfer_params,
                        num_cached_tokens=request.num_cached_tokens,
                    ))

            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

        # Remove the stopped requests from the running and waiting queues.
        if stopped_running_reqs:
            self.running = [
                req for req in self.running if req not in stopped_running_reqs
            ]
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)

        # KV Connector: update state for finished KV Transfers.
        self._update_from_kv_xfer_finished(model_runner_output)

        # self._handle_counts( old_block_ids_cnt, new_block_ids_cnt) # sefi         
        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }

        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():
                # Set finished request set in EngineCoreOutputs for this client.
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(
                        finished_requests=finished_set)
            finished_req_ids.clear()

        
        if engine_core_outputs:
            # Return stats to only one of the front-ends.
            next(iter(engine_core_outputs.values())).scheduler_stats = (
                self.make_stats(spec_decoding_stats))

        
        return engine_core_outputs

    def _update_request_with_output(
        self,
        request: Request,
        new_token_ids: list[int],
    ) -> tuple[list[int], bool]:
        # Append generated tokens and check for stop. Note that if
        # a request is still being prefilled, we expect the model runner
        # to return empty token ids for the request.
        stopped = False
        for num_new, output_token_id in enumerate(new_token_ids, 1):
            request.append_output_token_ids(output_token_id)

            # Check for stop and update request state.
            # This must be called before we make the EngineCoreOutput.
            stopped = check_stop(request, self.max_model_len)
            if stopped:
                del new_token_ids[num_new:]  # Trim new tokens if needed.
                break
        return new_token_ids, stopped

    def _free_encoder_inputs(self, request: Request) -> None:
        cached_encoder_input_ids = (
            self.encoder_cache_manager.get_cached_input_ids(request))
        # OPTIMIZATION: Avoid list(set) if the set is empty.
        if not cached_encoder_input_ids:
            return

        # Here, we use list(set) to avoid modifying the set while iterating
        # over it.
        for input_id in list(cached_encoder_input_ids):
            mm_positions = request.mm_positions[input_id]
            start_pos = mm_positions.offset
            num_tokens = mm_positions.length
            if start_pos + num_tokens <= request.num_computed_tokens:
                # The encoder output is already processed and stored
                # in the decoder's KV cache.
                self.encoder_cache_manager.free_encoder_input(
                    request, input_id)

    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs)."""
        return len(self.running), len(self.waiting)

    def add_request(self, request: Request) -> None:
        self.waiting.add_request(request)
        self.requests[request.request_id] = request
        if self.log_stats:
            request.record_event(EngineCoreEventType.QUEUED)

    def finish_requests(
        self,
        request_ids: Union[str, Iterable[str]],
        finished_status: RequestStatus,
    ) -> None:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.
        """
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids, )
        else:
            request_ids = set(request_ids)

        running_requests_to_remove = []
        waiting_requests_to_remove = []
        valid_requests = []

        # First pass: collect requests to remove from queues
        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None:
                # Invalid request ID.
                continue

            valid_requests.append(request)
            if request.status == RequestStatus.RUNNING:
                running_requests_to_remove.append(request)
            else:
                waiting_requests_to_remove.append(request)

        # Remove all requests from queues at once for better efficiency
        for request in running_requests_to_remove:
            self.running.remove(request)
        if waiting_requests_to_remove:
            self.waiting.remove_requests(waiting_requests_to_remove)

        # Second pass: set status and free requests
        for request in valid_requests:
            request.status = finished_status
            self._free_request(request)

    def _free_request(self, request: Request) -> Optional[dict[str, Any]]:
        assert request.is_finished()

        delay_free_blocks, kv_xfer_params = self._connector_finished(request)
        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        self.finished_req_ids.add(request_id)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        if not delay_free_blocks:
            self._free_blocks(request)

        return kv_xfer_params

    def _free_blocks(self, request: Request):
        assert request.is_finished()
        self._validate_queue_consistency("line 1383 in _free_blocks()")
        self.kv_cache_manager.free(request)
        self._validate_queue_consistency("line 1385 in _free_blocks()")
        self.kv_cache_manager.free_block_hashes(request)
        self._validate_queue_consistency("line 1387 in _free_blocks()")
        del self.requests[request.request_id]

    def get_num_unfinished_requests(self) -> int:
        return len(self.waiting) + len(self.running)

    def has_finished_requests(self) -> bool:
        return len(self.finished_req_ids) > 0

    def reset_prefix_cache(self) -> bool:
        return self.kv_cache_manager.reset_prefix_cache()

    def make_stats(
        self,
        spec_decoding_stats: Optional[SpecDecodingStats] = None,
    ) -> Optional[SchedulerStats]:
        if not self.log_stats:
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
        assert prefix_cache_stats is not None
        return SchedulerStats(
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
            kv_cache_usage=self.kv_cache_manager.usage,
            prefix_cache_stats=prefix_cache_stats,
            spec_decoding_stats=spec_decoding_stats,
            num_corrupted_reqs=sum(req.is_output_corrupted
                                   for req in self.running),
        )

    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: Optional[SpecDecodingStats],
        num_draft_tokens: int,
        num_accepted_tokens: int,
    ) -> Optional[SpecDecodingStats]:
        if not self.log_stats:
            return None
        if spec_decoding_stats is None:
            spec_decoding_stats = SpecDecodingStats.new(self.num_spec_tokens)
        spec_decoding_stats.observe_draft(
            num_draft_tokens=num_draft_tokens,
            num_accepted_tokens=num_accepted_tokens)
        return spec_decoding_stats

    def shutdown(self) -> None:
        if self.kv_event_publisher:
            self.kv_event_publisher.shutdown()

    ########################################################################
    # KV Connector Related Methods
    ########################################################################

    def get_kv_connector(self) -> Optional[KVConnectorBase_V1]:
        return self.connector

    def _connector_finished(
            self, request: Request) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Invoke the KV connector request_finished() method if applicable.

        Returns optional kv transfer parameters to be included with the
        request outputs.
        """
        if self.connector is None:
            return False, None

        # (block_ids, ) = self.kv_cache_manager.get_block_ids(request.request_id)# sefi
        # return self.connector.request_finished(request, block_ids)
        ## sefi
        # block_ids = self.kv_cache_manager.get_block_ids(request.request_id)
        
        # Keep per-group block ids (tuple of lists) and pass them as-is.
        block_ids = self.kv_cache_manager.get_block_ids(request.request_id)

        # Try calling connector.request_finished with tuple-of-lists (preferred).
        try:
            return self.connector.request_finished(request, block_ids)
        except TypeError:
            # Fallback: some connectors expect a single flattened list.
            try:
                from numpy import concatenate
                flat = concatenate(block_ids).tolist()
                return self.connector.request_finished(request, flat)
            except Exception as e:
                logger.exception("connector.request_finished fallback failed: %s", e)
                return False, None

    def _update_waiting_for_remote_kv(self, request: Request) -> bool:
        """
        KV Connector: check if the request_id is finished_recving.

        The finished_recving_kv_req_ids list is populated
        on the previous steps()'s update_from_output based
        on the worker side connector.

        When the kv transfer is ready, we cache the blocks
        and the request state will be moved back to WAITING from
        WAITING_FOR_REMOTE_KV.
        """
        assert self.connector is not None
        if request.request_id not in self.finished_recving_kv_req_ids:
            return False

        # Now that the blocks are ready, actually cache them.
        (block_ids, ) = self.kv_cache_manager.get_block_ids(request.request_id)
        num_computed_tokens = len(block_ids) * self.block_size
        # Handle the case where num request tokens less then one block.
        num_computed_tokens = min(num_computed_tokens, request.num_tokens)
        if num_computed_tokens == request.num_tokens:
            num_computed_tokens -= 1
        # This will cache the blocks iff caching is enabled.
        self.kv_cache_manager.cache_blocks(request, num_computed_tokens)

        # Update the request state for scheduling.
        request.num_computed_tokens = num_computed_tokens

        # Return that we are ready.
        self.finished_recving_kv_req_ids.remove(request.request_id)
        return True

    def _update_from_kv_xfer_finished(self,
                                      model_runner_output: ModelRunnerOutput):
        """
        KV Connector: update the scheduler state based on the output.

        The Worker side connectors add finished_recving and
        finished_sending reqs to the output.
        * if finished_sending: free the blocks
        # if finished_recving: add to state so we can
            scheduler the request during the next step.
        """
        # KV Connector:: update recv and send status from last step.
        for req_id in (model_runner_output.finished_recving or ()):
            logger.debug("Finished recving KV transfer for request %s", req_id)
            self.finished_recving_kv_req_ids.add(req_id)
        for req_id in (model_runner_output.finished_sending or ()):
            logger.debug("Finished sending KV transfer for request %s", req_id)
            self._free_blocks(self.requests[req_id])
