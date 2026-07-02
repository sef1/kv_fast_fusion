# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# GLM-specific reasoning parser for GLM-4.5/5/5.1 models.
# (Wiki patches 0006 + 0011 folded, rebased to vLLM v0.19.1.)
#
# GLM models use multiple tokens to mark the end of reasoning:
# - 154842 (</think>)            : standard end-of-thinking marker (special=False)
# - 154829 (<|observation|>)     : observation/tool-result marker (special=True, EOS)
# - 154827 (<|user|>)            : user input boundary (special=True, EOS)
# - 154843 (<tool_call>)         : tool call start (special=False)
#
# IMPORTANT: The GLM-5.1 tokenizer uses PLAIN XML tags (e.g., <think>, <tool_call>)
# NOT HTML comment-style tags (e.g., <!--think-->, <!--tool_call-->).
# The vocab keys are the plain XML forms.

from collections.abc import Sequence

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser


class GLMReasoningParser(DeepSeekR1ReasoningParser):
    """
    Reasoning parser for GLM-4.5/5/5.1 models.

    Extends DeepSeekR1ReasoningParser to handle GLM-specific thinking
    end markers and the MTP chunk boundary problem.
    """

    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        vocab = self.vocab
        # CRITICAL: Use the ACTUAL token strings from GLM-5.1 tokenizer,
        # which are plain XML tags, NOT HTML comment-style tags.
        self.tool_call_start_id = vocab.get("<tool_call>")       # 154843
        self.observation_id = vocab.get("<|observation|>")       # 154829
        self.input_end_id = vocab.get("<|user|>")                # 154827

        # All token IDs that mark the end of reasoning.
        self.reasoning_end_ids = {self.end_token_id}   # 154842 (</think>) is always included
        if self.tool_call_start_id is not None:
            self.reasoning_end_ids.add(self.tool_call_start_id)
        if self.observation_id is not None:
            self.reasoning_end_ids.add(self.observation_id)
        if self.input_end_id is not None:
            self.reasoning_end_ids.add(self.input_end_id)

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """Check if reasoning has ended in the given token sequence.

        Recognizes all GLM end-of-reasoning markers, not just 154842 (</think>).
        """
        start_token_id = self.start_token_id
        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] == start_token_id:
                return False
            if input_ids[i] in self.reasoning_end_ids:
                return True
        return False

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Sequence[int]
    ) -> bool:
        """Check if reasoning ends in the delta IDs during streaming."""
        return any(tid in self.reasoning_end_ids for tid in delta_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """Extract token IDs after any GLM reasoning-end marker.

        Override base class which only recognizes end_token_id (154842).
        We need to also recognize 154843, 154829, 154827 so that
        content token IDs are correctly extracted for the tool parser.
        """
        last_end_idx = -1
        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] in self.reasoning_end_ids:
                last_end_idx = i
                break

        if last_end_idx == -1:
            return []
        if last_end_idx == len(input_ids) - 1:
            return []
        return input_ids[last_end_idx + 1:]

    def _id_to_text(self, token_id: int) -> str:
        """Get the text for a token ID using a cached reverse vocab mapping."""
        if not hasattr(self, '_id_to_text_cache'):
            self._id_to_text_cache = {v: k for k, v in self.vocab.items()}
        return self._id_to_text_cache.get(token_id, "")

    def _strip_trailing_eos_tags(self, text: str) -> str:
        """Strip trailing EOS tag text (<|observation|> or <|user|>) from text."""
        for eos_id in [self.observation_id, self.input_end_id]:
            if eos_id is not None:
                eos_text = self._id_to_text(eos_id)
                if eos_text and text.endswith(eos_text):
                    text = text[:-len(eos_text)]
        return text

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """Extract reasoning content from a streaming delta.

        Handles GLM-specific reasoning-end markers with proper tag handling.
        Key insight: the tool parser (Glm4MoeModelToolParser) accumulates
        delta_text into a buffer and searches for <tool_call> within it.
        Any stray text BEFORE <tool_call> in the buffer gets emitted as
        a plain text block. So we must ensure that the content we pass does
        NOT include the <tool_call> tag prefix or any other stray text when a
        tool_call follows.
        """
        # Skip single start/end tokens (same as base class)
        if len(delta_token_ids) == 1 and (
            delta_token_ids[0] in [self.start_token_id, self.end_token_id]
        ):
            return None

        # Detect which GLM markers are in this delta
        has_tool_call_start = (
            self.tool_call_start_id is not None
            and self.tool_call_start_id in delta_token_ids
        )
        has_think_end = self.end_token_id in delta_token_ids
        has_observation = (
            self.observation_id is not None
            and self.observation_id in delta_token_ids
        )
        has_input_end = (
            self.input_end_id is not None
            and self.input_end_id in delta_token_ids
        )
        start_in_previous = self.start_token_id in previous_token_ids
        start_in_delta = self.start_token_id in delta_token_ids

        # ---- Case A: </think> + <tool_call> in same chunk ----
        # MTP case where both markers arrive together.
        # Split at </think> boundary: content starts from <tool_call> onward.
        # Do NOT include </think> in content - it would leak as plain text.
        if has_think_end and has_tool_call_start:
            think_end_text = self._id_to_text(self.end_token_id)  # "</think>"
            think_end_pos = delta_text.rfind(think_end_text) if think_end_text else -1
            if think_end_pos >= 0:
                reasoning = delta_text[:think_end_pos]
                # If start_token also in delta, strip it from reasoning
                if start_in_delta:
                    start_text = self._id_to_text(self.start_token_id)
                    start_pos = reasoning.find(start_text) if start_text else -1
                    if start_pos >= 0:
                        reasoning = reasoning[start_pos + len(start_text):]
                # Content starts AFTER </think> (skip the </think> tag itself)
                content = delta_text[think_end_pos + len(think_end_text):]
                content = self._strip_trailing_eos_tags(content)
                return DeltaMessage(
                    reasoning=reasoning if reasoning else None,
                    content=content if content else None,
                )
            return DeltaMessage(reasoning=delta_text if delta_text else None)

        # ---- Case B: <tool_call> in delta (no </think> in same chunk) ----
        # Model transitioned directly to tool call without </think>.
        # Pass full delta_text as content for the tool parser.
        # If start_token is in delta, strip it first.
        if has_tool_call_start and not has_think_end:
            content = delta_text
            if start_in_delta:
                start_text = self._id_to_text(self.start_token_id)
                start_pos = content.find(start_text) if start_text else -1
                if start_pos >= 0:
                    reasoning_prefix = content[:start_pos]
                    content = content[start_pos + len(start_text):]
                else:
                    reasoning_prefix = ""
            else:
                reasoning_prefix = ""
            content = self._strip_trailing_eos_tags(content)
            if reasoning_prefix and content:
                return DeltaMessage(reasoning=reasoning_prefix, content=content)
            elif content:
                return DeltaMessage(content=content)
            elif reasoning_prefix:
                return DeltaMessage(reasoning=reasoning_prefix)
            return None

        # ---- Case C: </think> in delta (no <tool_call> in same chunk) ----
        # Do NOT put </think> in content - it would leak to the tool parser
        # as plain text. The tool parser will find <tool_call> in the next chunk.
        if has_think_end:
            end_index = delta_text.rfind(self._id_to_text(self.end_token_id))
            if end_index >= 0:
                reasoning = delta_text[:end_index]
                if start_in_delta:
                    start_text = self._id_to_text(self.start_token_id)
                    start_pos = reasoning.find(start_text) if start_text else -1
                    if start_pos >= 0:
                        reasoning = reasoning[start_pos + len(start_text):]
                has_eos = has_observation or has_input_end
                if has_eos:
                    # </think> + EOS: strip both tags from content
                    content = delta_text[end_index:]
                    content = self._strip_trailing_eos_tags(content)
                    think_text = self._id_to_text(self.end_token_id)
                    if think_text and content.endswith(think_text):
                        content = content[:-len(think_text)]
                    return DeltaMessage(
                        reasoning=reasoning if reasoning else None,
                        content=content if content else None,
                    )
                else:
                    # </think> alone: content=None, tool parser gets <tool_call> in next chunk
                    return DeltaMessage(
                        reasoning=reasoning if reasoning else None,
                        content=None,
                    )
            return DeltaMessage(reasoning=delta_text if delta_text else None)

        # ---- Case D: <|observation|> or <|user|> in delta, no </think> ----
        if has_observation or has_input_end:
            marker_id = self.observation_id if has_observation else self.input_end_id
            marker_text = self._id_to_text(marker_id)
            marker_pos = delta_text.find(marker_text) if marker_text else -1

            if marker_pos >= 0:
                reasoning = delta_text[:marker_pos]
                after_marker = delta_text[marker_pos + len(marker_text):]
                if reasoning:
                    return DeltaMessage(
                        reasoning=reasoning,
                        content=after_marker if after_marker else None,
                    )
                else:
                    return DeltaMessage(
                        content=after_marker if after_marker else None
                    )
            else:
                # Tag text filtered by skip_special_tokens
                return DeltaMessage(reasoning=delta_text if delta_text else None)

        # ---- Fallback: handle tokens with no special markers (patch 0011) ----
        # When no reasoning_end marker has been seen yet, default to reasoning.
        # GLM-5.1 frequently omits the 154841 (<think>) start token, so the
        # reasoning text arrives without a start marker. Before any
        # reasoning_end_ids token appears, we assume content is reasoning.
        # Once a reasoning_end marker is in previous_token_ids, subsequent
        # tokens are real content (or tool call arguments).
        if any(tid in previous_token_ids for tid in self.reasoning_end_ids):
            return DeltaMessage(content=delta_text)

        if start_in_previous or start_in_delta:
            return DeltaMessage(reasoning=delta_text)

        return DeltaMessage(reasoning=delta_text)
