from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
import torch


# Implement a custom logits processor.
#    It works as a state machine:
#     - While in "forced mode," it forces tokens from the current fixed segment.
#     - When not forcing, it lets the model generate freely (filling the blanks).
#     - When the generated token matches the expected token from the next forced segment,
#       it switches back to forced mode.
class TemplateFillingProcessor(LogitsProcessor):
    def __init__(self, prompt_length, forced_segments):
        self.prompt_length = prompt_length  # length (in tokens) of the prompt only
        self.forced_segments = forced_segments  # list of fixed segments (list of token ids)
        self.current_segment = 0  # index in forced_segments we are enforcing
        self.segment_pos = 0      # position within the current forced segment
        self.in_forced_mode = True  # start by enforcing the first fixed segment

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        current_length = input_ids.shape[1]

        # Do not interfere with the prompt.
        if current_length <= self.prompt_length:
            return scores

        # In forced mode, we want to force tokens from the current fixed segment.
        if self.in_forced_mode:
            segment = self.forced_segments[self.current_segment]
            # If we've already generated some tokens in this segment, check if they match.
            if self.segment_pos < len(segment):
                # Check the last generated token.
                last_generated = input_ids[0, current_length - 1].item()
                expected = segment[self.segment_pos]
                if last_generated == expected:
                    self.segment_pos += 1
                    # If we've finished the current fixed segment, switch to free mode.
                    if self.segment_pos >= len(segment):
                        self.in_forced_mode = False
            # If still in forced mode, force the next token.
            if self.in_forced_mode and self.segment_pos < len(segment):
                target_token_id = segment[self.segment_pos]
                forced_scores = torch.full_like(scores, -float('inf'))
                forced_scores[:, target_token_id] = scores[:, target_token_id] + 1000.0
                return forced_scores
            # Otherwise, exit forced mode.
            self.in_forced_mode = False

        # In free mode (for the blank), let the model generate freely.
        # But check if the generated output appears to start the next fixed segment.
        if not self.in_forced_mode:
            # If there is a next fixed segment to enforce...
            if self.current_segment < len(self.forced_segments) - 1:
                next_segment = self.forced_segments[self.current_segment + 1]
                # Check if the last generated token matches the first token of the next fixed segment.
                last_generated = input_ids[0, current_length - 1].item()
                if last_generated == next_segment[0]:
                    # Transition to forcing the next segment.
                    self.current_segment += 1
                    self.segment_pos = 1  # the first token is already generated
                    self.in_forced_mode = True
                    # Force the next token if available.
                    if self.segment_pos < len(next_segment):
                        target_token_id = next_segment[self.segment_pos]
                        forced_scores = torch.full_like(scores, -float('inf'))
                        forced_scores[:, target_token_id] = scores[:, target_token_id] + 1000.0
                        return forced_scores
        return scores

