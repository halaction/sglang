import json
import logging
import re
from typing import List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    _GetInfoFunc,
)

logger = logging.getLogger(__name__)


class Qwen25Detector(BaseFormatDetector):
    """
    Detector for Qwen 2.5 and Qwen 3 model function call format.

    Format Structure:
    ```
    <tool_call>\n{"name":"func1", "arguments":{...}}\n</tool_call>\n<tool_call>\n{"name":"func2", "arguments":{...}}\n</tool_call>
    ```

    Key Components:
    - Tool Call Tags: `<tool_call>` and `</tool_call>` wrap each individual call
    - Function Call Object: JSON object with "name" and "arguments" fields

    Reference: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct?chat_template=default
    """

    def __init__(self):
        """
        Initializes the detector with necessary state variables.
        """
        super().__init__()
        self.bot_token = "<tool_call>\n"
        self.eot_token = "\n</tool_call>"
        self.tool_call_separator = "\n"

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a Qwen 2.5 format tool call."""
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])

        # Find all <tool_call>\n...\n</tool_call> blocks
        pattern = rf"{re.escape(self.bot_token)}(.*?){re.escape(self.eot_token)}"
        match_result_list = re.findall(pattern, text, re.DOTALL)
        calls = []
        for match_result in match_result_list:
            try:
                parsed_call = json.loads(match_result.strip())
                calls.extend(self.parse_base_json(parsed_call, tools))
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Failed to parse JSON part: {match_result}, JSON parse error: {str(e)}"
                )
                continue
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing for Qwen 2.5 tool calls.
        Parses complete <tool_call>...</tool_call> blocks from the buffer.
        """
        self._buffer += new_text
        current_text = self._buffer
        start_token = self.bot_token.strip()
        end_token = self.eot_token.strip()

        def _strip_end_tokens(text: str) -> str:
            if not text:
                return text
            return text.replace(self.eot_token, "").replace(end_token, "")

        if start_token not in current_text:
            partial_len = self._ends_with_partial_token(current_text, start_token)
            if partial_len:
                normal_text = current_text[:-partial_len]
                self._buffer = current_text[-partial_len:]
            else:
                normal_text = current_text
                self._buffer = ""
            return StreamingParseResult(normal_text=_strip_end_tokens(normal_text))

        calls = []
        normal_text_parts = []
        buffer = current_text

        while True:
            start_idx = buffer.find(start_token)
            if start_idx == -1:
                break

            if start_idx > 0:
                prefix = _strip_end_tokens(buffer[:start_idx])
                if prefix:
                    normal_text_parts.append(prefix)
                buffer = buffer[start_idx:]

            end_idx = buffer.find(end_token, len(start_token))
            if end_idx == -1:
                break

            payload = buffer[len(start_token) : end_idx].strip()
            if payload:
                try:
                    parsed_call = json.loads(payload)
                    calls.extend(self.parse_base_json(parsed_call, tools))
                except json.JSONDecodeError as e:
                    logger.warning(
                        "Failed to parse JSON part: %s, JSON parse error: %s",
                        payload,
                        str(e),
                    )

            buffer = buffer[end_idx + len(end_token) :]

        if buffer and start_token not in buffer:
            partial_len = self._ends_with_partial_token(buffer, start_token)
            if partial_len:
                normal_text_parts.append(_strip_end_tokens(buffer[:-partial_len]))
                buffer = buffer[-partial_len:]
            else:
                normal_text_parts.append(_strip_end_tokens(buffer))
                buffer = ""

        self._buffer = buffer

        return StreamingParseResult(
            normal_text="".join(normal_text_parts),
            calls=calls,
        )

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin='<tool_call>\n{"name":"' + name + '", "arguments":',
            end="}\n</tool_call>",
            trigger="<tool_call>",
        )
