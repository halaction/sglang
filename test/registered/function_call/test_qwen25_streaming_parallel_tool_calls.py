import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.qwen25_detector import Qwen25Detector
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(1.0, "default")


class TestQwen25StreamingParallelToolCalls(unittest.TestCase):
    """Test case for streaming parallel tool call parsing with Qwen25 Detector."""

    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_current_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "state": {"type": "string"},
                            "unit": {"type": "string"},
                        },
                        "required": ["city", "state", "unit"],
                    },
                ),
            ),
        ]
        self.detector = Qwen25Detector()

    def _accumulate_tool_calls(self, tool_calls, result):
        if not result.calls:
            return
        for call in result.calls:
            if call.tool_index is None:
                continue
            while len(tool_calls) <= call.tool_index:
                tool_calls.append({"name": "", "parameters": ""})
            if call.name:
                tool_calls[call.tool_index]["name"] = call.name
            if call.parameters:
                tool_calls[call.tool_index]["parameters"] += call.parameters

    def test_streaming_parallel_tool_calls(self):
        """Test parsing two streaming parallel tool calls with Qwen25 Detector."""

        chunks = [
            "<tool_call>\n",
            '{"name": "get_current_weather", "arguments": {"city": "New York City", "state": "NY"',
            ', "unit": "celsius"}}',
            "\n</tool_call>",
            "\n<tool_call>\n",
            '{"name": "get_current_weather", "arguments": {"city": "Baltimore", "state": "MD", "unit": "celsius"}}',
            "\n</tool_call>",
        ]

        tool_calls = []
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            self._accumulate_tool_calls(tool_calls, result)

        self.assertEqual(len(tool_calls), 2, "Should parse 2 tool calls")
        self.assertEqual(tool_calls[0]["name"], "get_current_weather")
        self.assertEqual(tool_calls[1]["name"], "get_current_weather")

        params0 = json.loads(tool_calls[0]["parameters"])
        self.assertEqual(
            params0,
            {"city": "New York City", "state": "NY", "unit": "celsius"},
        )

        params1 = json.loads(tool_calls[1]["parameters"])
        self.assertEqual(
            params1,
            {"city": "Baltimore", "state": "MD", "unit": "celsius"},
        )


if __name__ == "__main__":
    unittest.main()
