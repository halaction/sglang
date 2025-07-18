import logging
from typing import Union

from fastapi import Request
from fastapi.responses import ORJSONResponse, StreamingResponse

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ErrorResponse,
    TokenizeRequest,
    TokenizeResponse,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.managers.io_struct import GenerateReqInput

logger = logging.getLogger(__name__)


class OpenAIServingChatTokenize(OpenAIServingChat):
    def _convert_to_internal_request(
        self,
        request: TokenizeRequest,
    ) -> tuple[GenerateReqInput, TokenizeRequest]:
        if request.prompt is not None:
            adapted_request = GenerateReqInput(text=request.prompt)
        elif request.messages is not None:
            chat_completion_request = ChatCompletionRequest(
                messages=request.messages,
                model=request.model,
            )
            adapted_request, _ = super()._convert_to_internal_request(
                chat_completion_request
            )
        else:
            adapted_request = GenerateReqInput(input_ids=[])

        return adapted_request, request

    async def _handle_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: TokenizeRequest,
        raw_request: Request,
    ) -> StreamingResponse:
        """Handle streaming chat tokenize request"""
        raise NotImplementedError

    async def _handle_non_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: TokenizeRequest,
        raw_request: Request,
    ) -> Union[TokenizeResponse, ErrorResponse, ORJSONResponse]:
        """Handle non-streaming chat tokenize request"""
        try:
            tokenized_obj = await self.tokenizer_manager._tokenize_one_request(
                adapted_request
            )
            input_ids = tokenized_obj.input_ids
        except ValueError as e:
            return self.create_error_response(str(e))

        response = TokenizeResponse(
            count=len(input_ids),
            max_model_len=self.tokenizer_manager.context_len,
            tokens=input_ids,
        )

        return response


OpenAIServingTokenize = OpenAIServingChatTokenize
