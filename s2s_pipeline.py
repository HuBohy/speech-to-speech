import logging
import os
import sys
from copy import copy
from pathlib import Path
from queue import Queue
from threading import Event
from typing import Optional
from sys import platform

from s2s.VAD.vad_handler import VADHandler
from s2s.arguments_classes.language_model_arguments import LanguageModelHandlerArguments
from s2s.arguments_classes.mlx_language_model_arguments import (
    MLXLanguageModelHandlerArguments,
)
from s2s.arguments_classes.module_arguments import ModuleArguments
from s2s.arguments_classes.parler_tts_arguments import ParlerTTSHandlerArguments
from s2s.arguments_classes.socket_receiver_arguments import SocketReceiverArguments
from s2s.arguments_classes.socket_sender_arguments import SocketSenderArguments
from s2s.arguments_classes.vad_arguments import VADHandlerArguments
from s2s.arguments_classes.whisper_stt_arguments import WhisperSTTHandlerArguments
from s2s.arguments_classes.melo_tts_arguments import MeloTTSHandlerArguments
import torch
import nltk
from rich.console import Console
from transformers import (
    HfArgumentParser,
)

from s2s.utils.thread_manager import ThreadManager

# Ensure that the necessary NLTK resources are available
try:
    nltk.data.find("tokenizers/punkt_tab")
except (LookupError, OSError):
    nltk.download("punkt_tab")
try:
    nltk.data.find("tokenizers/averaged_perceptron_tagger_eng")
except (LookupError, OSError):
    nltk.download("averaged_perceptron_tagger_eng")

# caching allows ~50% compilation time reduction
# see https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit#heading=h.o2asbxsrp1ma
CURRENT_DIR = Path(__file__).resolve().parent
os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(CURRENT_DIR, "tmp")

console = Console()
logging.getLogger("numba").setLevel(logging.WARNING)  # quiet down numba logs


def prepare_args(args, prefix):
    """
    Rename arguments by removing the prefix and prepares the gen_kwargs.
    """

    gen_kwargs = {}
    for key in copy(args.__dict__):
        if key.startswith(prefix):
            value = args.__dict__.pop(key)
            new_key = key[len(prefix) + 1 :]  # Remove prefix and underscore
            if new_key.startswith("gen_"):
                gen_kwargs[new_key[4:]] = value  # Remove 'gen_' and add to dict
            else:
                args.__dict__[new_key] = value

    args.__dict__["gen_kwargs"] = gen_kwargs


class S2SPipeline():
    def __init__(self) -> None:
        parser = HfArgumentParser(
            (
                ModuleArguments,
                SocketReceiverArguments,
                SocketSenderArguments,
                VADHandlerArguments,
                WhisperSTTHandlerArguments,
                LanguageModelHandlerArguments,
                MLXLanguageModelHandlerArguments,
                ParlerTTSHandlerArguments,
                MeloTTSHandlerArguments,
            )
        )

        # 0. Parse CLI arguments
        if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
            # Parse configurations from a JSON file if specified
            (
                module_kwargs,
                socket_receiver_kwargs,
                socket_sender_kwargs,
                vad_handler_kwargs,
                whisper_stt_handler_kwargs,
                language_model_handler_kwargs,
                mlx_language_model_handler_kwargs,
                parler_tts_handler_kwargs,
                melo_tts_handler_kwargs,
            ) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        else:
            # Parse arguments from command line if no JSON file is provided
            (
                module_kwargs,
                socket_receiver_kwargs,
                socket_sender_kwargs,
                vad_handler_kwargs,
                whisper_stt_handler_kwargs,
                language_model_handler_kwargs,
                mlx_language_model_handler_kwargs,
                parler_tts_handler_kwargs,
                melo_tts_handler_kwargs,
            ) = parser.parse_args_into_dataclasses()

        # 1. Handle logger
        global logger
        logging.basicConfig(
            level=module_kwargs.log_level.upper(),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        logger = logging.getLogger(__name__)

        # torch compile logs
        if module_kwargs.log_level == "debug":
            torch._logging.set_logs(graph_breaks=True, recompiles=True, cudagraphs=True)

        def optimal_mac_settings(mac_optimal_settings: Optional[str], *handler_kwargs):
            if mac_optimal_settings:
                for kwargs in handler_kwargs:
                    if hasattr(kwargs, "device"):
                        kwargs.device = "mps"
                    if hasattr(kwargs, "mode"):
                        kwargs.mode = "local"
                    if hasattr(kwargs, "stt"):
                        kwargs.stt = "whisper-mlx"
                    if hasattr(kwargs, "llm"):
                        kwargs.llm = "mlx-lm"
                    if hasattr(kwargs, "tts"):
                        kwargs.tts = "melo"

        optimal_mac_settings(
            module_kwargs.local_mac_optimal_settings,
            module_kwargs,
        )

        if platform == "darwin":
            if module_kwargs.device == "cuda":
                raise ValueError(
                    "Cannot use CUDA on macOS. Please set the device to 'cpu' or 'mps'."
                )
            if module_kwargs.llm != "mlx-lm":
                logger.warning(
                    "For macOS users, it is recommended to use mlx-lm. You can activate it by passing --llm mlx-lm."
                )
            if module_kwargs.tts != "melo":
                logger.warning(
                    "If you experiences issues generating the voice, considering setting the tts to melo."
                )

        # 2. Prepare each part's arguments
        def overwrite_device_argument(common_device: Optional[str], *handler_kwargs):
            if common_device:
                for kwargs in handler_kwargs:
                    if hasattr(kwargs, "lm_device"):
                        kwargs.lm_device = common_device
                    if hasattr(kwargs, "tts_device"):
                        kwargs.tts_device = common_device
                    if hasattr(kwargs, "stt_device"):
                        kwargs.stt_device = common_device

        # Call this function with the common device and all the handlers
        overwrite_device_argument(
            module_kwargs.device,
            language_model_handler_kwargs,
            mlx_language_model_handler_kwargs,
            parler_tts_handler_kwargs,
            whisper_stt_handler_kwargs,
        )

        prepare_args(whisper_stt_handler_kwargs, "stt")
        prepare_args(language_model_handler_kwargs, "lm")
        prepare_args(mlx_language_model_handler_kwargs, "mlx_lm")
        prepare_args(parler_tts_handler_kwargs, "tts")
        prepare_args(melo_tts_handler_kwargs, "melo")

        # 3. Build the pipeline
        self.stop_event = Event()
        # used to stop putting received audio chunks in queue until all setences have been processed by the TTS
        self.should_listen = Event()
        self.recv_audio_chunks_queue = Queue()
        self.send_audio_chunks_queue = Queue()
        self.spoken_prompt_queue = Queue()
        self.text_prompt_queue = Queue()
        self.lm_response_queue = Queue()

        if module_kwargs.mode == "local":
            from s2s.connections.local_audio_streamer import LocalAudioStreamer

            local_audio_streamer = LocalAudioStreamer(
                input_queue=self.recv_audio_chunks_queue, output_queue=self.send_audio_chunks_queue
            )
            self.comms_handlers = [local_audio_streamer]
            self.should_listen.set()
        else:
            from s2s.connections.socket_receiver import SocketReceiver
            from s2s.connections.socket_sender import SocketSender

            self.comms_handlers = [
                SocketReceiver(
                    self.stop_event,
                    self.recv_audio_chunks_queue,
                    self.should_listen,
                    host=socket_receiver_kwargs.recv_host,
                    port=socket_receiver_kwargs.recv_port,
                    chunk_size=socket_receiver_kwargs.chunk_size,
                ),
                SocketSender(
                    self.stop_event,
                    self.send_audio_chunks_queue,
                    host=socket_sender_kwargs.send_host,
                    port=socket_sender_kwargs.send_port,
                ),
            ]

        self.vad = VADHandler(
            self.stop_event,
            queue_in=self.recv_audio_chunks_queue,
            queue_out=self.spoken_prompt_queue,
            setup_args=(self.should_listen,),
            setup_kwargs=vars(vad_handler_kwargs),
        )
        if module_kwargs.stt == "whisper":
            from s2s.STT.whisper_stt_handler import WhisperSTTHandler

            self.stt = WhisperSTTHandler(
                self.stop_event,
                queue_in=self.spoken_prompt_queue,
                queue_out=self.text_prompt_queue,
                setup_kwargs=vars(whisper_stt_handler_kwargs),
            )
        elif module_kwargs.stt == "whisper-mlx":
            from s2s.STT.lightning_whisper_mlx_handler import LightningWhisperSTTHandler

            self.stt = LightningWhisperSTTHandler(
                self.stop_event,
                queue_in=self.spoken_prompt_queue,
                queue_out=self.text_prompt_queue,
                setup_kwargs=vars(whisper_stt_handler_kwargs),
            )
        else:
            raise ValueError("The STT should be either whisper or whisper-mlx")
        if module_kwargs.llm == "transformers":
            from s2s.LLM.language_model import LanguageModelHandler

            self.lm = LanguageModelHandler(
                self.stop_event,
                queue_in=self.text_prompt_queue,
                queue_out=self.lm_response_queue,
                setup_kwargs=vars(language_model_handler_kwargs),
            )
        elif module_kwargs.llm == "mlx-lm":
            from s2s.LLM.mlx_language_model import MLXLanguageModelHandler

            self.lm = MLXLanguageModelHandler(
                self.stop_event,
                queue_in=self.text_prompt_queue,
                queue_out=self.lm_response_queue,
                setup_kwargs=vars(mlx_language_model_handler_kwargs),
            )
        else:
            raise ValueError("The LLM should be either transformers or mlx-lm")
        if module_kwargs.tts == "parler":
            from s2s.TTS.parler_handler import ParlerTTSHandler

            self.tts = ParlerTTSHandler(
                self.stop_event,
                queue_in=self.lm_response_queue,
                queue_out=self.send_audio_chunks_queue,
                setup_args=(self.should_listen,),
                setup_kwargs=vars(parler_tts_handler_kwargs),
            )

        elif module_kwargs.tts == "melo":
            try:
                from s2s.TTS.melo_handler import MeloTTSHandler
            except RuntimeError as e:
                logger.error(
                    "Error importing MeloTTSHandler. You might need to run: python -m unidic download"
                )
                raise e
            self.tts = MeloTTSHandler(
                self.stop_event,
                queue_in=self.lm_response_queue,
                queue_out=self.send_audio_chunks_queue,
                setup_args=(self.should_listen,),
                setup_kwargs=vars(melo_tts_handler_kwargs),
            )
        else:
            raise ValueError("The TTS should be either parler or melo")

    def run(self):
        # 4. Run the pipeline
        try:
            pipeline_manager = ThreadManager([*self.comms_handlers, self.vad, self.stt, self.lm, self.tts])
            pipeline_manager.start()

        except KeyboardInterrupt:
            pipeline_manager.stop()
