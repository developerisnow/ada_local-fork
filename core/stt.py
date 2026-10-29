"""
Speech-to-text with a local, transcription-based wake word.
Uses RealTimeSTT for audio segmentation and Whisper transcription.
"""

import threading
import re
import time
from typing import Optional, Callable
from config import (
    WAKE_WORD, REALTIMESTT_MODEL, WAKE_WORD_SENSITIVITY,
    USE_PORCUPINE_WAKE_WORD, STT_RECORD_TIMEOUT,
    GRAY, RESET, CYAN, YELLOW, GREEN
)


class STTListener:
    """
    Real-time STT listener with wake word detection using RealTimeSTT.
    Uses transcription gating by default; Porcupine is explicitly opt-in.
    """
    
    def __init__(self, wake_word_callback: Callable, speech_callback: Callable):
        self.wake_word_callback = wake_word_callback
        self.speech_callback = speech_callback
        self.running = False
        self.listening_thread = None
        
        # RealTimeSTT recorder
        self.recorder = None
        self.initialized = False
        self._wake_until = 0.0
        self._recording_started_at = None
        
        print(f"{CYAN}[STT] Initializing RealTimeSTT listener...{RESET}")
        print(f"{CYAN}[STT] Wake word: '{WAKE_WORD}'{RESET}")
        print(f"{CYAN}[STT] Porcupine enabled: {USE_PORCUPINE_WAKE_WORD}{RESET}")
    
    def initialize(self, *, use_microphone: bool = True) -> bool:
        """Initialize ASR; external PCM input can replace the microphone for tests."""
        try:
            from RealtimeSTT import AudioToTextRecorder
            import torch
            
            print(f"{CYAN}[STT] Loading RealTimeSTT...{RESET}")
            
            # Check CUDA availability
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                cuda_device = torch.cuda.current_device()
                cuda_name = torch.cuda.get_device_name(cuda_device)
                print(f"{GREEN}[STT] ✓ CUDA is available (Device: {cuda_name}){RESET}")
            else:
                print(f"{YELLOW}[STT] ⚠ CUDA is not available, will use CPU{RESET}")
            
            device = "cuda" if cuda_available else "cpu"
            print(f"{CYAN}[STT] Initializing AudioToTextRecorder on {device}...{RESET}")
            self.recorder = AudioToTextRecorder(
                model=REALTIMESTT_MODEL,
                language="",
                device=device,
                compute_type="default" if cuda_available else "int8",
                use_microphone=use_microphone,
                spinner=False,
                wakeword_backend="pvporcupine" if USE_PORCUPINE_WAKE_WORD else "",
                wake_words=WAKE_WORD if USE_PORCUPINE_WAKE_WORD else "",
                wake_words_sensitivity=WAKE_WORD_SENSITIVITY,
                on_wakeword_detected=self._on_wakeword_detected,
                on_recording_start=self._on_recording_start,
            )
            
            # Verify device after initialization
            if hasattr(self.recorder, 'model') and hasattr(self.recorder.model, 'device'):
                actual_device = str(self.recorder.model.device)
                print(f"{GREEN}[STT] ✓ Model device: {actual_device}{RESET}")
            elif hasattr(self.recorder, '_device'):
                print(f"{GREEN}[STT] ✓ Recorder device: {self.recorder._device}{RESET}")
            
            self.initialized = True
            print(f"{CYAN}[STT] ✓ RealTimeSTT initialized successfully (model: {REALTIMESTT_MODEL}, wake word: '{WAKE_WORD}'){RESET}")
            return True
        except ImportError:
            print(f"{GRAY}[STT] ✗ RealTimeSTT not installed. Install with: pip install realtimestt{RESET}")
            return False
        except Exception as e:
            print(f"{GRAY}[STT] ✗ RealTimeSTT initialization error: {e}{RESET}")
            import traceback
            traceback.print_exc()
            return False
    
    def _on_wakeword_detected(self):
        """Callback when wake word is detected."""
        print(f"\n{CYAN}[STT] 👂 Wake word '{WAKE_WORD}' detected! Listening...{RESET}")
        # Notify callback if set
        if self.wake_word_callback:
            self.wake_word_callback()

    def start(self):
        """Start listening."""
        if not self.initialized:
            print(f"{YELLOW}[STT] Not initialized. Call initialize() first.{RESET}")
            return False
        
        if self.running:
            print(f"{YELLOW}[STT] Already running.{RESET}")
            return True
        
        self.running = True
        print(f"{CYAN}[STT] Starting RealTimeSTT listener...{RESET}")
        
        # Start RealTimeSTT in a background thread
        try:
            self.listening_thread = threading.Thread(
                target=self._run_listener,
                daemon=True
            )
            self.listening_thread.start()
            print(f"{CYAN}[STT] ✓ Listener started{RESET}")
            return True
        except Exception as e:
            print(f"{GRAY}[STT] Failed to start listener: {e}{RESET}")
            self.running = False
            return False
    
    def _run_listener(self):
        """Main listening loop using RealTimeSTT's text() method."""
        try:
            print(f"{GRAY}[STT] 🔄 Starting transcription loop...{RESET}")
            while self.running:
                if not self.recorder:
                    break
                
                print(f"{GRAY}[STT] ⏳ Waiting for wake word '{WAKE_WORD}'...{RESET}")
                
                # Transcribe one utterance, then enforce the configured wake boundary.
                transcription_start = time.time()
                text = self.recorder.text()
                transcription_time = time.time() - transcription_start
                
                print(f"{CYAN}[STT] ✓ Transcription completed in {transcription_time:.2f}s{RESET}")
                
                if text and text.strip():
                    self._handle_transcription(text, self._recording_started_at)
                self._recording_started_at = None
                
        except Exception as e:
            print(f"{GRAY}[STT] Listener error: {e}{RESET}")
            import traceback
            traceback.print_exc()
            self.running = False

    def _on_recording_start(self):
        self._recording_started_at = time.monotonic()

    def _handle_transcription(self, text: str, started_at: Optional[float] = None):
        """Dispatch only a wake-word command or its bounded follow-up utterance."""
        now = time.monotonic()
        utterance_start = now if started_at is None else started_at
        match = re.search(r"\b" + re.escape(WAKE_WORD) + r"\b", text, re.IGNORECASE)
        if USE_PORCUPINE_WAKE_WORD:
            command = text[match.end():] if match else text
        elif match:
            self._on_wakeword_detected()
            self._wake_until = now + STT_RECORD_TIMEOUT
            command = text[match.end():]
        elif utterance_start < self._wake_until:
            command = text
        else:
            return
        command = command.strip(" \t\r\n,.:;!?")
        if command:
            self._wake_until = 0.0
            self.speech_callback(command)
    
    def stop(self):
        """Stop listening."""
        self.running = False
        if self.recorder:
            try:
                print(f"{CYAN}[STT] Shutting down recorder...{RESET}")
                self.recorder.shutdown()
            except Exception as e:
                print(f"{GRAY}[STT] Error stopping recorder: {e}{RESET}")
        if self.listening_thread:
            self.listening_thread.join(timeout=2.0)
        print(f"{CYAN}[STT] Listener stopped{RESET}")
