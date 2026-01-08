import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import queue
import threading

from glados.core.engine import Glados


class FakeAudioIO:
    def __init__(self):
        self.stopped = False

    def stop_speaking(self):
        self.stopped = True


def test_submit_text_when_interruptible_sets_queue_and_flags():
    gl = Glados.__new__(Glados)  # Bypass __init__ to avoid starting threads

    gl.interruptible = True
    gl.currently_speaking_event = threading.Event()
    gl.currently_speaking_event.set()  # Simulate that assistant is speaking
    gl.processing_active_event = threading.Event()
    gl.audio_io = FakeAudioIO()
    gl.llm_queue = queue.Queue()

    gl.submit_text("Hello typed world")

    # Text should be queued
    assert gl.llm_queue.get_nowait() == "Hello typed world"
    # Processing event should be set
    assert gl.processing_active_event.is_set()
    # stop_speaking should have been called
    assert gl.audio_io.stopped is True


def test_submit_text_when_not_interruptible_ignores_input():
    gl = Glados.__new__(Glados)

    gl.interruptible = False
    gl.currently_speaking_event = threading.Event()
    gl.currently_speaking_event.set()  # Assistant is speaking
    gl.processing_active_event = threading.Event()
    gl.audio_io = FakeAudioIO()
    gl.llm_queue = queue.Queue()

    gl.submit_text("Should be ignored")

    assert gl.llm_queue.empty()
    assert not gl.processing_active_event.is_set()
    # stop_speaking should not have been called in non-interruptible mode
    assert gl.audio_io.stopped is False
