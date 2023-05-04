import numpy as np
import os
import sys
import tensorflow as tf
from os.path import dirname
from ovos_plugin_manager.templates.stt import STT
from ovos_plugin_manager.templates.transformers import AudioTransformer
from ovos_utils.log import LOG
from ovos_utils.xdg_utils import xdg_data_home


class SpeechFlowLangClassifier(AudioTransformer):
    def __init__(self, config=None):
        config = config or {}
        super().__init__("ovos-audio-transformer-plugin-speechflow-langdetect", 10, config)
        self.token_list = [
            "chinese",
            "english",
            "french",
            "german",
            "indonesian",
            "italian",
            "japanese",
            "korean",
            "portuguese",
            "russian",
            "spanish",
            "turkish",
            "vietnamese",
            "other"
        ]
        model = self.config.get("model") or f"{dirname(__file__)}/pretrained"
        self.engine = tf.saved_model.load(model)

    @staticmethod
    def audiochunk2array(audio_data):
        # Convert buffer to float32 using NumPy
        audio_as_np_int16 = np.frombuffer(audio_data, dtype=np.int16)
        audio_as_np_float32 = audio_as_np_int16.astype(np.float32)

        # Normalise float32 array so that values are between -1.0 and +1.0
        max_int16 = 2 ** 15
        data = audio_as_np_float32 / max_int16
        return data

    # plugin api
    def transform(self, audio_data):
        # Download Thai language sample from Omniglot and cvert to suitable form
        signal = self.audiochunk2array(audio_data)
        output, prob = self.engine.predict_pb(signal)

        lang = self.token_list[output.numpy()]
        prob = prob.numpy() * 100

        LOG.info(f"Detected speech language '{lang}' with probability {prob}")
        return audio_data, {"stt_lang": lang.split(":")[0], "lang_probability": prob}


if __name__ == "__main__":
    from speech_recognition import Recognizer, AudioFile, AudioData

    jfk = "/home/miro/PycharmProjects/ovos-stt-plugin-fasterwhisper/jfk.wav"
    with AudioFile(jfk) as source:
        audio = Recognizer().record(source)

    s = SpeechFlowLangClassifier()
    s.transform(audio.get_wav_data())
    # {'stt_lang': 'en', 'lang_probability': 0.8076384663581848}
