import os
from librosa import resample as lib_resample
from soundfile import write as sf_write
from soundfile import read as sf_read
from datetime import datetime


# Resample audio to 44100 and save it.
def audio_save(temporary_path, save_path="./audio_save"):
    # create save path
    os.makedirs(save_path, exist_ok=True)

    data, sample_rate = sf_read(temporary_path, dtype='float32')
    data = lib_resample(data, orig_sr=sample_rate, target_sr=44100)

    filepath = os.path.join(save_path, "{}.wav".format(datetime.now().strftime("%Y%m%d_%H%M%S")))
    sf_write(file=filepath, data=data, samplerate=44100, subtype='PCM_24')
    return filepath


# Load ASR model.
def load_ASR_model(model_path="/home/nvidia/7th_ASR/7th_asr_model.nemo"):
    try:
        import nemo.collections.asr as nemo_asr
        # print('Start Loading Nemo')
        ASR_model = nemo_asr.models.EncDecCTCModel.restore_from(model_path)
        # print('Done loading Nemo')
        return ASR_model
    except:
        return None


# Get ASR model result.
def get_ASR_result(ASR_model, accuracy_word, wav_filepath):
    try:
        from ASR_metrics import utils as metrics
        asr_result = ASR_model.transcribe(paths2audio_files=[wav_filepath])
        # s1 = request.form.get('defaultText')
        s2 = " ".join(asr_result)  # 识别结果
        result = {
            "asr_result": asr_result,
            "word_error_rate": metrics.calculate_cer(accuracy_word, s2),
            "word_accuracy_rate": 1 - metrics.calculate_cer(accuracy_word, s2)
        }
        return result
    except:
        return None