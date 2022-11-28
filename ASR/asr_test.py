import nemo.collections.asr as nemo_asr
from ASR_metrics import utils as metrics


def nemo_asr_y():
    try_model_1 = nemo_asr.models.EncDecCTCModel.restore_from("models/7th_Hackathon_citrinet.nemo")
    asr_result = try_model_1.transcribe(paths2audio_files=["audio_save/all6.wav"])
    s1 = "请检测出果皮纸箱和瓶子"  # 指定正确答案
    s2 = " ".join(asr_result)  # 识别结果
    print("字错率: {}".format(metrics.calculate_cer(s1, s2)))  # 计算字错率cer
    print("准确率: {}".format(1 - metrics.calculate_cer(s1, s2)))  # 计算准确率accuracy

    print("asr_result: {}".format(s2))


def main():
    nemo_asr_y()


if __name__ == '__main__':
    main()


