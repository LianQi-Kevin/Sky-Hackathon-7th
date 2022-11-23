# import os
# import numpy as np
import gradio as gr
# import cv2
# from PIL import Image
from ASR_metrics import utils as metrics
import argparse
from audio_utils import audio_save

print(gr.__version__)
exit()

# global variable
LoadModelType = False
HaveAudio = False
ASR_Model = None


# ASR components function
class ASR_components:
    # when upload audio, change global var
    @staticmethod
    def update_audio():
        global LoadModelType, HaveAudio
        HaveAudio = True
        # print("Set global var 'HaveAudio' True")

        # update visible type
        if LoadModelType and HaveAudio:
            return gr.update(visible=True), gr.update(visible=True)
        else:
            return gr.update(visible=False), gr.update(visible=False)

    # when clear audio, change global var
    @staticmethod
    def clear_audio():
        global HaveAudio
        HaveAudio = False
        print("Set global var 'HaveAudio' False")
        return gr.update(visible=False), gr.update(visible=False)

    # switch "microphone" and "upload"
    @staticmethod
    def audio_type_change(state):
        if state:
            return gr.update(source="microphone")
        else:
            return gr.update(source="upload")

    # load ASR model
    @staticmethod
    def ASR_model_load_click(model_path="/home/nvidia/7th_ASR/7th_asr_model.nemo"):
        global LoadModelType, HaveAudio, ASR_Model

        # load model
        if ASR_Model is None:
            try:
                import nemo.collections.asr as nemo_asr
                ASR_Model = nemo_asr.models.EncDecCTCModel.restore_from(model_path)
            except:
                ASR_Model = None

        # set global var
        if ASR_Model is not None:
            LoadModelType = True
            print("Successful load ASR model")
            # print("Set global var 'LoadModelType' True")

        LoadModelType = True

        # update visible type
        if LoadModelType and HaveAudio:
            return gr.update(visible=True), gr.update(visible=True), gr.update(value="Successful Load")
        else:
            return gr.update(visible=False), gr.update(visible=False), gr.update(value="Load Model")
            # return gr.update(visible=False), gr.update(visible=False), gr.update(value="Successful Load")

    # free ASR memory used
    @staticmethod
    def ASR_model_kill():
        global ASR_Model, LoadModelType
        ASR_Model = None
        LoadModelType = False
        return gr.update(visible=False), gr.update(visible=False), gr.update(value="Load Model")

    @staticmethod
    # get ASR result
    def detection_click(temporary_path, accuracy_word):
        global ASR_Model
        wav_filepath = audio_save(temporary_path)
        try:
            asr_result = ASR_Model.transcribe(paths2audio_files=[wav_filepath])
            # s1 = request.form.get('defaultText')
            s2 = " ".join(asr_result)  # 识别结果
            result = {
                "asr_result": asr_result,
                "word_error_rate": metrics.calculate_cer(accuracy_word, s2),
                "word_accuracy_rate": 1 - metrics.calculate_cer(accuracy_word, s2)
            }
            return result["asr_result"], str(result["word_error_rate"]), str(result["word_accuracy_rate"])
        except:
            return "DETECT ERROR", "DETECT ERROR", "DETECT ERROR"


# CV components function
# class CV_components:
#     @staticmethod
#
#     def load_engine:


# Web page
def UI(args):
    # gr.Column()   垂直      | gr.ROW()  水平
    with gr.Blocks(title="Sky Hackathon 7th 无辑", css="WebUI/utils/WebUI.css") as demo:
        gr.Markdown(
            """
            <h2 align="center" face="STCAIYUN">Sky Hackathon 7th —— 无辑</h2>
            
            ---
            """)
        with gr.Row():
            # ASR Block
            with gr.Column():
                gr.Markdown("""<h3 align="center">ASR</h3>""")
                with gr.Box():
                    with gr.Row():
                        ASR_loadModel = gr.Button(value="Load Model", elem_id="ASR_load_model")
                        ASR_killModel = gr.Button(value="free memory", elem_id="ASR_kill_model")
                    correct_answer = gr.Textbox(value="请检测出纸箱", lines=1, max_lines=1,
                                                show_label=True, interactive=True, label="Type correct word")
                    with gr.Group():
                        audio_components = gr.Audio(label="Audio to be detected", source="upload",
                                                    type="filepath", interactive=True, show_label=True)
                        micro_type = gr.Checkbox(label="Use microphone", value=False)

                ASR_startDetection = gr.Button(value="Start Detection", variant="primary", elem_id="ASR_detection",
                                               visible=False)
                with gr.Group(visible=False) as ASR_result_group:
                    ASR_result = gr.Textbox(label="Detection result", interactive=False)
                    ASR_CER = gr.Textbox(label="Word error rate", interactive=False)
                    ASR_accuracy = gr.Textbox(label="Word accuracy rate", interactive=False)

            # CV Block
            with gr.Column():
                gr.Markdown("""<h3 align="center">CV</h3>""")

                with gr.Box():
                    with gr.Row():
                        CV_loadModel = gr.Button(value="Load Model", elem_id="CV_load_model")
                        CV_killModel = gr.Button(value="free memory", elem_id="CV_kill_model")
                    category_list = gr.CheckboxGroup(choices=args.category_list, label="Category to be checked out",
                                                     show_label=True, type="value", interactive=True,
                                                     value=[args.category_list[0], args.category_list[1]])
                    detect_mode = gr.Radio(choices=["Single Img", "mAP", "Video"], label="Detection Mode",
                                           show_label=True, type="value", value="Single Img", interactive=True)
                CV_startDetection = gr.Button(value="Start Detection", variant="primary", elem_id="CV_detection")

                with gr.Box(visible=True) as single_img_mode:
                    with gr.Column():
                        input_img = gr.Image()

                CV_json_result = gr.Json(label="Detection Result", show_label=True, visible=True)

        # ASR action
        micro_type.change(ASR_components.audio_type_change, inputs=[micro_type], outputs=[audio_components])
        audio_components.change(ASR_components.update_audio, [], [ASR_startDetection, ASR_result_group])
        audio_components.clear(ASR_components.clear_audio, [], [ASR_startDetection, ASR_result_group])
        ASR_killModel.click(ASR_components.ASR_model_kill, inputs=[],
                            outputs=[ASR_startDetection, ASR_result_group, ASR_loadModel])
        ASR_loadModel.click(ASR_components.ASR_model_load_click, inputs=[],
                            outputs=[ASR_startDetection, ASR_result_group, ASR_loadModel])
        ASR_startDetection.click(ASR_components.detection_click, inputs=[audio_components, correct_answer],
                                 outputs=[ASR_result, ASR_CER, ASR_accuracy])

        # CV Action

        # style
        # correct_answer.style(container=True)

    demo.launch(inbrowser=False, debug=True, show_error=True, server_port=5000, server_name="0.0.0.0")


if __name__ == '__main__':
    # argparse
    parser = argparse.ArgumentParser("messages")
    args = parser.parse_args()

    args.category_list = ["bottle", "carton", "peel"]

    UI(args)
