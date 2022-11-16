import os
import numpy as np
import gradio as gr
import cv2
from PIL import Image

from utils.audio_utils import audio_save, load_ASR_model, get_ASR_result


# global variable
LoadModelType = False
HaveAudio = False
ASR_Model = None


# when upload audio, change global var
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
def clear_audio():
    global HaveAudio
    HaveAudio = False
    print("Set global var 'HaveAudio' False")
    return gr.update(visible=False), gr.update(visible=False)


# switch "microphone" and "upload"
def audio_type_change(state):
    if state:
        return gr.update(source="microphone")
    else:
        return gr.update(source="upload")


# load ASR model
def ASR_model_load_click():
    global LoadModelType, HaveAudio, ASR_Model

    # load model ***
    if ASR_Model is None:
        ASR_Model = load_ASR_model(model_path="/home/nvidia/7th_ASR/7th_asr_model.nemo")

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
def ASR_model_kill():
    global ASR_Model, LoadModelType
    ASR_Model = None
    LoadModelType = False
    return gr.update(visible=False), gr.update(visible=False), gr.update(value="Load Model")


# get ASR result
def detection_click(temporary_path, accuracy_word):
    global ASR_Model
    wav_filepath = audio_save(temporary_path)
    result = get_ASR_result(ASR_Model, accuracy_word, wav_filepath=wav_filepath)
    if result is None:
        return "DETECT ERROR", "DETECT ERROR", "DETECT ERROR"
    else:
        return result["asr_result"], str(result["word_error_rate"]), str(result["word_accuracy_rate"])


# Web page
def UI():
    # gr.Column()   垂直      | gr.ROW()  水平
    with gr.Blocks(title="Sky Hackathon 7th 无辑", css="utils/WebUI.css") as demo:
        page_title = gr.Markdown(
            """
            <h2 align="center" face="STCAIYUN">Sky Hackathon 7th —— 无辑</h2>
            
            ---
            """)
        with gr.Row():
            with gr.Column() as ASR_Block:
                gr.Markdown("""<h3 align="center">ASR</h3>""")
                with gr.Row():
                    ASR_loadModel = gr.Button(value="Load Model", elem_id="ASR_load_model")
                    ASR_killModel = gr.Button(value="free memory", elem_id="ASR_kill_model")
                correct_answer = gr.Textbox(value="请检测出纸箱", lines=1, max_lines=1,
                                            show_label=True, interactive=True, label="请输入正确答案")
                with gr.Group():
                    audio_components = gr.Audio(label="Audio to be detected", source="upload",
                                                type="filepath", interactive=True, show_label=True)
                    micro_type = gr.Checkbox(label="Use microphone", value=False)

                ASR_startDetection = gr.Button(value="开始检测", variant="primary", elem_id="ASR_detection", visible=False)
                with gr.Group(visible=False) as ASR_result_group:
                    ASR_result = gr.Textbox(label="Detection result", interactive=False)
                    ASR_CER = gr.Textbox(label="Word error rate", interactive=False)
                    ASR_accuracy = gr.Textbox(label="Word accuracy rate", interactive=False)

                # ASR_log = gr.Textbox(interactive=False, visible=True, label="Debug Information")

            with gr.Column() as CV_Block:
                gr.Markdown("""<h3 align="center">CV</h3>""")

        # ASR action
        micro_type.change(audio_type_change, inputs=[micro_type], outputs=[audio_components])
        audio_components.change(update_audio, [], [ASR_startDetection, ASR_result_group])
        audio_components.clear(clear_audio, [], [ASR_startDetection, ASR_result_group])

        ASR_killModel.click(ASR_model_kill, inputs=[], outputs=[ASR_startDetection, ASR_result_group, ASR_loadModel])
        ASR_loadModel.click(ASR_model_load_click, inputs=[], outputs=[ASR_startDetection, ASR_result_group, ASR_loadModel])
        ASR_startDetection.click(detection_click, inputs=[audio_components, correct_answer],
                                 outputs=[ASR_result, ASR_CER, ASR_accuracy])

        # style
        # correct_answer.style(container=True)

    demo.launch(inbrowser=False, debug=True, show_error=True, server_port=5000, server_name="0.0.0.0")


if __name__ == '__main__':
    UI()