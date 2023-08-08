import gradio as gr
import infer
import sys
import os
import argparse

current_path = sys.path[0]

modellist=["没有模型啊！"]
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--share", help="open the share function", dest="ifshare",type=bool, default="False")
parser.add_argument("-p", "--port", help="select the port,if you don't chose,it will open the share function.", dest="listenport" ,type=int, default="6006")
args = parser.parse_args()



def interface():
    with gr.Blocks(theme=gr.themes.Soft()) as application:
        gr.Markdown("diffsinger自动标注")
        with gr.Column():
            with gr.Row():
                pro = gr.Dropdown(choices=modellist,value=modellist[0],label="选择处理模型")
                fresh = gr.Button("刷新")
            with gr.Row():
                dev = gr.Dropdown(choices=["AUTO","GPU"],label="选择推理设备",value="AUTO")
            gr.Markdown("将音频上传至此，注意文件名不要过于复杂，尽量只包含英文，以防未知bug")
            with gr.Row():
                inputaudio = gr.Audio(type="filepath")
            with gr.Row():
                start=gr.Button("开始自动标注")
            with gr.Row():
                result = gr.Text()

                def model():
                    files = os.listdir(current_path)
                    for pt in files:  # 获取所有模型路
                        if ".pth" in pt:
                            modellist.append(pt)
                        else:
                            pass
                    print(modellist)
                    modellist.pop(0)
                    return pro.update(choices=modellist, value=modellist[0])
        fresh.click(model,[],[pro])
        start.click(infer.infer,[pro,dev,inputaudio],[result])
    application.queue(concurrency_count=511, max_size=1022).launch(
        share=args.ifshare,
        server_name="0.0.0.0",
        server_port=args.listenport,
        quiet=True,)
if __name__=="__main__":
    interface()
