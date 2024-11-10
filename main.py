from tkinter import *
from tkinter import ttk

import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import time

start_time = time.time()
MODEL_NAME = "IlyaGusev/saiga_mistral_7b_lora"

# Загружаем модель
config = PeftConfig.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
	config.base_model_name_or_path,
        load_in_8bit=False,
	torch_dtype=torch.float16,
	device_map="auto",
        offload_folder="offload/"
).to(torch.device("cuda"))
model = PeftModel.from_pretrained(
	model,
	MODEL_NAME,
	torch_dtype=torch.float16,
        offload_folder="offload/"
).to(torch.device("cuda"))
model.eval()

# Определяем токенайзер
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
generation_config = GenerationConfig.from_pretrained(MODEL_NAME)

# Функция для обработки запросов
def generate(model, tokenizer, prompt, generation_config):
	data = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
	data = {k: v.to(model.device) for k, v in data.items()}
	output_ids = model.generate(
    	**data,
    	generation_config=generation_config
	)[0]
	output_ids = output_ids[len(data["input_ids"][0]):]
	output = tokenizer.decode(output_ids, skip_special_tokens=True)
	##field2.config(text=output.strip())
	field2.delete(1.0,END)
	field2.insert(INSERT, output.strip())

def perefraZ():
        PROMT_TEMPLATE = '<s>system\nТы — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им.</s><s>user\n{inp}</s><s>bot\n'
        global field1
        inp = field1.get("1.0", "end")
        prompt = 'Перефразируй следующий текст более простыми словами: ' + PROMT_TEMPLATE.format(inp=inp)
        # Отправляем запрос в llm
        generate(model, tokenizer, prompt, generation_config)

def sokr():
        PROMT_TEMPLATE = '<s>system\nТы — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им.</s><s>user\n{inp}</s><s>bot\n'
        global field1
        global percent
        inp = field1.get("1.0", "end")
        prompt = 'Сократи следующий текст на ' + str(percent.get()) + '%: ' + PROMT_TEMPLATE.format(inp=inp)
        # Отправляем запрос в llm
        generate(model, tokenizer, prompt, generation_config)

# Интерфейс
root = Tk()

root['bg'] = '#fafafa'
root.title = 'Сократитель текста'
root.geometry('700x400')
root.resizable(width=False, height=False)

canvas = Canvas(root, height=700, width=400)
canvas.pack()

frame = Frame(root, bg='white')
frame.place(relx=0, rely=0, relwidth=1, relheight=1)
frame2 = Frame(root, bg='red')
frame2.place(relx=0.32, rely=0.1, relwidth=0.0245, relheight=0.73)
frame3 = Frame(root, bg='blue')
frame3.place(relx=0.655, rely=0.1, relwidth=0.025, relheight=0.73)

title = Label(frame, text='Сокращатель', bg='white', font=40)
title.pack()

field1 = Text(frame, width=25, height=18, bg='yellow')
field1.place(x='20',y='40')

scroll = Scrollbar(frame2, command = field1.yview)
scroll.pack(side=RIGHT, fill=Y)
field1.config(yscrollcommand=scroll.set)


field2 = Text(frame, width=25, height=18, bg='yellow')
field2.place(x='475',y='40')

scroll2 = Scrollbar(frame3, command = field2.yview)
scroll2.pack(side=LEFT, fill=Y)
field2.config(yscrollcommand=scroll2.set)

l1 = Label(frame, text="Сократить на %", bg='white', font=40)
l1.pack(side=TOP, pady=10)
percent = ttk.Entry(frame)
percent.pack(side=TOP, pady=10)
btn1 = Button(frame, text='--------->',width=20,height=2, bg='gray', command=sokr)
btn1.pack(side=TOP, pady=20)
l2 = Label(frame, text="Перефразировать", bg='white', font=40)
l2.pack(side=TOP, pady=50)
btn2 = Button(frame, text='--------->',width=20,height=2, bg='gray', command=perefraZ)
btn2.pack(side=TOP, pady=20)

btn3 = Button(frame, text='3',width=5,height=2, bg='gray')
btn3.place(x='20',y='340')

btn4 = Button(frame, text='4',width=5,height=2, bg='gray')
btn4.place(x='70',y='340')

btn5 = Button(frame, text='5',width=5,height=2, bg='gray')
btn5.place(x='120',y='340')

btn6 = Button(frame, text='6',width=5,height=2, bg='gray')
btn6.place(x='170',y='340')

btn7 = Button(frame, text='7',width=5,height=2, bg='gray')
btn7.place(x='220',y='340')

btn8 = Button(frame, text='8',width=5,height=2, bg='gray')
btn8.place(x='475',y='340')

btn9 = Button(frame, text='9',width=5,height=2, bg='gray')
btn9.place(x='550',y='340')

btn10 = Button(frame, text='10',width=5,height=2, bg='gray')
btn10.place(x='625',y='340')

root.mainloop()
