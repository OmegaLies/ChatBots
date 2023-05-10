# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
# pip install streamkit
# pip install streamlit-chat
# pip install transformers
# pip install openai
import streamlit as st
import ruGPT_3 as gpt
import sqlite3
from streamlit_chat import message
import openai
import requests

try:
    conn = sqlite3.connect('Messages.db')
    c = conn.cursor()
except sqlite3.Error as e:
    print(e)

openai_api_key = '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'
single_request_model = 'text-davinci-003'
context_chat_model = 'gpt-3.5-turbo'


def main():
    st.title("Чат-бот для генерации текста")
    sber, openAI = st.tabs(['SberDevices', 'OpenAI'])
    with sber:
        models = ["sberbank-ai/rugpt3small_based_on_gpt2",
                  "sberbank-ai/rugpt3medium_based_on_gpt2"]
        model_choice = st.selectbox("Выберите модель", models)
        fill_chat(model_choice)
        text = st.text_input("Ваше сообщение:", key='sber_input')
        length = st.slider("Выберите макс длину", min_value=0, max_value=300, value=50, step=10)
        if st.button("Сгенерировать", key='generate_sber'):
            add_message(text=text, role="user", chat=model_choice)
            tok, model = gpt.load_tokenizer_and_model(model_choice)
            generated = gpt.generate(model, tok, text, num_beams=10, max_length=length)
            st.markdown(generated[0])
            add_message(text=generated[0], role="assistant", chat=model_choice)
    with openAI:
        model_choice_openai = st.selectbox("Выберите модель", [single_request_model, context_chat_model])
        if model_choice_openai == single_request_model:
            fill_chat(single_request_model)
            max_tokens = st.slider("Выберите максимальное количество токенов", min_value=0, max_value=516,
                                   value=250, step=50)
        elif model_choice_openai == context_chat_model:
            fill_chat(context_chat_model)
        input_text = st.text_input("Ваше сообщение:", key='open_ai_input')
        if st.button("Сгенерировать", key='generate_openai'):
            add_message(text=input_text, role='user', chat=model_choice_openai)
            if model_choice_openai == single_request_model:
                res = single_request(input_text)
            elif model_choice_openai == context_chat_model:
                res = context_request()
            st._rerun()


def add_message(role, text, chat):
    c.execute("INSERT INTO Messages (text, role, chat) VALUES (?,?,?)", (text, role, chat,))
    conn.commit()


def single_request(text):
    openai.api_key = openai_api_key
    response = openai.Completion.create(
        engine=single_request_model,
        prompt=text,
        max_tokens=516,
        temperature=0.5
    )
    result = response['choices'][0]['text']
    add_message(text=result, role='assistant', chat=single_request_model)
    return result


def context_request():
    db_messages = c.execute("SELECT * FROM Messages WHERE chat = ?", (context_chat_model,)).fetchall()
    messages_history = []
    for db_message in db_messages:
        messages_history.append({
            "role": db_message[2],
            "content": db_message[1]
        })

    openai.api_key = openai_api_key
    response = openai.ChatCompletion.create(
        model=context_chat_model,
        messages=messages_history
    )

    result = response['choices'][0]['message']['content']
    add_message(text=result, role='assistant', chat=context_chat_model)
    return result


def fill_chat(chat_model):
    messages = c.execute("SELECT * FROM Messages WHERE chat = ?", (chat_model,)).fetchall()
    for mess in messages:
        if mess[2] == 'user':
            message(message=mess[1], is_user=True, key=f'user: {mess[0]}')
        else:
            message(message=mess[1], is_user=False, key=f'assistant: {mess[0]}')


if __name__ == '__main__':
    main()
