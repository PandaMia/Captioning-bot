import os
import json
import torch
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from generate_caption import CaptionGenerator
from beheaded_inception3 import beheaded_inception_v3
from model import CaptionNet

data_path = 'source/data'
model_path = 'source/model'


def start(update, context):
    update.message.reply_text('Отправь мне картинку и я сгенерирую к ней описание!')


def error(update, context):
    update.message.reply_text('Произошла ошибка!')


def get_caption(update, context):
    file = update.message.photo[-1].get_file()
    file_name = update.message.photo[-1].file_id + '.jpg'
    file_path = os.path.join(data_path, file_name)
    file.download(file_path)
    caption = generator.generate(file_path, greedy=False)
    os.remove(file_path)
    update.message.reply_text(caption)


def main():
    TOKEN = ""  # paste token for your bot
    updater = Updater(TOKEN, use_context=True)
    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler("start", start))              # handler for start command
    dispatcher.add_handler(MessageHandler(Filters.photo, get_caption))  # handle for get caption
    dispatcher.add_error_handler(error)                                 # handler for errors

    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    with open(os.path.join(model_path, 'token2idx.json'), 'r') as f:
        token2idx = json.load(f)
    with open(os.path.join(model_path, 'idx2token.json'), 'r') as f:
        idx2token = json.load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    inception = beheaded_inception_v3().train(False).to(device)
    rnn_model = CaptionNet(vocab_size=len(token2idx), num_lstm_layers=2, device=device).to(device)
    rnn_model.load_state_dict(torch.load(os.path.join(model_path, 'model_weights.pth'),
                                         map_location=torch.device('cpu')))
    generator = CaptionGenerator(token2idx, idx2token, inception, rnn_model, device)

    main()
