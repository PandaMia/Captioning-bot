import numpy as np
import torch
from torch.distributions import Categorical
from PIL import Image


class CaptionGenerator:

    def __init__(self, token2idx, idx2token, inception, rnn_model, device):
        self.token2idx = token2idx
        self.idx2token = idx2token
        self.inception = inception
        self.rnn_model = rnn_model
        self.device = device

    def load_image(self, image_path):
        image = Image.open(image_path).resize((299, 299))
        image = np.array(image).astype('float32') / 255.
        return image

    def generate(self,
                 image_path,
                 T=1.,
                 max_len=50,
                 greedy=True):

        image = self.load_image(image_path)

        assert isinstance(image, np.ndarray) and np.max(image) <= 1 \
               and np.min(image) >= 0 and image.shape[-1] == 3

        caption_prefix = [self.token2idx['<SOS>']]
        self.rnn_model.eval()
        with torch.no_grad():
            image = torch.tensor(image.transpose([2, 0, 1]), dtype=torch.float32).to(self.device)
            vectors_8x8, vectors_neck, logits = self.inception(image[None])

            # слово за словом генерируем описание картинки
            for _ in range(max_len):
                caption_prefix_tensor = torch.tensor(caption_prefix + [self.token2idx['<PAD>']]).unsqueeze(0).to(self.device)
                logits = self.rnn_model(vectors_neck, caption_prefix_tensor)
                probs = torch.softmax(logits[:, -1] / T, dim=-1)  # получаем распределение для последнего токена
                if greedy:
                    sampled = torch.argmax(probs, dim=-1).item()  # получаем новый токен жадно
                else:
                    sampled = Categorical(probs).sample().item()  # или через сэмплирование из распределения
                caption_prefix.append(sampled)                    # добавляем новое слово в префикс
                if sampled == self.token2idx['<EOS>']:
                    break

        caption = [self.idx2token[str(token)] for token in caption_prefix]
        caption = ' '.join(caption[1:-1])

        return caption

if __name__ == '__main__':
    import os
    import json
    from beheaded_inception3 import beheaded_inception_v3
    from model import CaptionNet

    data_path = 'source/data'
    model_path = 'source/model'

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

    caption = generator.generate(os.path.join(data_path, 'sonya.jpg'), greedy=False)
    print(caption)