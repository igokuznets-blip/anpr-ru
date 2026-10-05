"""Компактная CRNN (CNN + BiLSTM + CTC) для распознавания номеров РФ."""
import torch
import torch.nn as nn

from plate_rules import OCR_ALPHABET

IMG_H, IMG_W = 32, 128


class CRNN(nn.Module):
    def __init__(self, n_classes: int = len(OCR_ALPHABET) + 1):  # +1 = blank (индекс 0)
        super().__init__()

        def block(i, o, pool):
            return [nn.Conv2d(i, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=True)] + \
                   ([nn.MaxPool2d(pool)] if pool else [])

        self.cnn = nn.Sequential(
            *block(1, 64, (2, 2)),        # 16x64
            *block(64, 128, (2, 2)),      # 8x32
            *block(128, 256, None),
            *block(256, 256, (2, 1)),     # 4x32
            *block(256, 512, None),
            *block(512, 512, (4, 1)),     # 1x32
        )
        self.rnn = nn.LSTM(512, 256, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):                 # x: B,1,32,128
        f = self.cnn(x).squeeze(2)        # B,512,32
        f, _ = self.rnn(f.permute(0, 2, 1))
        return self.fc(f)                 # B,T=32,C


def greedy_decode(logits: torch.Tensor) -> list[str]:
    """CTC greedy: убираем повторы и blank."""
    out = []
    for seq in logits.argmax(-1).tolist():
        s, prev = [], 0
        for k in seq:
            if k != prev and k != 0:
                s.append(OCR_ALPHABET[k - 1])
            prev = k
        out.append("".join(s))
    return out


def preprocess(img_gray):
    """numpy gray (H,W) uint8 -> tensor 1,1,32,128."""
    import cv2
    x = cv2.resize(img_gray, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(x).float().div(255.).sub(0.5).div(0.5)[None, None]
