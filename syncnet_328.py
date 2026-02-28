import os
import cv2
import argparse
from typing import Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset as TorchDataset, DataLoader


class Dataset(TorchDataset):
    def __init__(self, dataset_dir: str, mode: str):
        self.dataset_dir = dataset_dir
        self.mode = mode

        img_dir = os.path.join(dataset_dir, "full_body_img")
        lms_dir = os.path.join(dataset_dir, "landmarks")

        num_frames = len(os.listdir(img_dir))
        self.img_path_list = [os.path.join(img_dir, f"{i}.jpg") for i in range(num_frames)]
        self.lms_path_list = [os.path.join(lms_dir, f"{i}.lms") for i in range(num_frames)]

        audio_map = {
            "wenet": "aud_wenet.npy",
            "hubert": "aud_hu.npy",
            "ave": "aud_ave.npy",
        }
        if mode not in audio_map:
            raise ValueError(f"Unsupported mode: {mode}. Expected one of {list(audio_map.keys())}")

        audio_feats_path = os.path.join(dataset_dir, audio_map[mode])
        self.audio_feats = np.load(audio_feats_path).astype(np.float32, copy=False)

        self.crop_boxes = [self._read_crop_box(p) for p in self.lms_path_list]
        self.y = torch.ones(1, dtype=torch.float32)

    def __len__(self) -> int:
        return self.audio_feats.shape[0] - 1

    @staticmethod
    def _read_crop_box(lms_path: str) -> Tuple[int, int, int, int]:
        lms = np.loadtxt(lms_path, dtype=np.float32).astype(np.int32)

        xmin = int(lms[1][0])
        ymin = int(lms[52][1])
        xmax = int(lms[31][0])

        width = xmax - xmin
        ymax = ymin + width

        return xmin, ymin, xmax, ymax

    @staticmethod
    def _safe_crop(img: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
        h, w = img.shape[:2]
        xmin, ymin, xmax, ymax = box

        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)

        if xmax <= xmin or ymax <= ymin:
            side = min(h, w)
            x0 = (w - side) // 2
            y0 = (h - side) // 2
            return img[y0:y0 + side, x0:x0 + side]

        return img[ymin:ymax, xmin:xmax]

    def get_audio_features(self, index: int) -> torch.Tensor:
        left = index - 8
        right = index + 8

        feat_shape = self.audio_feats.shape[1:]
        out = np.zeros((16, *feat_shape), dtype=np.float32)

        src_left = max(0, left)
        src_right = min(self.audio_feats.shape[0], right)

        dst_left = src_left - left
        dst_right = dst_left + (src_right - src_left)

        out[dst_left:dst_right] = self.audio_feats[src_left:src_right]
        return torch.from_numpy(out)

    def process_img(self, img: np.ndarray, crop_box: Tuple[int, int, int, int]) -> torch.Tensor:
        crop_img = self._safe_crop(img, crop_box)
        crop_img = cv2.resize(crop_img, (328, 328), interpolation=cv2.INTER_AREA)

        img_real = crop_img[4:324, 4:324]
        img_real = np.ascontiguousarray(img_real.transpose(2, 0, 1), dtype=np.float32) / 255.0
        return torch.from_numpy(img_real)

    def __getitem__(self, idx: int):
        img = cv2.imread(self.img_path_list[idx], cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {self.img_path_list[idx]}")

        img_real_T = self.process_img(img, self.crop_boxes[idx])
        audio_feat = self.get_audio_features(idx)

        if self.mode == "ave":
            audio_feat = audio_feat.reshape(32, 16, 16)
        else:
            audio_feat = audio_feat.reshape(32, 32, 32)

        return img_real_T, audio_feat, self.y


class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.residual = residual

    def forward(self, x):
        out = self.bn(self.conv(x))
        if self.residual:
            out = out + x
        return self.act(out)


class nonorm_Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size, stride, padding)
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        return self.act(self.conv(x))


class Conv2dTranspose(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.deconv(x)))


class SyncNet_color(nn.Module):
    def __init__(self, mode):
        super().__init__()

        self.face_encoder = nn.Sequential(
            Conv2d(3, 32, kernel_size=7, stride=1, padding=3),
            Conv2d(32, 32, kernel_size=5, stride=2, padding=1),

            Conv2d(32, 64, kernel_size=5, stride=2, padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
        )

        p1 = 128
        p2 = (1, 2)
        if mode == "hubert":
            p1 = 32
            p2 = (2, 2)
        elif mode == "ave":
            p1 = 32
            p2 = 1

        self.audio_encoder = nn.Sequential(
            Conv2d(p1, 128, kernel_size=3, stride=1, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=p2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 256, kernel_size=3, stride=2, padding=2),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, face_sequences, audio_sequences):
        face_embedding = self.face_encoder(face_sequences)
        audio_embedding = self.audio_encoder(audio_sequences)

        audio_embedding = audio_embedding.flatten(1)
        face_embedding = face_embedding.flatten(1)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)

        return audio_embedding, face_embedding


logloss = nn.BCELoss()

def cosine_loss(a, v, y):
    d = F.cosine_similarity(a, v)
    return logloss(d.unsqueeze(1), y)


def train(save_dir: str, dataset_dir: str, mode: str, epochs: int = 100):
    os.makedirs(save_dir, exist_ok=True)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    train_dataset = Dataset(dataset_dir, mode=mode)

    cpu_count = os.cpu_count() or 4
    num_workers = min(8, cpu_count)

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SyncNet_color(mode).to(device)
    model.train()

    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3
    )

    best_loss = float("inf")
    best_ckpt_path = os.path.join(save_dir, "best.pth")

    for epoch in range(epochs):
        running_loss = 0.0
        num_batches = 0

        for imgT, audioT, y in train_loader:
            imgT = imgT.to(device, non_blocking=True)
            audioT = audioT.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            audio_embedding, face_embedding = model(imgT, audioT)
            loss = cosine_loss(audio_embedding, face_embedding, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        epoch_loss = running_loss / max(1, num_batches)
        print(f"Epoch {epoch + 1:03d} | loss = {epoch_loss:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"New best model saved to: {best_ckpt_path} (loss={best_loss:.6f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="test", type=str)
    parser.add_argument("--dataset_dir", default="./dataset/May", type=str)
    parser.add_argument("--asr", default="ave", type=str)
    parser.add_argument("--epochs", default=100, type=int)
    opt = parser.parse_args()

    train(opt.save_dir, opt.dataset_dir, opt.asr, opt.epochs)
