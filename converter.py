"""
PETSCII conversion engine.
Uses CuPy (GPU) if CUDA is available, otherwise Numba JIT parallel CPU.
Character matching is done in RGB color space, not grayscale.
"""
import numpy as np
import cv2
from petscii_data import get_font_bitmaps, get_palette_rgb

try:
    import cupy as cp
    _t = cp.array([[1.0, 2.0]])
    _t @ _t.T
    del _t
    _asnumpy = cp.asnumpy
    _USE_GPU = True
    print("\033[92mGPU acceleration enabled (CuPy)\033[0m")
except Exception:
    cp = np
    _asnumpy = lambda x: x
    _USE_GPU = False
    print("\033[93m! Running on CPU (Numba JIT parallel)\033[0m")

from numba import njit, prange


@njit(parallel=True, cache=True)
def _match_pixels_to_palette(hsv_pixels, pal_hsv, grey_penalty):
    M = hsv_pixels.shape[0]
    P = pal_hsv.shape[0]
    out = np.empty(M, dtype=np.int32)
    for i in prange(M):
        h = hsv_pixels[i, 0]
        s = hsv_pixels[i, 1]
        v = hsv_pixels[i, 2]
        sat_w    = (s / 255.0) * 4.0 + 0.5
        skin_hue = max(0.0, 1.0 - h / 20.0)
        skin_sat = max(0.0, min(1.0, (s - 60.0) / 120.0))
        skin_val = max(0.0, 1.0 - abs(v - 140.0) / 100.0)
        skin_w   = skin_hue * skin_sat * skin_val
        best_dist = 1e18
        best_idx  = 0
        for j in range(P):
            dh = abs(h - pal_hsv[j, 0])
            if dh > 90.0:
                dh = 180.0 - dh
            ds = abs(s - pal_hsv[j, 1])
            dv = abs(v - pal_hsv[j, 2])
            dist = sat_w * dh * dh + 2.0 * ds * ds + 0.4 * dv * dv
            dist += (v / 255.0) * grey_penalty[j]
            if j == 9:
                dist -= skin_w * 6000.0
            if dist < best_dist:
                best_dist = dist
                best_idx  = j
        out[i] = best_idx
    return out


@njit(parallel=True, cache=True)
def _match_cells_color(cells_rgb, pix_pal_idx, flat, pal_rgb, is_text, n, N, P):
    out_char = np.empty(n, dtype=np.int32)
    out_fg   = np.empty(n, dtype=np.int32)
    out_bg   = np.empty(n, dtype=np.int32)

    for i in prange(n):
        votes_fg = np.zeros(P, dtype=np.float32)
        votes_bg = np.zeros(P, dtype=np.float32)
        med = np.float32(0.0)
        for k in range(64):
            med += cells_rgb[i, k, 0] + cells_rgb[i, k, 1] + cells_rgb[i, k, 2]
        med /= np.float32(192.0)
        for k in range(64):
            b = cells_rgb[i, k, 0] + cells_rgb[i, k, 1] + cells_rgb[i, k, 2]
            j = pix_pal_idx[i, k]
            if b >= med:
                votes_fg[j] += 1.0
            else:
                votes_bg[j] += 1.0

        best_fg = 0
        best_bg = 0
        for j in range(1, P):
            if votes_fg[j] > votes_fg[best_fg]:
                best_fg = j
            if votes_bg[j] > votes_bg[best_bg]:
                best_bg = j

        fg_r = pal_rgb[best_fg, 0]; fg_g = pal_rgb[best_fg, 1]; fg_b = pal_rgb[best_fg, 2]
        bg_r = pal_rgb[best_bg, 0]; bg_g = pal_rgb[best_bg, 1]; bg_b = pal_rgb[best_bg, 2]

        # Text chars get 30% error discount
        TEXT_BONUS = np.float32(0.70)

        best_err  = 1e18
        best_char = 0
        for c in range(N):
            err = np.float32(0.0)
            for k in range(64):
                if flat[c, k] > 0.5:
                    dr = cells_rgb[i, k, 0] - fg_r
                    dg = cells_rgb[i, k, 1] - fg_g
                    db = cells_rgb[i, k, 2] - fg_b
                else:
                    dr = cells_rgb[i, k, 0] - bg_r
                    dg = cells_rgb[i, k, 1] - bg_g
                    db = cells_rgb[i, k, 2] - bg_b
                err += dr*dr + dg*dg + db*db
            if is_text[c] > 0.5:
                err *= TEXT_BONUS
            if err < best_err:
                best_err  = err
                best_char = c

        out_char[i] = best_char
        out_fg[i]   = best_fg
        out_bg[i]   = best_bg

    return out_char, out_fg, out_bg


class PetsciiConverter:
    def __init__(self, char_indices: list[int] | None = None, rgb_mode: bool = False):
        self.font = get_font_bitmaps()
        self.palette = get_palette_rgb()
        self.rgb_mode = rgb_mode

        if char_indices is None:
            char_indices = list(range(0, 128))
        self.char_indices = np.array(char_indices, dtype=np.int32)

        self._is_text    = np.array([1.0 if i < 64 else 0.0 for i in char_indices], dtype=np.float32)
        self._flat_np    = self.font[self.char_indices].reshape(len(self.char_indices), 64).astype(np.float32)
        self._flat       = cp.array(self._flat_np)
        self._flat_inv   = 1.0 - self._flat

        self._reinit_palette()
        self._warmup()

    def _reinit_palette(self):
        pal_hsv = cv2.cvtColor(self.palette.reshape(1, -1, 3), cv2.COLOR_RGB2HSV).astype(np.float32).reshape(-1, 3)
        self._pal_hsv_np      = pal_hsv
        self._pal_f_np        = self.palette.astype(np.float32)
        self._grey_penalty_np = np.zeros(len(self.palette), dtype=np.float32)
        if len(self.palette) == 16:
            self._grey_penalty_np[[11, 12, 15]] = 8000.0
            self._grey_penalty_np[0] = 5000.0
        self._pal_f        = cp.array(self._pal_f_np)
        self._pal_hsv      = cp.array(pal_hsv)
        self._grey_penalty = cp.array(self._grey_penalty_np)

    def _warmup(self):
        dummy_hsv = np.zeros((1, 3),     dtype=np.float32)
        dummy_rgb = np.zeros((1, 64, 3), dtype=np.float32)
        dummy_idx = np.zeros((1, 64),    dtype=np.int32)
        N = len(self.char_indices)
        P = len(self.palette)
        _match_pixels_to_palette(dummy_hsv, self._pal_hsv_np, self._grey_penalty_np)
        _match_cells_color(dummy_rgb, dummy_idx, self._flat_np, self._pal_f_np, self._is_text, 1, N, P)

    def _pixels_to_palette_idx(self, pixels_rgb_u8: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(pixels_rgb_u8.reshape(1, -1, 3), cv2.COLOR_RGB2HSV).astype(np.float32).reshape(-1, 3)
        return _match_pixels_to_palette(hsv, self._pal_hsv_np, self._grey_penalty_np)

    def convert_frame(self, frame_bgr: np.ndarray, cols: int, rows: int) -> np.ndarray:
        small    = cv2.resize(frame_bgr, (cols * 8, rows * 8), interpolation=cv2.INTER_AREA)
        cells_np = small.reshape(rows, 8, cols, 8, 3).transpose(0, 2, 1, 3, 4)
        n        = rows * cols
        cells_rgb_np = cells_np[:, :, :, :, ::-1].reshape(n, 64, 3).astype(np.float32)

        if self.rgb_mode:
            cells_cp   = cp.array(cells_np)
            cells_gray = cells_cp.mean(axis=4).reshape(n, 64).astype(cp.float32) / 255.0
            lo = cells_gray.min(axis=1, keepdims=True)
            hi = cells_gray.max(axis=1, keepdims=True)
            cell_bin = (cells_gray > (lo + hi) / 2.0).astype(cp.float32)
            scores_n = cell_bin @ self._flat.T + (1.0 - cell_bin) @ self._flat_inv.T
            scores_i = cell_bin @ self._flat_inv.T + (1.0 - cell_bin) @ self._flat.T
            best_n   = cp.argmax(scores_n, axis=1)
            best_i   = cp.argmax(scores_i, axis=1)
            use_inv  = scores_i[cp.arange(n), best_i] > scores_n[cp.arange(n), best_n]
            char_idx = _asnumpy(cp.where(use_inv, self.char_indices[best_i], self.char_indices[best_n]))
            bitmaps  = cp.array(self.font[char_idx].reshape(n, 64).astype(bool))
            bitmaps[use_inv] = ~bitmaps[use_inv]
            bitmaps_f = bitmaps.astype(cp.float32)
            cells_rgb_cp = cp.array(cells_rgb_np)
            w_fg = bitmaps_f[:, :, None]
            w_bg = (1.0 - bitmaps_f)[:, :, None]
            fg_colors = (cells_rgb_cp * w_fg).sum(axis=1) / w_fg.sum(axis=1).clip(1)
            bg_colors = (cells_rgb_cp * w_bg).sum(axis=1) / w_bg.sum(axis=1).clip(1)
        else:
            pixels_rgb_u8 = cells_rgb_np.reshape(n * 64, 3).clip(0, 255).astype(np.uint8)
            pix_pal_idx   = self._pixels_to_palette_idx(pixels_rgb_u8).reshape(n, 64)
            char_local, fg_idx_arr, bg_idx_arr = _match_cells_color(
                cells_rgb_np, pix_pal_idx, self._flat_np, self._pal_f_np, self._is_text,
                n, len(self.char_indices), len(self.palette)
            )
            char_idx  = self.char_indices[char_local]
            bitmaps   = self.font[char_idx].reshape(n, 64).astype(bool)
            fg_colors = cp.array(self._pal_f_np[fg_idx_arr])
            bg_colors = cp.array(self._pal_f_np[bg_idx_arr])

        fg_bgr = fg_colors[:, ::-1].astype(cp.uint8)
        bg_bgr = bg_colors[:, ::-1].astype(cp.uint8)

        bitmaps_cp = cp.array(bitmaps)
        mask = bitmaps_cp.reshape(n, 8, 8)[:, :, :, None]
        out  = (cp.where(mask, fg_bgr[:, None, None, :], bg_bgr[:, None, None, :])
                .reshape(rows, cols, 8, 8, 3)
                .transpose(0, 2, 1, 3, 4)
                .reshape(rows * 8, cols * 8, 3))

        return _asnumpy(out)
