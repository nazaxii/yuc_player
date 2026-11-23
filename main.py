import array
import configparser
import json
import math
import os
import random
import shutil
import subprocess
import sys

os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
os.environ["QT_SCALE_FACTOR"] = "1"
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"
os.environ["QT_SCALE_FACTOR_ROUNDING_POLICY"] = "PassThrough"

from PyQt6.QtCore import (
    QPoint,
    QRect,
    QStandardPaths,
    Qt,
    QThread,
    QTimer,
    QUrl,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QFontMetrics,
    QImage,
    QPainter,
    QPixmap,
)
from PyQt6.QtMultimedia import QAudioOutput, QMediaDevices, QMediaPlayer
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QLabel,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

try:
  import numpy as np
except ImportError:
  np = None

try:
  from mutagen import File as MutagenFile
except ImportError:
  MutagenFile = None


AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".ogg",
    ".flac",
    ".m4a",
    ".opus",
    ".aac",
    ".wma",
}

DEFAULT_THEME_COLORS = {
    "primary_color": "#f58db1",
    "selected_color": "#ffffff",
    "secondary_color": "#888888",
    "search_bar_text": "#ffffff",
}


def make_smooth_font(
    family: str, point_size: int, bold: bool = False
) -> QFont:
  font = QFont(family, point_size)
  font.setBold(bold)
  font.setStyleStrategy(
      QFont.StyleStrategy.PreferAntialias | QFont.StyleStrategy.PreferQuality
  )
  return font


def parse_qcolor(value: str | None, fallback_hex: str) -> QColor:
  if isinstance(value, str):
    cleaned = value.strip()
    c = QColor(cleaned)
    if c.isValid():
      return c
  return QColor(fallback_hex)


def load_theme_config(theme_dir: str, folder_name: str) -> dict:
  cfg_path = os.path.join(theme_dir, "theme.cfg")
  parser = configparser.ConfigParser()
  if os.path.isfile(cfg_path):
    try:
      parser.read(cfg_path, encoding="utf-8")
    except Exception:
      pass

  colors_sec = parser["Colors"] if parser.has_section("Colors") else {}
  desc_sec = parser["Description"] if parser.has_section("Description") else {}

  theme_name = str(desc_sec.get("name", folder_name)).strip() or folder_name
  theme_desc = str(desc_sec.get("description", "")).strip()

  primary_hex = str(
      colors_sec.get("primary_color", DEFAULT_THEME_COLORS["primary_color"])
  ).strip()
  selected_hex = str(
      colors_sec.get("selected_color", DEFAULT_THEME_COLORS["selected_color"])
  ).strip()
  secondary_hex = str(
      colors_sec.get("secondary_color", DEFAULT_THEME_COLORS["secondary_color"])
  ).strip()
  search_hex = str(
      colors_sec.get(
          "search_bar_text", DEFAULT_THEME_COLORS["search_bar_text"]
      )
  ).strip()

  return {
      "folder_name": folder_name,
      "dir": theme_dir,
      "name": theme_name,
      "description": theme_desc,
      "primary_color": parse_qcolor(
          primary_hex, DEFAULT_THEME_COLORS["primary_color"]
      ),
      "selected_color": parse_qcolor(
          selected_hex, DEFAULT_THEME_COLORS["selected_color"]
      ),
      "secondary_color": parse_qcolor(
          secondary_hex, DEFAULT_THEME_COLORS["secondary_color"]
      ),
      "search_bar_text": parse_qcolor(
          search_hex, DEFAULT_THEME_COLORS["search_bar_text"]
      ),
  }


def format_duration(seconds: float | None) -> str:
  if seconds is None or seconds <= 0:
    return "--:--"
  total = int(round(seconds))
  mins, secs = divmod(total, 60)
  hours, mins = divmod(mins, 60)
  if hours > 0:
    return f"{hours}:{mins:02d}:{secs:02d}"
  return f"{mins:02d}:{secs:02d}"


def format_timer_mm_ss(ms: int) -> str:
  if ms <= 0:
    return "00:00"
  total_sec = ms // 1000
  mins = min(99, total_sec // 60)
  secs = total_sec % 60
  return f"{mins:02d}:{secs:02d}"


def format_file_size(size_bytes: int) -> str:
  if size_bytes < 1024:
    return f"{size_bytes} B"
  if size_bytes < 1024 * 1024:
    return f"{size_bytes / 1024:.1f} KB"
  return f"{size_bytes / (1024 * 1024):.1f} MB"


def read_track_metadata(file_path: str) -> dict:
  base = os.path.basename(file_path)
  title, ext = os.path.splitext(base)
  fmt = ext.lstrip(".").upper() if ext else "AUDIO"

  try:
    size_bytes = os.path.getsize(file_path)
  except OSError:
    size_bytes = 0

  duration_sec: float | None = None
  bitrate_kbps: int | None = None

  if MutagenFile is not None and os.path.exists(file_path):
    try:
      audio = MutagenFile(file_path)
      if audio is not None and getattr(audio, "info", None) is not None:
        length = getattr(audio.info, "length", None)
        if length and length > 0:
          duration_sec = float(length)
        bitrate = getattr(audio.info, "bitrate", None)
        if bitrate and bitrate > 0:
          bitrate_kbps = int(round(bitrate / 1000))
    except Exception:
      pass

  if (
      bitrate_kbps is None
      and duration_sec
      and duration_sec > 0
      and size_bytes > 0
  ):
    bitrate_kbps = int(round((size_bytes * 8) / duration_sec / 1000))

  return {
      "path": file_path,
      "title": title,
      "format": fmt,
      "size_bytes": size_bytes,
      "size_str": format_file_size(size_bytes),
      "duration_sec": duration_sec,
      "duration_str": format_duration(duration_sec),
      "bitrate_kbps": bitrate_kbps,
  }


def _compute_7_bands_pure_python(samples: list[float], max_rows: int) -> list[float]:
  n = min(len(samples), 128)
  if n < 16:
    return [0.0] * 7
  chunk = samples[-n:]
  windowed = [
      chunk[i] * (0.5 - 0.5 * math.cos(2.0 * math.pi * i / (n - 1)))
      for i in range(n)
  ]
  half = n // 2
  band_edges = [
      1,
      max(2, half // 16),
      max(3, half // 8),
      max(5, half // 5),
      max(8, half // 3),
      max(14, half // 2),
      max(22, int(half * 0.78)),
      half,
  ]
  bands: list[float] = []
  for b in range(7):
    k_start = band_edges[b]
    k_end = max(k_start + 1, band_edges[b + 1])
    power_sum = 0.0
    count = 0
    for k in range(k_start, min(half, k_end)):
      re = 0.0
      im = 0.0
      angle_step = -2.0 * math.pi * k / n
      for idx_s, val in enumerate(windowed):
        ang = angle_step * idx_s
        re += val * math.cos(ang)
        im += val * math.sin(ang)
      mag = math.hypot(re, im) / (n * 0.5)
      power_sum += mag * mag
      count += 1
    rms = math.sqrt(power_sum / max(1, count))
    level = min(float(max_rows), (rms * 18.0) ** 0.75 * (max_rows * 0.65))
    bands.append(max(0.0, level))
  return bands


class AudioAnalyzerThread(QThread):
  """Читає PCM через ffmpeg та віддає 7 частотних смуг (0..max_rows)."""

  bands_ready = pyqtSignal(list)

  SAMPLE_RATE = 11025
  CHUNK_SAMPLES = 512

  def __init__(self, max_rows: int = 7, parent=None):
    super().__init__(parent)
    self.max_rows = max_rows
    self._file_path: str | None = None
    self._start_sec: float = 0.0
    self._paused: bool = False
    self._stop_requested: bool = False
    self._proc: subprocess.Popen | None = None
    self._has_ffmpeg = shutil.which("ffmpeg") is not None
    self._tick = 0.0

    if np is not None:
      self._hann = np.hanning(self.CHUNK_SAMPLES).astype(np.float32)
      freqs = np.fft.rfftfreq(self.CHUNK_SAMPLES, d=1.0 / self.SAMPLE_RATE)
      edges = [40.0, 120.0, 280.0, 600.0, 1300.0, 2800.0, 5200.0]
      self._band_masks: list[object] = []
      for i in range(7):
        f_low = edges[i]
        f_high = edges[i + 1] if i + 1 < len(edges) else self.SAMPLE_RATE * 0.5
        mask = (freqs >= f_low) & (freqs < f_high)
        if not np.any(mask):
          idx = min(len(freqs) - 1, max(1, (i + 1) * 8))
          mask = np.zeros_like(freqs, dtype=bool)
          mask[idx] = True
        self._band_masks.append(mask)
    else:
      self._hann = None
      self._band_masks = []

  def configure(self, file_path: str, start_sec: float = 0.0):
    self._file_path = file_path
    self._start_sec = max(0.0, float(start_sec))
    self._paused = False
    self._terminate_proc()

  def set_paused(self, paused: bool):
    self._paused = bool(paused)

  def stop_stream(self):
    self._stop_requested = True
    self._file_path = None
    self._terminate_proc()

  def _terminate_proc(self):
    proc = self._proc
    self._proc = None
    if proc is not None:
      try:
        proc.kill()
      except Exception:
        pass
      try:
        proc.wait(timeout=0.2)
      except Exception:
        pass

  def _fallback_bands(self) -> list[float]:
    self._tick += 0.25
    result: list[float] = []
    for i in range(7):
      w = 0.5 + 0.5 * math.sin(self._tick * (1.1 + i * 0.22) + i * 0.9)
      jitter = random.uniform(0.25, 1.0)
      val = w * jitter * float(self.max_rows)
      result.append(max(0.0, min(float(self.max_rows), val)))
    return result

  def _process_chunk(self, raw_bytes: bytes) -> list[float]:
    if np is not None:
      arr = np.frombuffer(raw_bytes, dtype="<i2").astype(np.float32) / 32768.0
      if arr.size < self.CHUNK_SAMPLES:
        padded = np.zeros(self.CHUNK_SAMPLES, dtype=np.float32)
        padded[: arr.size] = arr
        arr = padded
      windowed = arr * self._hann
      spectrum = np.abs(np.fft.rfft(windowed)) / (self.CHUNK_SAMPLES * 0.5)

      levels: list[float] = []
      db_min = -50.0
      db_max = -10.0
      db_span = db_max - db_min

      for i, mask in enumerate(self._band_masks):
        band_vals = spectrum[mask]
        if band_vals.size:
          rms = float(np.sqrt(np.mean(band_vals * band_vals)))
          peak = float(np.max(band_vals))
          amp = rms * 0.8 + peak * 0.2
        else:
          amp = 0.0

        amp *= 1.0 + i * 0.12
        db = 20.0 * math.log10(max(amp, 1e-6))
        norm = max(0.0, min(1.0, (db - db_min) / db_span))
        curved = norm**1.25
        levels.append(curved * float(self.max_rows))
      return levels

    samples_int = array.array("h")
    samples_int.frombytes(raw_bytes)
    if sys.byteorder != "little":
      samples_int.byteswap()
    floats = [s / 32768.0 for s in samples_int[:128]]
    return _compute_7_bands_pure_python(floats, self.max_rows)

  def run(self):
    bytes_per_sample = 2
    chunk_bytes = self.CHUNK_SAMPLES * bytes_per_sample

    while not self._stop_requested:
      path = self._file_path
      if not path or not os.path.exists(path):
        self.msleep(45)
        continue

      if self._paused:
        self.msleep(45)
        continue

      if not self._has_ffmpeg:
        while (
            not self._stop_requested
            and self._file_path == path
            and not self._paused
        ):
          self.bands_ready.emit(self._fallback_bands())
          self.msleep(50)
        continue

      cmd = ["ffmpeg", "-nostdin", "-loglevel", "quiet"]
      if self._start_sec > 0.05:
        cmd.extend(["-ss", f"{self._start_sec:.2f}"])
      cmd.extend([
          "-i",
          path,
          "-f",
          "s16le",
          "-acodec",
          "pcm_s16le",
          "-ac",
          "1",
          "-ar",
          str(self.SAMPLE_RATE),
          "-",
      ])

      try:
        creationflags = (
            subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        )
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=chunk_bytes * 4,
            creationflags=creationflags,
        )
      except Exception:
        self._has_ffmpeg = False
        continue

      self._proc = proc
      stdout = proc.stdout
      if stdout is None:
        self._terminate_proc()
        self.msleep(45)
        continue

      try:
        while not self._stop_requested and self._file_path == path:
          if self._paused:
            self.msleep(35)
            continue

          data = stdout.read(chunk_bytes)
          if not data or len(data) < chunk_bytes:
            break

          levels = self._process_chunk(data)
          self.bands_ready.emit(levels)
          self.msleep(25)
      except Exception:
        pass
      finally:
        if self._proc is proc:
          self._terminate_proc()

      if self._file_path == path and not self._stop_requested:
        self.msleep(100)


class SpriteButton(QWidget):
  clicked = pyqtSignal()

  def __init__(
      self,
      normal_px: QPixmap,
      pressed_px: QPixmap,
      x: int,
      y: int,
      w: int,
      h: int,
      parent=None,
      active_px: QPixmap | None = None,
  ):
    super().__init__(parent)
    self.normal_px = normal_px
    self.pressed_px = pressed_px
    self.active_px = active_px
    self.is_pressed = False
    self.is_active = False

    self.setGeometry(x, y, w, h)
    self.setFixedSize(w, h)
    self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
    self.setCursor(Qt.CursorShape.PointingHandCursor)

  def set_sprites(
      self,
      normal_px: QPixmap,
      pressed_px: QPixmap,
      active_px: QPixmap | None = None,
  ):
    self.normal_px = normal_px
    self.pressed_px = pressed_px
    if active_px is not None:
      self.active_px = active_px
    self.update()

  def set_active(self, active: bool):
    self.is_active = active
    self.update()

  def paintEvent(self, event):
    painter = QPainter(self)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
    painter.resetTransform()

    if self.is_pressed and not self.pressed_px.isNull():
      px = self.pressed_px
    elif (
        self.is_active
        and self.active_px is not None
        and not self.active_px.isNull()
    ):
      px = self.active_px
    else:
      px = self.normal_px

    if not px.isNull():
      painter.drawPixmap(0, 0, px)

  def mousePressEvent(self, event):
    if event.button() == Qt.MouseButton.LeftButton:
      self.is_pressed = True
      self.update()
      event.accept()

  def mouseReleaseEvent(self, event):
    if event.button() == Qt.MouseButton.LeftButton:
      was_pressed = self.is_pressed
      self.is_pressed = False
      self.update()
      if was_pressed and self.rect().contains(event.pos()):
        self.clicked.emit()
      event.accept()


class VolumeThumb(QWidget):
  drag_started = pyqtSignal()
  position_changed = pyqtSignal(int)

  def __init__(
      self, thumb_px: QPixmap, x: int, y: int, w: int, h: int, parent=None
  ):
    super().__init__(parent)
    self.thumb_px = thumb_px
    self.is_dragging = False
    self.grab_offset_x = 0

    self.setGeometry(x, y, w, h)
    self.setFixedSize(w, h)
    self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
    self.setCursor(Qt.CursorShape.PointingHandCursor)

  def set_thumb_pixmap(self, thumb_px: QPixmap):
    self.thumb_px = thumb_px
    self.update()

  def paintEvent(self, event):
    painter = QPainter(self)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
    painter.resetTransform()

    if not self.thumb_px.isNull():
      painter.drawPixmap(0, 0, self.thumb_px)

  def mousePressEvent(self, event):
    if event.button() == Qt.MouseButton.LeftButton:
      self.is_dragging = True
      self.grab_offset_x = int(event.position().x())
      self.drag_started.emit()
      event.accept()

  def mouseMoveEvent(self, event):
    if self.is_dragging and (event.buttons() & Qt.MouseButton.LeftButton):
      new_x = int(self.x() + event.position().x() - self.grab_offset_x)
      self.position_changed.emit(new_x)
      event.accept()

  def mouseReleaseEvent(self, event):
    if event.button() == Qt.MouseButton.LeftButton and self.is_dragging:
      self.is_dragging = False
      event.accept()


class ProgressThumb(QWidget):
  drag_started = pyqtSignal()
  position_changed = pyqtSignal(int)
  drag_ended = pyqtSignal(int)

  def __init__(
      self, thumb_px: QPixmap, x: int, y: int, w: int, h: int, parent=None
  ):
    super().__init__(parent)
    self.thumb_px = thumb_px
    self.is_dragging = False
    self.grab_offset_x = 0

    self.setGeometry(x, y, w, h)
    self.setFixedSize(w, h)
    self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
    self.setCursor(Qt.CursorShape.PointingHandCursor)

  def set_thumb_pixmap(self, thumb_px: QPixmap):
    self.thumb_px = thumb_px
    self.update()

  def paintEvent(self, event):
    painter = QPainter(self)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
    painter.resetTransform()

    if not self.thumb_px.isNull():
      painter.drawPixmap(0, 0, self.thumb_px)

  def mousePressEvent(self, event):
    if event.button() == Qt.MouseButton.LeftButton:
      self.is_dragging = True
      self.grab_offset_x = int(event.position().x())
      self.drag_started.emit()
      event.accept()

  def mouseMoveEvent(self, event):
    if self.is_dragging and (event.buttons() & Qt.MouseButton.LeftButton):
      new_x = int(self.x() + event.position().x() - self.grab_offset_x)
      self.position_changed.emit(new_x)
      event.accept()

  def mouseReleaseEvent(self, event):
    if event.button() == Qt.MouseButton.LeftButton and self.is_dragging:
      self.is_dragging = False
      self.drag_ended.emit(self.x())
      event.accept()


class SearchBarWidget(QWidget):
  text_changed = pyqtSignal(str)
  SEARCH_FONT_SIZE = 11

  def __init__(
      self,
      normal_px: QPixmap,
      active_px: QPixmap,
      x: int,
      y: int,
      w: int,
      h: int,
      parent=None,
      font_family: str = "Sans Serif",
      text_color: QColor | None = None,
  ):
    super().__init__(parent)
    self.normal_px = normal_px
    self.active_px = active_px
    self.query_text = ""
    self.is_focused = False
    self.cursor_visible = True
    self.text_color = text_color or QColor("#ffffff")

    self.setGeometry(x, y, w, h)
    self.setFixedSize(w, h)
    self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
    self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
    self.setCursor(Qt.CursorShape.IBeamCursor)

    self.text_font = make_smooth_font(font_family, self.SEARCH_FONT_SIZE)

    self.blink_timer = QTimer(self)
    self.blink_timer.setInterval(500)
    self.blink_timer.timeout.connect(self._toggle_cursor)

  def apply_theme(
      self, normal_px: QPixmap, active_px: QPixmap, text_color: QColor
  ):
    self.normal_px = normal_px
    self.active_px = active_px
    self.text_color = text_color
    self.update()

  def _toggle_cursor(self):
    if self.is_focused:
      self.cursor_visible = not self.cursor_visible
      self.update()

  def focusInEvent(self, event):
    self.is_focused = True
    self.cursor_visible = True
    self.blink_timer.start()
    self.update()
    super().focusInEvent(event)

  def focusOutEvent(self, event):
    self.is_focused = False
    self.cursor_visible = False
    self.blink_timer.stop()
    self.update()
    super().focusOutEvent(event)

  def paintEvent(self, event):
    painter = QPainter(self)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    painter.resetTransform()

    bg = (
        self.active_px
        if (self.is_focused and not self.active_px.isNull())
        else self.normal_px
    )
    if not bg.isNull():
      painter.drawPixmap(0, 0, bg)

    painter.setFont(self.text_font)
    fm = QFontMetrics(self.text_font)
    pad_left = 8
    pad_right = 8
    avail_w = max(10, self.width() - pad_left - pad_right)
    text_rect = QRect(pad_left, 0, avail_w, self.height())

    if self.query_text:
      painter.setPen(self.text_color)
      display_str = fm.elidedText(
          self.query_text, Qt.TextElideMode.ElideLeft, avail_w
      )
      painter.drawText(
          text_rect,
          Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
          display_str,
      )
      if self.is_focused and self.cursor_visible:
        adv = fm.horizontalAdvance(display_str)
        cx = min(self.width() - pad_right - 1, pad_left + adv + 1)
        cy1 = 4
        cy2 = self.height() - 4
        painter.drawLine(cx, cy1, cx, cy2)
    else:
      if not self.is_focused:
        painter.setPen(QColor("#888888"))
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            "Search...",
        )
      elif self.cursor_visible:
        painter.setPen(self.text_color)
        painter.drawLine(pad_left, 4, pad_left, self.height() - 4)

  def keyPressEvent(self, event):
    key = event.key()
    if key == Qt.Key.Key_Escape:
      if self.query_text:
        self.query_text = ""
        self.text_changed.emit(self.query_text)
      self.clearFocus()
      self.update()
      event.accept()
      return

    if key == Qt.Key.Key_Backspace:
      if self.query_text:
        self.query_text = self.query_text[:-1]
        self.cursor_visible = True
        self.text_changed.emit(self.query_text)
        self.update()
      event.accept()
      return

    if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
      self.clearFocus()
      event.accept()
      return

    text = event.text()
    if (
        text
        and text.isprintable()
        and not (
            event.modifiers()
            & (
                Qt.KeyboardModifier.ControlModifier
                | Qt.KeyboardModifier.AltModifier
            )
        )
    ):
      self.query_text += text
      self.cursor_visible = True
      self.text_changed.emit(self.query_text)
      self.update()
      event.accept()
      return

    super().keyPressEvent(event)


class MarqueeLabel(QWidget):
  FONT_SIZE = 12
  GAP = 32

  def __init__(
      self,
      x: int,
      y: int,
      w: int,
      h: int,
      parent=None,
      font_family: str = "Sans Serif",
      text_color: QColor | None = None,
  ):
    super().__init__(parent)
    self.text = ""
    self.is_scrolling_active = False
    self.scroll_offset = 0.0
    self.text_color = text_color or QColor("#f58db1")

    self.setGeometry(x, y, w, h)
    self.setFixedSize(w, h)
    self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)

    self.font = make_smooth_font(font_family, self.FONT_SIZE)

    self.scroll_timer = QTimer(self)
    self.scroll_timer.setInterval(30)
    self.scroll_timer.timeout.connect(self._tick_scroll)

  def set_text_color(self, color: QColor):
    self.text_color = color
    self.update()

  def set_font_family(self, font_family: str):
    self.font = make_smooth_font(font_family, self.FONT_SIZE)
    self.update()

  def set_text(self, text: str):
    if text != self.text:
      self.text = text
      self.scroll_offset = 0.0
      self._sync_timer_state()
      self.update()

  def set_playing(self, playing: bool):
    self.is_scrolling_active = bool(playing)
    self._sync_timer_state()
    self.update()

  def _needs_scroll(self) -> bool:
    if not self.text:
      return False
    fm = QFontMetrics(self.font)
    return fm.horizontalAdvance(self.text) > self.width()

  def _sync_timer_state(self):
    if self.is_scrolling_active and self._needs_scroll():
      if not self.scroll_timer.isActive():
        self.scroll_timer.start()
    else:
      if self.scroll_timer.isActive():
        self.scroll_timer.stop()

  def _tick_scroll(self):
    if not self.text or not self._needs_scroll():
      self.scroll_offset = 0.0
      self.scroll_timer.stop()
      self.update()
      return

    fm = QFontMetrics(self.font)
    cycle_w = fm.horizontalAdvance(self.text) + self.GAP
    if cycle_w > 0:
      self.scroll_offset = (self.scroll_offset + 1.2) % cycle_w
    self.update()

  def paintEvent(self, event):
    if not self.text:
      return

    painter = QPainter(self)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    painter.resetTransform()
    painter.setClipRect(self.rect())
    painter.setFont(self.font)
    painter.setPen(self.text_color)

    fm = QFontMetrics(self.font)
    text_w = fm.horizontalAdvance(self.text)

    if text_w <= self.width() or not self.is_scrolling_active:
      elided = fm.elidedText(self.text, Qt.TextElideMode.ElideRight, self.width())
      painter.drawText(
          self.rect(),
          Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
          elided,
      )
      return

    offset_int = int(self.scroll_offset)
    cycle_w = text_w + self.GAP
    x1 = -offset_int
    while x1 < self.width():
      rect = QRect(x1, 0, text_w, self.height())
      painter.drawText(
          rect,
          Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
          self.text,
      )
      x1 += cycle_w


class PlaylistView(QWidget):
  track_activated = pyqtSignal(int)

  CARD_W = 538
  CARD_H = 47
  ICON_X = 8
  ICON_Y = 14
  ICON_SIZE = 19

  TITLE_FONT_SIZE = 12
  SUB_FONT_SIZE = 9

  def __init__(
      self,
      icon_px: QPixmap,
      x: int,
      y: int,
      w: int,
      h: int,
      parent=None,
      font_family: str | None = None,
      primary_color: QColor | None = None,
      selected_color: QColor | None = None,
      secondary_color: QColor | None = None,
  ):
    super().__init__(parent)
    self.icon_px = icon_px
    self.tracks: list[dict] = []
    self.filtered_indices: list[int] = []
    self.filter_query: str = ""
    self.selected_indices: set[int] = set()
    self.playing_index: int | None = None
    self.scroll_y: int = 0

    self.color_normal = primary_color or QColor("#f58db1")
    self.color_selected = selected_color or QColor("#ffffff")
    self.color_subtext = secondary_color or QColor("#888888")

    self.setGeometry(x, y, w, h)
    self.setFixedSize(w, h)
    self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
    self.setCursor(Qt.CursorShape.PointingHandCursor)

    resolved_family = font_family or "Sans Serif"

    self.title_font = make_smooth_font(resolved_family, self.TITLE_FONT_SIZE)
    self.sub_font = make_smooth_font(resolved_family, self.SUB_FONT_SIZE)

    self.rebuild_filter()

  def apply_theme(
      self,
      icon_px: QPixmap,
      primary_color: QColor,
      selected_color: QColor,
      secondary_color: QColor,
  ):
    self.icon_px = icon_px
    self.color_normal = primary_color
    self.color_selected = selected_color
    self.color_subtext = secondary_color
    self.update()

  def rebuild_filter(self):
    q = self.filter_query.strip().lower()
    if not q:
      self.filtered_indices = list(range(len(self.tracks)))
    else:
      self.filtered_indices = [
          i
          for i, t in enumerate(self.tracks)
          if q in t["title"].lower()
          or q in t["format"].lower()
          or q in os.path.basename(t["path"]).lower()
      ]
    self.clamp_scroll()
    self.update()

  def set_filter_query(self, query: str):
    self.filter_query = query
    self.rebuild_filter()

  def max_scroll(self) -> int:
    total_h = len(self.filtered_indices) * self.CARD_H
    return max(0, total_h - self.height())

  def clamp_scroll(self):
    self.scroll_y = max(0, min(self.max_scroll(), self.scroll_y))

  def ensure_visible_real_index(self, real_idx: int):
    if real_idx not in self.filtered_indices:
      return
    vis_pos = self.filtered_indices.index(real_idx)
    card_top = vis_pos * self.CARD_H
    card_bottom = card_top + self.CARD_H
    if card_top < self.scroll_y:
      self.scroll_y = card_top
    elif card_bottom > self.scroll_y + self.height():
      self.scroll_y = card_bottom - self.height()
    self.clamp_scroll()

  def _apply_ordered_tracks(self, ordered_tracks: list[dict]):
    playing_path = (
        self.tracks[self.playing_index]["path"]
        if self.playing_index is not None
        and 0 <= self.playing_index < len(self.tracks)
        else None
    )
    selected_paths = {
        self.tracks[idx]["path"]
        for idx in self.selected_indices
        if 0 <= idx < len(self.tracks)
    }

    self.tracks = ordered_tracks
    self.playing_index = None
    new_selected: set[int] = set()

    for new_idx, tr in enumerate(self.tracks):
      if (
          playing_path is not None
          and self.playing_index is None
          and tr["path"] == playing_path
      ):
        self.playing_index = new_idx
      if tr["path"] in selected_paths:
        new_selected.add(new_idx)

    if not new_selected and self.tracks:
      new_selected = {0}
    self.selected_indices = new_selected
    self.rebuild_filter()

  def sort_by_name(self):
    ordered = sorted(
        self.tracks,
        key=lambda t: (
            t["title"].lower(),
            t["format"].lower(),
            t["path"].lower(),
        ),
    )
    self._apply_ordered_tracks(ordered)

  def sort_by_format_and_name(self):
    format_order = {
        "MP3": 0,
        "OGG": 1,
        "M4A": 2,
        "OPUS": 3,
        "FLAC": 4,
        "WAV": 5,
        "AAC": 6,
    }
    ordered = sorted(
        self.tracks,
        key=lambda t: (
            format_order.get(t["format"].upper(), 99),
            t["format"].upper(),
            t["title"].lower(),
            t["path"].lower(),
        ),
    )
    self._apply_ordered_tracks(ordered)

  def add_tracks(self, file_paths: list[str], auto_sort_alpha: bool = True) -> int:
    if not file_paths:
      return 0
    added_count = 0
    for path in file_paths:
      if path:
        self.tracks.append(read_track_metadata(path))
        added_count += 1

    if added_count > 0:
      if not self.selected_indices and self.tracks:
        self.selected_indices = {0}
      if auto_sort_alpha:
        self.sort_by_name()
      else:
        self.rebuild_filter()
    return added_count

  def set_tracks_from_paths(
      self, file_paths: list[str], auto_sort_alpha: bool = True
  ):
    self.tracks = [read_track_metadata(p) for p in file_paths if p]
    self.selected_indices = {0} if self.tracks else set()
    self.playing_index = None
    if auto_sort_alpha:
      self.sort_by_name()
    else:
      self.rebuild_filter()

  def clear_all_tracks(self):
    self.tracks.clear()
    self.filtered_indices.clear()
    self.selected_indices.clear()
    self.playing_index = None
    self.scroll_y = 0
    self.update()

  def remove_duplicates_by_title(self) -> int:
    if not self.tracks:
      return 0
    seen_titles: set[str] = set()
    unique_tracks: list[dict] = []
    removed_count = 0

    for tr in self.tracks:
      key = tr["title"].strip().lower()
      if key in seen_titles:
        removed_count += 1
        continue
      seen_titles.add(key)
      unique_tracks.append(tr)

    if removed_count > 0:
      self._apply_ordered_tracks(unique_tracks)
    return removed_count

  def get_saved_paths(self) -> list[str]:
    return [t["path"] for t in self.tracks]

  def update_track_duration(self, real_idx: int, duration_sec: float):
    if 0 <= real_idx < len(self.tracks) and duration_sec > 0:
      track = self.tracks[real_idx]
      track["duration_sec"] = duration_sec
      track["duration_str"] = format_duration(duration_sec)
      if track["bitrate_kbps"] is None and track["size_bytes"] > 0:
        track["bitrate_kbps"] = int(
            round((track["size_bytes"] * 8) / duration_sec / 1000)
        )
      self.update()

  def remove_selected_tracks(self) -> bool:
    if not self.selected_indices:
      return False

    removed = self.selected_indices
    playing_removed = (
        self.playing_index is not None and self.playing_index in removed
    )

    new_tracks: list[dict] = []
    new_playing_index: int | None = None

    for old_idx, track in enumerate(self.tracks):
      if old_idx in removed:
        continue
      if old_idx == self.playing_index:
        new_playing_index = len(new_tracks)
      new_tracks.append(track)

    self.tracks = new_tracks
    self.selected_indices.clear()
    self.playing_index = new_playing_index
    self.rebuild_filter()
    return playing_removed

  def real_index_at_pos(self, pos: QPoint) -> int | None:
    if (
        pos.x() < 0
        or pos.x() >= self.width()
        or pos.y() < 0
        or pos.y() >= self.height()
    ):
      return None
    vis_idx = (pos.y() + self.scroll_y) // self.CARD_H
    if 0 <= vis_idx < len(self.filtered_indices):
      return self.filtered_indices[vis_idx]
    return None

  def paintEvent(self, event):
    painter = QPainter(self)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    painter.resetTransform()
    painter.setClipRect(self.rect())

    fm_title = QFontMetrics(self.title_font)
    fm_sub = QFontMetrics(self.sub_font)

    left_x = self.ICON_X + self.ICON_SIZE + 9
    right_margin = 12

    for vis_idx, real_idx in enumerate(self.filtered_indices):
      card_top = vis_idx * self.CARD_H - self.scroll_y
      if card_top + self.CARD_H < 0:
        continue
      if card_top > self.height():
        break

      track = self.tracks[real_idx]
      is_selected = real_idx in self.selected_indices
      top_color = self.color_selected if is_selected else self.color_normal

      if not self.icon_px.isNull():
        painter.drawPixmap(self.ICON_X, card_top + self.ICON_Y, self.icon_px)

      painter.setFont(self.title_font)
      painter.setPen(top_color)

      dur_str = track["duration_str"]
      dur_w = fm_title.horizontalAdvance(dur_str)
      dur_x = self.CARD_W - right_margin - dur_w
      top_row_rect_right = QRect(dur_x, card_top + 5, dur_w, 20)
      painter.drawText(
          top_row_rect_right,
          Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
          dur_str,
      )

      title_max_w = max(20, dur_x - left_x - 10)
      numbered_title = f"{real_idx + 1}. {track['title']}"
      elided_title = fm_title.elidedText(
          numbered_title, Qt.TextElideMode.ElideRight, title_max_w
      )
      top_row_rect_left = QRect(left_x, card_top + 5, title_max_w, 20)
      painter.drawText(
          top_row_rect_left,
          Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
          elided_title,
      )

      painter.setFont(self.sub_font)
      painter.setPen(self.color_subtext)

      bitrate_part = (
          f"{track['bitrate_kbps']} kbps"
          if track["bitrate_kbps"]
          else "-- kbps"
      )
      sub_text = f"{track['format']} • {bitrate_part} • {track['size_str']}"
      sub_max_w = max(20, self.CARD_W - left_x - right_margin)
      elided_sub = fm_sub.elidedText(
          sub_text, Qt.TextElideMode.ElideRight, sub_max_w
      )
      sub_row_rect = QRect(left_x, card_top + 25, sub_max_w, 16)
      painter.drawText(
          sub_row_rect,
          Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
          elided_sub,
      )

  def mousePressEvent(self, event):
    if event.button() == Qt.MouseButton.LeftButton:
      real_idx = self.real_index_at_pos(event.position().toPoint())
      if real_idx is not None:
        ctrl_held = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        if ctrl_held:
          if real_idx in self.selected_indices:
            self.selected_indices.remove(real_idx)
          else:
            self.selected_indices.add(real_idx)
        else:
          self.selected_indices = {real_idx}
        self.update()
      else:
        if not (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
          self.selected_indices.clear()
          self.update()
      event.accept()
      return
    super().mousePressEvent(event)

  def mouseDoubleClickEvent(self, event):
    if event.button() == Qt.MouseButton.LeftButton:
      real_idx = self.real_index_at_pos(event.position().toPoint())
      if real_idx is not None:
        self.selected_indices = {real_idx}
        self.update()
        self.track_activated.emit(real_idx)
        event.accept()
        return
    super().mouseDoubleClickEvent(event)

  def wheelEvent(self, event):
    if self.max_scroll() > 0:
      delta_y = event.angleDelta().y()
      if delta_y != 0:
        step = (delta_y // 120) * self.CARD_H
        if step == 0:
          step = self.CARD_H if delta_y > 0 else -self.CARD_H
        self.scroll_y -= step
        self.clamp_scroll()
        self.update()
      event.accept()
      return
    super().wheelEvent(event)


class OptionsCardWidget(QWidget):
  clicked = pyqtSignal()

  TITLE_FONT_SIZE = 12
  SUB_FONT_SIZE = 9

  def __init__(
      self,
      title: str,
      subtitle: str,
      icon_px: QPixmap,
      x: int,
      y: int,
      w: int = 265,
      h: int = 47,
      icon_x: int = 6,
      icon_y: int = 6,
      icon_size: int = 35,
      font_family: str = "Sans Serif",
      clickable: bool = True,
      primary_color: QColor | None = None,
      selected_color: QColor | None = None,
      secondary_color: QColor | None = None,
      parent=None,
  ):
    super().__init__(parent)
    self._title = title
    self._subtitle = subtitle
    self.icon_px = icon_px
    self.icon_x = icon_x
    self.icon_y = icon_y
    self.icon_size = icon_size
    self.clickable = clickable
    self._flash_white = False

    self.color_normal = primary_color or QColor("#f58db1")
    self.color_flash = selected_color or QColor("#ffffff")
    self.color_subtext = secondary_color or QColor("#888888")

    self.setGeometry(x, y, w, h)
    self.setFixedSize(w, h)
    self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
    if self.clickable:
      self.setCursor(Qt.CursorShape.PointingHandCursor)

    self.title_font = make_smooth_font(font_family, self.TITLE_FONT_SIZE)
    self.sub_font = make_smooth_font(font_family, self.SUB_FONT_SIZE)

    self._flash_timer = QTimer(self)
    self._flash_timer.setSingleShot(True)
    self._flash_timer.setInterval(220)
    self._flash_timer.timeout.connect(self._end_flash)

  def apply_theme(
      self,
      primary_color: QColor,
      selected_color: QColor,
      secondary_color: QColor,
      icon_px: QPixmap | None = None,
  ):
    self.color_normal = primary_color
    self.color_flash = selected_color
    self.color_subtext = secondary_color
    if icon_px is not None:
      self.icon_px = icon_px
    self.update()

  def set_title(self, title: str):
    if title != self._title:
      self._title = title
      self.update()

  def set_subtitle(self, subtitle: str):
    if subtitle != self._subtitle:
      self._subtitle = subtitle
      self.update()

  def trigger_flash(self):
    self._flash_white = True
    self.update()
    self._flash_timer.start()

  def _end_flash(self):
    self._flash_white = False
    self.update()

  def paintEvent(self, event):
    painter = QPainter(self)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    painter.resetTransform()
    painter.setClipRect(self.rect())

    if not self.icon_px.isNull():
      painter.drawPixmap(self.icon_x, self.icon_y, self.icon_px)
      text_left = self.icon_x + self.icon_size + 8
    else:
      text_left = 10

    right_margin = 8
    avail_w = max(20, self.width() - text_left - right_margin)

    fm_title = QFontMetrics(self.title_font)
    fm_sub = QFontMetrics(self.sub_font)

    painter.setFont(self.title_font)
    painter.setPen(self.color_flash if self._flash_white else self.color_normal)
    elided_title = fm_title.elidedText(
        self._title, Qt.TextElideMode.ElideRight, avail_w
    )
    painter.drawText(
        QRect(text_left, 5, avail_w, 20),
        Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
        elided_title,
    )

    painter.setFont(self.sub_font)
    painter.setPen(self.color_subtext)
    elided_sub = fm_sub.elidedText(
        self._subtitle, Qt.TextElideMode.ElideRight, avail_w
    )
    painter.drawText(
        QRect(text_left, 25, avail_w, 16),
        Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
        elided_sub,
    )

  def mousePressEvent(self, event):
    if self.clickable and event.button() == Qt.MouseButton.LeftButton:
      self.trigger_flash()
      self.clicked.emit()
      event.accept()
      return
    super().mousePressEvent(event)


class ThemesListView(QWidget):
  theme_selected = pyqtSignal(str)

  CARD_W = 265
  CARD_H = 47
  TITLE_FONT_SIZE = 12
  SUB_FONT_SIZE = 9

  def __init__(
      self,
      x: int,
      y: int,
      w: int,
      h: int,
      font_family: str = "Sans Serif",
      primary_color: QColor | None = None,
      selected_color: QColor | None = None,
      secondary_color: QColor | None = None,
      parent=None,
  ):
    super().__init__(parent)
    self.themes: list[dict] = []
    self.active_folder_name: str = "Default"
    self.flashing_folder_name: str | None = None
    self.scroll_y: int = 0

    self.color_normal = primary_color or QColor("#f58db1")
    self.color_flash = selected_color or QColor("#ffffff")
    self.color_subtext = secondary_color or QColor("#888888")

    self.setGeometry(x, y, w, h)
    self.setFixedSize(w, h)
    self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
    self.setCursor(Qt.CursorShape.PointingHandCursor)

    self.title_font = make_smooth_font(font_family, self.TITLE_FONT_SIZE)
    self.sub_font = make_smooth_font(font_family, self.SUB_FONT_SIZE)

    self._flash_timer = QTimer(self)
    self._flash_timer.setSingleShot(True)
    self._flash_timer.setInterval(220)
    self._flash_timer.timeout.connect(self._end_flash)

  def apply_theme(
      self, primary_color: QColor, selected_color: QColor, secondary_color: QColor
  ):
    self.color_normal = primary_color
    self.color_flash = selected_color
    self.color_subtext = secondary_color
    self.update()

  def set_themes(self, themes: list[dict], active_folder_name: str):
    self.themes = themes
    self.active_folder_name = active_folder_name
    self.clamp_scroll()
    self.update()

  def set_active_theme_folder(self, folder_name: str):
    self.active_folder_name = folder_name
    self.update()

  def _end_flash(self):
    self.flashing_folder_name = None
    self.update()

  def max_scroll(self) -> int:
    total_h = len(self.themes) * self.CARD_H
    return max(0, total_h - self.height())

  def clamp_scroll(self):
    self.scroll_y = max(0, min(self.max_scroll(), self.scroll_y))

  def _theme_at_pos(self, pos: QPoint) -> dict | None:
    if (
        pos.x() < 0
        or pos.x() >= self.width()
        or pos.y() < 0
        or pos.y() >= self.height()
    ):
      return None
    idx = (pos.y() + self.scroll_y) // self.CARD_H
    if 0 <= idx < len(self.themes):
      return self.themes[idx]
    return None

  def paintEvent(self, event):
    painter = QPainter(self)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    painter.resetTransform()
    painter.setClipRect(self.rect())

    fm_title = QFontMetrics(self.title_font)
    fm_sub = QFontMetrics(self.sub_font)

    left_x = 10
    right_margin = 8
    avail_w = max(20, self.CARD_W - left_x - right_margin)

    for idx, theme in enumerate(self.themes):
      card_top = idx * self.CARD_H - self.scroll_y
      if card_top + self.CARD_H < 0:
        continue
      if card_top > self.height():
        break

      folder_name = theme["folder_name"]
      is_flashing = folder_name == self.flashing_folder_name
      is_active = folder_name == self.active_folder_name
      title_color = (
          self.color_flash
          if (is_flashing or is_active)
          else self.color_normal
      )

      painter.setFont(self.title_font)
      painter.setPen(title_color)
      elided_title = fm_title.elidedText(
          theme["name"], Qt.TextElideMode.ElideRight, avail_w
      )
      painter.drawText(
          QRect(left_x, card_top + 5, avail_w, 20),
          Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
          elided_title,
      )

      painter.setFont(self.sub_font)
      painter.setPen(self.color_subtext)
      sub_text = theme["description"]
      elided_sub = fm_sub.elidedText(
          sub_text, Qt.TextElideMode.ElideRight, avail_w
      )
      painter.drawText(
          QRect(left_x, card_top + 25, avail_w, 16),
          Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
          elided_sub,
      )

  def mousePressEvent(self, event):
    if event.button() == Qt.MouseButton.LeftButton:
      theme = self._theme_at_pos(event.position().toPoint())
      if theme is not None:
        self.flashing_folder_name = theme["folder_name"]
        self.update()
        self._flash_timer.start()
        self.theme_selected.emit(theme["folder_name"])
        event.accept()
        return
    super().mousePressEvent(event)

  def wheelEvent(self, event):
    if self.max_scroll() > 0:
      delta_y = event.angleDelta().y()
      if delta_y != 0:
        step = (delta_y // 120) * self.CARD_H
        if step == 0:
          step = self.CARD_H if delta_y > 0 else -self.CARD_H
        self.scroll_y -= step
        self.clamp_scroll()
        self.update()
      event.accept()
      return
    super().wheelEvent(event)


class AboutDialog(QDialog):

  def __init__(
      self, logo_path: str, font_family: str = "Sans Serif", parent=None
  ):
    super().__init__(parent)
    self.setWindowTitle("About YUC")
    self.setFixedSize(340, 270)
    self.setModal(True)

    layout = QVBoxLayout(self)
    layout.setContentsMargins(20, 20, 20, 20)
    layout.setSpacing(10)
    layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

    logo_label = QLabel(self)
    logo_label.setFixedSize(128, 128)
    logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    if os.path.isfile(logo_path):
      px = QPixmap(logo_path)
      if not px.isNull():
        px = px.scaled(
            128,
            128,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        logo_label.setPixmap(px)
    layout.addWidget(logo_label, alignment=Qt.AlignmentFlag.AlignHCenter)

    title_label = QLabel("YUC Music Player 1.0.0", self)
    title_label.setFont(make_smooth_font(font_family, 13, bold=True))
    title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(title_label)

    author_label = QLabel("Made by yixx", self)
    author_label.setFont(make_smooth_font(font_family, 10, bold=False))
    author_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(author_label)

    gh_label = QLabel(
        'GitHub: <a href="https://github.com/nazaxii">https://github.com/nazaxii</a>',
        self,
    )
    gh_label.setFont(make_smooth_font(font_family, 9, bold=False))
    gh_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    gh_label.setOpenExternalLinks(True)
    gh_label.setTextInteractionFlags(
        Qt.TextInteractionFlag.TextBrowserInteraction
    )
    layout.addWidget(gh_label)


class OptionsOverlayWidget(QWidget):
  clear_playlist_requested = pyqtSignal()
  remove_duplicates_requested = pyqtSignal()
  sort_by_name_requested = pyqtSignal()
  sort_by_format_requested = pyqtSignal()
  toggle_auto_load_requested = pyqtSignal()
  about_requested = pyqtSignal()
  theme_selected = pyqtSignal(str)

  def __init__(
      self,
      bg_px: QPixmap,
      icons: list[QPixmap],
      font_family: str,
      auto_load_enabled: bool,
      primary_color: QColor | None = None,
      selected_color: QColor | None = None,
      secondary_color: QColor | None = None,
      parent=None,
  ):
    super().__init__(parent)
    self.bg_px = bg_px
    self.color_normal = primary_color or QColor("#f58db1")
    self.color_flash = selected_color or QColor("#ffffff")
    self.color_subtext = secondary_color or QColor("#888888")

    self.setGeometry(0, 189, 540, 314)
    self.setFixedSize(540, 314)
    self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)

    left_y = [25, 72, 119, 166, 213, 260]

    def safe_icon(idx: int) -> QPixmap:
      return icons[idx] if 0 <= idx < len(icons) else QPixmap()

    self.card_clear = OptionsCardWidget(
        "Clear playlist",
        "Remove all songs from the playlist",
        safe_icon(0),
        4,
        left_y[0],
        font_family=font_family,
        primary_color=self.color_normal,
        selected_color=self.color_flash,
        secondary_color=self.color_subtext,
        parent=self,
    )
    self.card_clear.clicked.connect(self.clear_playlist_requested)

    self.card_dupes = OptionsCardWidget(
        "Remove duplicates",
        "Delete duplicate songs by title",
        safe_icon(1),
        4,
        left_y[1],
        font_family=font_family,
        primary_color=self.color_normal,
        selected_color=self.color_flash,
        secondary_color=self.color_subtext,
        parent=self,
    )
    self.card_dupes.clicked.connect(self.remove_duplicates_requested)

    self.card_sort_name = OptionsCardWidget(
        "Sort by name",
        "Sort the songs alphabetically",
        safe_icon(2),
        4,
        left_y[2],
        font_family=font_family,
        primary_color=self.color_normal,
        selected_color=self.color_flash,
        secondary_color=self.color_subtext,
        parent=self,
    )
    self.card_sort_name.clicked.connect(self.sort_by_name_requested)

    self.card_sort_format = OptionsCardWidget(
        "Sort songs by file format",
        "Sort by type: MP3, Ogg, M4A, etc.",
        safe_icon(3),
        4,
        left_y[3],
        font_family=font_family,
        primary_color=self.color_normal,
        selected_color=self.color_flash,
        secondary_color=self.color_subtext,
        parent=self,
    )
    self.card_sort_format.clicked.connect(self.sort_by_format_requested)

    self.card_auto_load = OptionsCardWidget(
        self._auto_load_title(auto_load_enabled),
        "Adds songs from the “music” folder",
        safe_icon(4),
        4,
        left_y[4],
        font_family=font_family,
        primary_color=self.color_normal,
        selected_color=self.color_flash,
        secondary_color=self.color_subtext,
        parent=self,
    )
    self.card_auto_load.clicked.connect(self.toggle_auto_load_requested)

    self.card_about = OptionsCardWidget(
        "About YUC",
        "Detailed information about the player",
        safe_icon(5),
        4,
        left_y[5],
        font_family=font_family,
        primary_color=self.color_normal,
        selected_color=self.color_flash,
        secondary_color=self.color_subtext,
        parent=self,
    )
    self.card_about.clicked.connect(self.about_requested)

    self.card_themes_header = OptionsCardWidget(
        "Themes",
        "Select an interface theme.",
        safe_icon(6),
        271,
        25,
        font_family=font_family,
        clickable=False,
        primary_color=self.color_normal,
        selected_color=self.color_flash,
        secondary_color=self.color_subtext,
        parent=self,
    )

    self.themes_list = ThemesListView(
        271,
        72,
        265,
        235,
        font_family=font_family,
        primary_color=self.color_normal,
        selected_color=self.color_flash,
        secondary_color=self.color_subtext,
        parent=self,
    )
    self.themes_list.theme_selected.connect(self.theme_selected)

  def apply_theme(
      self,
      bg_px: QPixmap,
      icons: list[QPixmap],
      primary_color: QColor,
      selected_color: QColor,
      secondary_color: QColor,
  ):
    self.bg_px = bg_px
    self.color_normal = primary_color
    self.color_flash = selected_color
    self.color_subtext = secondary_color

    def safe_icon(idx: int) -> QPixmap:
      return icons[idx] if 0 <= idx < len(icons) else QPixmap()

    self.card_clear.apply_theme(
        primary_color, selected_color, secondary_color, safe_icon(0)
    )
    self.card_dupes.apply_theme(
        primary_color, selected_color, secondary_color, safe_icon(1)
    )
    self.card_sort_name.apply_theme(
        primary_color, selected_color, secondary_color, safe_icon(2)
    )
    self.card_sort_format.apply_theme(
        primary_color, selected_color, secondary_color, safe_icon(3)
    )
    self.card_auto_load.apply_theme(
        primary_color, selected_color, secondary_color, safe_icon(4)
    )
    self.card_about.apply_theme(
        primary_color, selected_color, secondary_color, safe_icon(5)
    )
    self.card_themes_header.apply_theme(
        primary_color, selected_color, secondary_color, safe_icon(6)
    )
    self.themes_list.apply_theme(primary_color, selected_color, secondary_color)
    self.update()

  @staticmethod
  def _auto_load_title(enabled: bool) -> str:
    return f"Auto load: {'enabled' if enabled else 'disabled'}"

  def set_auto_load_enabled(self, enabled: bool):
    self.card_auto_load.set_title(self._auto_load_title(enabled))

  def set_themes(self, themes: list[dict], active_folder_name: str):
    self.themes_list.set_themes(themes, active_folder_name)

  def paintEvent(self, event):
    painter = QPainter(self)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
    painter.resetTransform()
    painter.setClipRect(self.rect())

    if not self.bg_px.isNull():
      painter.drawPixmap(0, 0, self.bg_px)
    else:
      painter.fillRect(self.rect(), QColor("#18181d"))


class YucPlayer(QWidget):
  volume_changed = pyqtSignal(int)

  TIMER_POSITIONS = [
      (13, 41),
      (49, 41),
      (85, 41),
      (121, 41),
      (157, 41),
  ]

  VIS_COLS_X = [204, 219, 234, 249, 264, 279, 294]
  VIS_ROWS_Y_BOTTOM_UP = [83, 76, 69, 62, 55, 48, 41]
  VIS_BLOCK_W = 14
  VIS_BLOCK_H = 6

  def __init__(self):
    super().__init__()
    self.setFixedSize(540, 540)
    self.setWindowTitle("Yuc Player")
    self.setWindowFlags(Qt.WindowType.FramelessWindowHint)

    self.titlebar_rect = QRect(0, 0, 451, 25)
    self.is_dragging = False
    self.drag_offset = QPoint()

    self.is_playing = False
    self.is_repeat = False
    self.is_shuffle = False
    self.is_options_open = False
    self.auto_load_enabled = False
    self.active_theme_folder = "Default"
    self.available_themes: list[dict] = []
    self.volume = 80
    self.current_time_str = "00:00"

    self.primary_color = QColor(DEFAULT_THEME_COLORS["primary_color"])
    self.selected_color = QColor(DEFAULT_THEME_COLORS["selected_color"])
    self.secondary_color = QColor(DEFAULT_THEME_COLORS["secondary_color"])
    self.search_bar_text_color = QColor(
        DEFAULT_THEME_COLORS["search_bar_text"]
    )

    self._pending_seek_pos_ms: int | None = None
    self._pending_resume_play: bool = False

    self.vis_levels = [0.0] * len(self.VIS_COLS_X)
    self.vis_targets = [0.0] * len(self.VIS_COLS_X)
    self.vis_rect = QRect(
        min(self.VIS_COLS_X),
        min(self.VIS_ROWS_Y_BOTTOM_UP),
        (max(self.VIS_COLS_X) - min(self.VIS_COLS_X)) + self.VIS_BLOCK_W,
        (max(self.VIS_ROWS_Y_BOTTOM_UP) - min(self.VIS_ROWS_Y_BOTTOM_UP))
        + self.VIS_BLOCK_H,
    )

    self.vol_min_x = 260
    self.vol_max_x = 430
    self.vol_y = 127
    self.vol_thumb_w = 22
    self.vol_thumb_h = 22
    self.vol_track_click_rect = QRect(260, 123, 192, 30)
    self.is_scrubbing_volume = False

    self.prog_min_x = 13
    self.prog_max_x = 494
    self.prog_y = 103
    self.prog_thumb_w = 33
    self.prog_thumb_h = 15
    self.prog_track_click_rect = QRect(13, 103, 514, 15)
    self.is_scrubbing_progress = False
    self.track_duration_ms = 0

    self.base_dir = os.path.dirname(os.path.abspath(__file__))
    self.themes_root_dir = os.path.join(self.base_dir, "themes")
    self.playlist_save_path = os.path.join(self.base_dir, "playlist.json")

    self.load_saved_playlist_config_only()
    self.scan_themes_folder()
    self._load_theme_assets(self.active_theme_folder)

    self.init_visualizer_engine()
    self.init_audio_engine()
    self.init_playlist()
    self.init_marquee_title()
    self.init_search_bar()
    self.init_buttons()
    self.init_volume_slider()
    self.init_progress_slider()
    self.load_saved_playlist()
    self.init_options_overlay()
    self.apply_current_theme()
    self.run_auto_load_if_enabled()

  def _resolve_asset_path(self, filename: str) -> str:
    theme_dir = os.path.join(self.themes_root_dir, self.active_theme_folder)
    theme_asset = os.path.join(theme_dir, filename)
    if os.path.exists(theme_asset):
      return theme_asset
    return os.path.join(self.base_dir, filename)

  def scan_themes_folder(self):
    themes_list: list[dict] = []
    if os.path.isdir(self.themes_root_dir):
      try:
        entries = sorted(
            os.listdir(self.themes_root_dir), key=lambda s: s.lower()
        )
      except OSError:
        entries = []

      for entry in entries:
        full_path = os.path.join(self.themes_root_dir, entry)
        if os.path.isdir(full_path):
          themes_list.append(load_theme_config(full_path, entry))

    self.available_themes = themes_list

    if not self.available_themes:
      self.active_theme_folder = "Default"
      self.primary_color = QColor(DEFAULT_THEME_COLORS["primary_color"])
      self.selected_color = QColor(DEFAULT_THEME_COLORS["selected_color"])
      self.secondary_color = QColor(DEFAULT_THEME_COLORS["secondary_color"])
      self.search_bar_text_color = QColor(
          DEFAULT_THEME_COLORS["search_bar_text"]
      )
      return

    matched = next(
        (
            t
            for t in self.available_themes
            if t["folder_name"] == self.active_theme_folder
        ),
        None,
    )
    if matched is None:
      matched = next(
          (
              t
              for t in self.available_themes
              if t["folder_name"].lower() == "default"
          ),
          self.available_themes[0],
      )
      self.active_theme_folder = matched["folder_name"]

    self.primary_color = matched["primary_color"]
    self.selected_color = matched["selected_color"]
    self.secondary_color = matched["secondary_color"]
    self.search_bar_text_color = matched["search_bar_text"]

  def _load_theme_assets(self, folder_name: str):
    self.active_theme_folder = folder_name
    self.bg_pixmap = self.load_exact_pixmap("main.png", 540, 540)
    if self.bg_pixmap.isNull():
      self.bg_pixmap = QPixmap(540, 540)
      self.bg_pixmap.fill(QColor("#1e1e24"))
      p = QPainter(self.bg_pixmap)
      p.setPen(QColor("#00ff66"))
      p.setFont(make_smooth_font("Monospace", 11))
      p.drawText(
          self.bg_pixmap.rect(),
          Qt.AlignmentFlag.AlignCenter,
          "main.png (540x540) не знайдено\n[Тягни за titlebar 0..451, 0..25]",
      )
      p.end()

    self.elements_pixmap = self.load_exact_pixmap("elements.png")
    self.options_pixmap = self.load_exact_pixmap("options.png", 540, 314)
    self.vol_bar_px = self.cut_sprite(353, 478, 184, 14)
    self.vis_block_px = self.cut_sprite(
        257, 150, self.VIS_BLOCK_W, self.VIS_BLOCK_H
    )
    self.digit_sprites = self.load_digit_sprites()

  def apply_current_theme(self):
    self._load_theme_assets(self.active_theme_folder)

    card_icon_px = self.cut_sprite(356, 514, 19, 19)
    self.playlist_view.apply_theme(
        card_icon_px,
        self.primary_color,
        self.selected_color,
        self.secondary_color,
    )

    self.marquee_title.set_text_color(self.primary_color)

    search_normal = self.cut_sprite(86, 478, 266, 23)
    search_active = self.cut_sprite(86, 503, 266, 23)
    self.search_bar.apply_theme(
        search_normal, search_active, self.search_bar_text_color
    )

    win_w, win_h = 38, 17
    self.btn_minimize.set_sprites(
        self.cut_sprite(0, 272, win_w, win_h),
        self.cut_sprite(88, 272, win_w, win_h),
    )
    self.btn_close.set_sprites(
        self.cut_sprite(0, 253, win_w, win_h),
        self.cut_sprite(88, 253, win_w, win_h),
    )

    tr_w, tr_h = 84, 62
    self.btn_prev.set_sprites(
        self.cut_sprite(0, 0, tr_w, tr_h),
        self.cut_sprite(88, 0, tr_w, tr_h),
    )
    self.play_normal = self.cut_sprite(0, 124, tr_w, tr_h)
    self.play_pressed = self.cut_sprite(88, 124, tr_w, tr_h)
    self.pause_normal = self.cut_sprite(0, 62, tr_w, tr_h)
    self.pause_pressed = self.cut_sprite(88, 62, tr_w, tr_h)
    self.sync_play_pause_sprite()

    self.btn_next.set_sprites(
        self.cut_sprite(0, 186, tr_w, tr_h),
        self.cut_sprite(88, 186, tr_w, tr_h),
    )

    rs_w, rs_h = 84, 30
    rep_n = self.cut_sprite(0, 310, rs_w, rs_h)
    rep_p = self.cut_sprite(88, 310, rs_w, rs_h)
    self.btn_repeat.set_sprites(rep_n, rep_p, active_px=rep_p)

    shuf_n = self.cut_sprite(0, 340, rs_w, rs_h)
    shuf_p = self.cut_sprite(88, 340, rs_w, rs_h)
    self.btn_shuffle.set_sprites(shuf_n, shuf_p, active_px=shuf_p)

    bot_w, bot_h = 84, 34
    self.btn_add_songs.set_sprites(
        self.cut_sprite(0, 374, bot_w, bot_h),
        self.cut_sprite(88, 374, bot_w, bot_h),
    )
    self.btn_remove_songs.set_sprites(
        self.cut_sprite(0, 408, bot_w, bot_h),
        self.cut_sprite(88, 408, bot_w, bot_h),
    )
    opt_n = self.cut_sprite(0, 442, bot_w, bot_h)
    opt_p = self.cut_sprite(88, 442, bot_w, bot_h)
    self.btn_options.set_sprites(opt_n, opt_p, active_px=opt_p)

    self.vol_thumb.set_thumb_pixmap(
        self.cut_sprite(393, 497, self.vol_thumb_w, self.vol_thumb_h)
    )
    self.prog_thumb.set_thumb_pixmap(
        self.cut_sprite(356, 497, self.prog_thumb_w, self.prog_thumb_h)
    )

    icon_x_coords = [181, 218, 255, 292, 329, 366, 403]
    card_icons = [self.cut_sprite(ix, 200, 35, 35) for ix in icon_x_coords]
    self.options_overlay.apply_theme(
        self.options_pixmap,
        card_icons,
        self.primary_color,
        self.selected_color,
        self.secondary_color,
    )
    self.options_overlay.set_themes(
        self.available_themes, self.active_theme_folder
    )
    self.update()

  def on_theme_selected(self, folder_name: str):
    matched = next(
        (t for t in self.available_themes if t["folder_name"] == folder_name),
        None,
    )
    if matched is None:
      return
    self.active_theme_folder = matched["folder_name"]
    self.primary_color = matched["primary_color"]
    self.selected_color = matched["selected_color"]
    self.secondary_color = matched["secondary_color"]
    self.search_bar_text_color = matched["search_bar_text"]
    self.apply_current_theme()
    self.save_playlist()

  def show_about_dialog(self):
    logo_path = os.path.join(self.base_dir, "icons", "yuc_logo_128.png")
    dlg = AboutDialog(logo_path, font_family=self.ui_font_family, parent=self)
    dlg.exec()

  def init_visualizer_engine(self):
    self.analyzer_thread = AudioAnalyzerThread(
        max_rows=len(self.VIS_ROWS_Y_BOTTOM_UP), parent=self
    )
    self.analyzer_thread.bands_ready.connect(self._on_analyzer_bands)
    self.analyzer_thread.start()

    self.vis_decay_timer = QTimer(self)
    self.vis_decay_timer.setInterval(40)
    self.vis_decay_timer.timeout.connect(self._step_visualizer_smoothing)
    self.vis_decay_timer.start()

  def _on_analyzer_bands(self, bands: list[float]):
    if not self.is_playing:
      return
    for i in range(min(len(self.vis_targets), len(bands))):
      incoming = float(bands[i])
      self.vis_targets[i] = max(self.vis_targets[i] * 0.55, incoming)
    if not self.vis_decay_timer.isActive():
      self.vis_decay_timer.start()

  def _step_visualizer_smoothing(self):
    max_rows = float(len(self.VIS_ROWS_Y_BOTTOM_UP))
    any_changed = False

    for i in range(len(self.vis_levels)):
      target = self.vis_targets[i] if self.is_playing else 0.0
      current = self.vis_levels[i]

      if target > current:
        new_val = min(target, current + (target - current) * 0.45)
      else:
        fall_step = 0.22 if self.is_playing else 0.65
        new_val = max(target, current - fall_step)

      decay_factor = 0.84 if self.is_playing else 0.5
      self.vis_targets[i] = max(0.0, self.vis_targets[i] * decay_factor)

      new_val = max(0.0, min(max_rows, new_val))
      if new_val != current:
        self.vis_levels[i] = new_val
        any_changed = True

    if any_changed:
      self.update(self.vis_rect)

  def load_custom_font(
      self, filename: str, fallback: str = "Sans Serif"
  ) -> str:
    font_path = os.path.join(self.base_dir, filename)
    if not os.path.exists(font_path):
      return fallback
    font_id = QFontDatabase.addApplicationFont(font_path)
    if font_id == -1:
      return fallback
    families = QFontDatabase.applicationFontFamilies(font_id)
    return families[0] if families else fallback

  def load_digit_sprites(self) -> dict[str, QPixmap]:
    w, h = 36, 48
    coords = {
        "0": (181, 0),
        "1": (219, 0),
        "2": (257, 0),
        "3": (181, 50),
        "4": (219, 50),
        "5": (257, 50),
        "6": (181, 100),
        "7": (219, 100),
        "8": (257, 100),
        "9": (181, 150),
        ":": (219, 150),
    }
    return {
        ch: self.cut_sprite(sx, sy, w, h) for ch, (sx, sy) in coords.items()
    }

  def init_audio_engine(self):
    self.media_devices = QMediaDevices(self)
    self.media_devices.audioOutputsChanged.connect(
        self._schedule_audio_rebind
    )

    self._rebind_timer = QTimer(self)
    self._rebind_timer.setSingleShot(True)
    self._rebind_timer.setInterval(120)
    self._rebind_timer.timeout.connect(self._rebind_audio_output)

    self.audio_output = QAudioOutput(
        self.media_devices.defaultAudioOutput(), self
    )
    self.audio_output.setVolume(self.volume / 100.0)

    self.media_player = QMediaPlayer(self)
    self.media_player.setAudioOutput(self.audio_output)

    self.media_player.playbackStateChanged.connect(
        self.on_playback_state_changed
    )
    self.media_player.mediaStatusChanged.connect(self.on_media_status_changed)
    self.media_player.durationChanged.connect(self.on_player_duration_changed)
    self.media_player.positionChanged.connect(self.on_player_position_changed)

  def _schedule_audio_rebind(self):
    self._rebind_timer.start()

  def _rebind_audio_output(self):
    default_dev = self.media_devices.defaultAudioOutput()
    if default_dev.isNull():
      return

    was_playing = (
        self.media_player.playbackState()
        == QMediaPlayer.PlaybackState.PlayingState
    )
    current_pos = self.media_player.position()
    current_source = self.media_player.source()

    self._pending_seek_pos_ms = current_pos if current_pos > 1000 else None
    self._pending_resume_play = was_playing

    self.media_player.blockSignals(True)
    self.media_player.stop()
    self.media_player.setSource(QUrl())
    self.media_player.setAudioOutput(None)
    self.media_player.blockSignals(False)

    old_player = self.media_player
    old_output = self.audio_output

    new_output = QAudioOutput(default_dev, self)
    new_output.setVolume(self.volume / 100.0)

    new_player = QMediaPlayer(self)
    new_player.setAudioOutput(new_output)
    new_player.playbackStateChanged.connect(self.on_playback_state_changed)
    new_player.mediaStatusChanged.connect(self.on_media_status_changed)
    new_player.durationChanged.connect(self.on_player_duration_changed)
    new_player.positionChanged.connect(self.on_player_position_changed)

    self.audio_output = new_output
    self.media_player = new_player

    if old_player is not None:
      old_player.deleteLater()
    if old_output is not None:
      old_output.deleteLater()

    if not current_source.isEmpty():
      self.media_player.setSource(current_source)

  def load_exact_pixmap(
      self, filename: str, w: int | None = None, h: int | None = None
  ) -> QPixmap:
    path = self._resolve_asset_path(filename)
    if not os.path.exists(path):
      return QPixmap()

    img = QImage(path)
    if img.isNull():
      return QPixmap()

    img = img.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
    img.setDevicePixelRatio(1.0)

    if (
        w is not None
        and h is not None
        and (img.width() != w or img.height() != h)
    ):
      img = img.scaled(
          w,
          h,
          Qt.AspectRatioMode.IgnoreAspectRatio,
          Qt.TransformationMode.FastTransformation,
      )
      img.setDevicePixelRatio(1.0)

    return QPixmap.fromImage(img)

  def cut_sprite(self, x: int, y: int, w: int, h: int) -> QPixmap:
    if self.elements_pixmap.isNull():
      fallback = QPixmap(w, h)
      fallback.fill(QColor(68, 68, 85, 180))
      return fallback

    px = self.elements_pixmap.copy(x, y, w, h)
    px.setDevicePixelRatio(1.0)
    return px

  def init_playlist(self):
    card_icon_px = self.cut_sprite(356, 514, 19, 19)
    neue_family = self.load_custom_font("font.ttf", fallback="Sans Serif")
    self.ui_font_family = neue_family
    self.playlist_view = PlaylistView(
        card_icon_px,
        1,
        189,
        538,
        314,
        self,
        font_family=neue_family,
        primary_color=self.primary_color,
        selected_color=self.selected_color,
        secondary_color=self.secondary_color,
    )
    self.playlist_view.track_activated.connect(self.play_track_at_index)

  def init_marquee_title(self):
    self.marquee_title = MarqueeLabel(
        318,
        41,
        212,
        20,
        self,
        font_family=self.ui_font_family,
        text_color=self.primary_color,
    )

  def init_search_bar(self):
    search_normal = self.cut_sprite(86, 478, 266, 23)
    search_active = self.cut_sprite(86, 503, 266, 23)
    self.search_bar = SearchBarWidget(
        search_normal,
        search_active,
        266,
        510,
        266,
        23,
        self,
        font_family=self.ui_font_family,
        text_color=self.search_bar_text_color,
    )
    self.search_bar.text_changed.connect(self.playlist_view.set_filter_query)

  def init_buttons(self):
    win_w, win_h = 38, 17

    min_normal = self.cut_sprite(0, 272, win_w, win_h)
    min_pressed = self.cut_sprite(88, 272, win_w, win_h)
    self.btn_minimize = SpriteButton(
        min_normal, min_pressed, 454, 1, win_w, win_h, self
    )
    self.btn_minimize.clicked.connect(self.showMinimized)

    close_normal = self.cut_sprite(0, 253, win_w, win_h)
    close_pressed = self.cut_sprite(88, 253, win_w, win_h)
    self.btn_close = SpriteButton(
        close_normal, close_pressed, 494, 1, win_w, win_h, self
    )
    self.btn_close.clicked.connect(self.close)

    tr_w, tr_h = 84, 62

    prev_normal = self.cut_sprite(0, 0, tr_w, tr_h)
    prev_pressed = self.cut_sprite(88, 0, tr_w, tr_h)
    self.btn_prev = SpriteButton(
        prev_normal, prev_pressed, 1, 125, tr_w, tr_h, self
    )
    self.btn_prev.clicked.connect(self.on_prev_track)

    self.play_normal = self.cut_sprite(0, 124, tr_w, tr_h)
    self.play_pressed = self.cut_sprite(88, 124, tr_w, tr_h)
    self.pause_normal = self.cut_sprite(0, 62, tr_w, tr_h)
    self.pause_pressed = self.cut_sprite(88, 62, tr_w, tr_h)

    self.btn_play_pause = SpriteButton(
        self.play_normal, self.play_pressed, 87, 125, tr_w, tr_h, self
    )
    self.btn_play_pause.clicked.connect(self.toggle_play_pause)
    self.sync_play_pause_sprite()

    next_normal = self.cut_sprite(0, 186, tr_w, tr_h)
    next_pressed = self.cut_sprite(88, 186, tr_w, tr_h)
    self.btn_next = SpriteButton(
        next_normal, next_pressed, 173, 125, tr_w, tr_h, self
    )
    self.btn_next.clicked.connect(self.on_next_track)

    rs_w, rs_h = 84, 30

    repeat_normal = self.cut_sprite(0, 310, rs_w, rs_h)
    repeat_pressed = self.cut_sprite(88, 310, rs_w, rs_h)
    self.btn_repeat = SpriteButton(
        repeat_normal,
        repeat_pressed,
        455,
        125,
        rs_w,
        rs_h,
        self,
        active_px=repeat_pressed,
    )
    self.btn_repeat.clicked.connect(self.toggle_repeat)

    shuffle_normal = self.cut_sprite(0, 340, rs_w, rs_h)
    shuffle_pressed = self.cut_sprite(88, 340, rs_w, rs_h)
    self.btn_shuffle = SpriteButton(
        shuffle_normal,
        shuffle_pressed,
        455,
        157,
        rs_w,
        rs_h,
        self,
        active_px=shuffle_pressed,
    )
    self.btn_shuffle.clicked.connect(self.toggle_shuffle)

    bot_w, bot_h = 84, 34

    add_normal = self.cut_sprite(0, 374, bot_w, bot_h)
    add_pressed = self.cut_sprite(88, 374, bot_w, bot_h)
    self.btn_add_songs = SpriteButton(
        add_normal, add_pressed, 1, 503, bot_w, bot_h, self
    )
    self.btn_add_songs.clicked.connect(self.on_add_songs)

    rem_normal = self.cut_sprite(0, 408, bot_w, bot_h)
    rem_pressed = self.cut_sprite(88, 408, bot_w, bot_h)
    self.btn_remove_songs = SpriteButton(
        rem_normal, rem_pressed, 87, 503, bot_w, bot_h, self
    )
    self.btn_remove_songs.clicked.connect(self.on_remove_songs)

    opt_normal = self.cut_sprite(0, 442, bot_w, bot_h)
    opt_pressed = self.cut_sprite(88, 442, bot_w, bot_h)
    self.btn_options = SpriteButton(
        opt_normal,
        opt_pressed,
        173,
        503,
        bot_w,
        bot_h,
        self,
        active_px=opt_pressed,
    )
    self.btn_options.clicked.connect(self.toggle_options)

  def init_options_overlay(self):
    icon_x_coords = [181, 218, 255, 292, 329, 366, 403]
    card_icons = [self.cut_sprite(ix, 200, 35, 35) for ix in icon_x_coords]

    self.options_overlay = OptionsOverlayWidget(
        self.options_pixmap,
        card_icons,
        self.ui_font_family,
        self.auto_load_enabled,
        primary_color=self.primary_color,
        selected_color=self.selected_color,
        secondary_color=self.secondary_color,
        parent=self,
    )
    self.options_overlay.set_themes(
        self.available_themes, self.active_theme_folder
    )
    self.options_overlay.clear_playlist_requested.connect(
        self.on_option_clear_playlist
    )
    self.options_overlay.remove_duplicates_requested.connect(
        self.on_option_remove_duplicates
    )
    self.options_overlay.sort_by_name_requested.connect(
        self.on_option_sort_by_name
    )
    self.options_overlay.sort_by_format_requested.connect(
        self.on_option_sort_by_format
    )
    self.options_overlay.toggle_auto_load_requested.connect(
        self.on_option_toggle_auto_load
    )
    self.options_overlay.about_requested.connect(self.show_about_dialog)
    self.options_overlay.theme_selected.connect(self.on_theme_selected)
    self.options_overlay.hide()

  def set_options_open(self, open_state: bool):
    self.is_options_open = bool(open_state)
    self.btn_options.set_active(self.is_options_open)
    if self.is_options_open:
      self.playlist_view.hide()
      self.options_overlay.show()
      self.options_overlay.raise_()
    else:
      self.options_overlay.hide()
      self.playlist_view.show()

  def toggle_options(self):
    self.set_options_open(not self.is_options_open)

  def _stop_active_playback_and_reset_ui(self):
    self.media_player.stop()
    self.analyzer_thread.stop_stream()
    self.analyzer_thread = AudioAnalyzerThread(
        max_rows=len(self.VIS_ROWS_Y_BOTTOM_UP), parent=self
    )
    self.analyzer_thread.bands_ready.connect(self._on_analyzer_bands)
    self.analyzer_thread.start()
    self.marquee_title.set_text("")
    self.track_duration_ms = 0
    self.set_progress_thumb_x(self.prog_min_x)
    self._update_timer_display(0)

  def on_option_clear_playlist(self):
    self.playlist_view.clear_all_tracks()
    self._stop_active_playback_and_reset_ui()
    self.save_playlist()

  def on_option_remove_duplicates(self):
    removed = self.playlist_view.remove_duplicates_by_title()
    if removed > 0:
      self.save_playlist()

  def on_option_sort_by_name(self):
    self.playlist_view.sort_by_name()
    self.save_playlist()

  def on_option_sort_by_format(self):
    self.playlist_view.sort_by_format_and_name()
    self.save_playlist()

  def on_option_toggle_auto_load(self):
    self.auto_load_enabled = not self.auto_load_enabled
    self.options_overlay.set_auto_load_enabled(self.auto_load_enabled)
    self.save_playlist()
    if self.auto_load_enabled:
      self.run_auto_load_if_enabled()

  def _get_music_dirs(self) -> list[str]:
    candidates: list[str] = []
    std_loc = QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.MusicLocation
    )
    if std_loc:
      candidates.append(std_loc)

    home = os.path.expanduser("~")
    candidates.append(os.path.join(home, "music"))
    candidates.append(os.path.join(home, "Music"))

    unique_dirs: list[str] = []
    seen: set[str] = set()
    for d in candidates:
      if not d:
        continue
      norm = os.path.abspath(d)
      if norm not in seen and os.path.isdir(norm):
        seen.add(norm)
        unique_dirs.append(norm)
    return unique_dirs

  def _scan_music_folder_for_new_tracks(self) -> list[str]:
    music_dirs = self._get_music_dirs()
    if not music_dirs:
      return []

    existing_paths = {
        os.path.abspath(t["path"]) for t in self.playlist_view.tracks
    }
    new_files: list[str] = []

    for music_dir in music_dirs:
      for root, _, files in os.walk(music_dir):
        for fname in files:
          _, ext = os.path.splitext(fname)
          if ext.lower() in AUDIO_EXTENSIONS:
            full_path = os.path.abspath(os.path.join(root, fname))
            if full_path not in existing_paths:
              existing_paths.add(full_path)
              new_files.append(full_path)

    return new_files

  def run_auto_load_if_enabled(self):
    if not self.auto_load_enabled:
      return
    new_files = self._scan_music_folder_for_new_tracks()
    if new_files:
      self.playlist_view.add_tracks(new_files, auto_sort_alpha=True)
      self.save_playlist()

  def init_volume_slider(self):
    thumb_px = self.cut_sprite(393, 497, self.vol_thumb_w, self.vol_thumb_h)
    self.vol_thumb = VolumeThumb(
        thumb_px,
        self.vol_min_x,
        self.vol_y,
        self.vol_thumb_w,
        self.vol_thumb_h,
        self,
    )
    self.vol_thumb.position_changed.connect(self.set_volume_thumb_x)
    self.set_volume_percent(self.volume)

  def init_progress_slider(self):
    prog_px = self.cut_sprite(356, 497, self.prog_thumb_w, self.prog_thumb_h)
    self.prog_thumb = ProgressThumb(
        prog_px,
        self.prog_min_x,
        self.prog_y,
        self.prog_thumb_w,
        self.prog_thumb_h,
        self,
    )
    self.prog_thumb.drag_started.connect(self._on_prog_drag_started)
    self.prog_thumb.position_changed.connect(self._on_prog_drag_moved)
    self.prog_thumb.drag_ended.connect(self._seek_to_thumb_x)

  def set_progress_thumb_x(self, target_x: int):
    clamped_x = max(self.prog_min_x, min(self.prog_max_x, int(target_x)))
    self.prog_thumb.move(clamped_x, self.prog_y)

  def _thumb_x_to_ratio(self, thumb_x: int) -> float:
    span = self.prog_max_x - self.prog_min_x
    if span <= 0:
      return 0.0
    clamped = max(self.prog_min_x, min(self.prog_max_x, thumb_x))
    return (clamped - self.prog_min_x) / span

  def _on_prog_drag_started(self):
    self.is_scrubbing_progress = True

  def _on_prog_drag_moved(self, target_x: int):
    self.set_progress_thumb_x(target_x)
    if self.track_duration_ms > 0:
      ratio = self._thumb_x_to_ratio(self.prog_thumb.x())
      preview_ms = int(ratio * self.track_duration_ms)
      self._update_timer_display(preview_ms)

  def _seek_to_thumb_x(self, thumb_x: int):
    self.set_progress_thumb_x(thumb_x)
    if self.track_duration_ms > 0:
      ratio = self._thumb_x_to_ratio(self.prog_thumb.x())
      target_ms = int(ratio * self.track_duration_ms)
      self.media_player.setPosition(target_ms)
      self._update_timer_display(target_ms)
      idx = self.playlist_view.playing_index
      if idx is not None and 0 <= idx < len(self.playlist_view.tracks):
        self.analyzer_thread.configure(
            self.playlist_view.tracks[idx]["path"], target_ms / 1000.0
        )
        self.analyzer_thread.set_paused(not self.is_playing)
    self.is_scrubbing_progress = False

  def _update_timer_display(self, ms: int):
    new_str = format_timer_mm_ss(ms)
    if new_str != self.current_time_str:
      self.current_time_str = new_str
      self.update(13, 41, 180, 48)

  def set_volume_thumb_x(self, target_x: int):
    clamped_x = max(self.vol_min_x, min(self.vol_max_x, int(target_x)))
    self.vol_thumb.move(clamped_x, self.vol_y)

    span = self.vol_max_x - self.vol_min_x
    new_volume = (
        int(round(((clamped_x - self.vol_min_x) / span) * 100)) if span > 0 else 0
    )
    self.volume = new_volume
    self.audio_output.setVolume(self.volume / 100.0)
    self.volume_changed.emit(self.volume)

  def set_volume_percent(self, percent: int):
    percent = max(0, min(100, int(percent)))
    span = self.vol_max_x - self.vol_min_x
    target_x = self.vol_min_x + int(round((percent / 100.0) * span))
    self.set_volume_thumb_x(target_x)

  def _read_playlist_json(self) -> dict:
    if not os.path.exists(self.playlist_save_path):
      return {}
    try:
      with open(self.playlist_save_path, "r", encoding="utf-8") as f:
        data = json.load(f)
      return data if isinstance(data, dict) else {}
    except Exception:
      return {}

  def load_saved_playlist_config_only(self):
    data = self._read_playlist_json()
    self.auto_load_enabled = bool(data.get("auto_load", False))
    saved_theme = data.get("theme", "Default")
    if isinstance(saved_theme, str) and saved_theme.strip():
      self.active_theme_folder = saved_theme.strip()

  def save_playlist(self):
    try:
      data = {
          "tracks": self.playlist_view.get_saved_paths(),
          "auto_load": self.auto_load_enabled,
          "theme": self.active_theme_folder,
      }
      with open(self.playlist_save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError:
      pass

  def load_saved_playlist(self):
    data = self._read_playlist_json()
    if not data:
      return
    self.auto_load_enabled = bool(data.get("auto_load", False))
    saved_theme = data.get("theme", self.active_theme_folder)
    if isinstance(saved_theme, str) and saved_theme.strip():
      self.active_theme_folder = saved_theme.strip()
    paths = data.get("tracks", [])
    if isinstance(paths, list):
      valid_paths = [p for p in paths if isinstance(p, str) and p]
      self.playlist_view.set_tracks_from_paths(
          valid_paths, auto_sort_alpha=True
      )

  def show_missing_song_error(self):
    msg = QMessageBox(self)
    msg.setWindowTitle("Yuc Player")
    msg.setIcon(QMessageBox.Icon.Warning)
    msg.setText("Error: Unable to find this song")
    msg.exec()

  def sync_play_pause_sprite(self):
    if self.is_playing:
      self.btn_play_pause.set_sprites(self.pause_normal, self.pause_pressed)
    else:
      self.btn_play_pause.set_sprites(self.play_normal, self.play_pressed)
    self.marquee_title.set_playing(self.is_playing)
    self.analyzer_thread.set_paused(not self.is_playing)
    if not self.vis_decay_timer.isActive():
      self.vis_decay_timer.start()

  def play_track_at_index(self, index: int):
    tracks = self.playlist_view.tracks
    if not (0 <= index < len(tracks)):
      return

    track = tracks[index]
    if not os.path.exists(track["path"]):
      self.show_missing_song_error()
      return

    self.playlist_view.playing_index = index
    self.playlist_view.selected_indices = {index}
    self.playlist_view.ensure_visible_real_index(index)
    self.playlist_view.update()

    self.marquee_title.set_text(track["title"])

    self.track_duration_ms = (
        int(track["duration_sec"] * 1000) if track["duration_sec"] else 0
    )
    self.set_progress_thumb_x(self.prog_min_x)
    self._update_timer_display(0)

    self.analyzer_thread.configure(track["path"], start_sec=0.0)
    self.media_player.setSource(QUrl.fromLocalFile(track["path"]))
    self.media_player.play()

  def toggle_play_pause(self):
    tracks = self.playlist_view.tracks
    if not tracks:
      return

    state = self.media_player.playbackState()
    if state == QMediaPlayer.PlaybackState.PlayingState:
      self.media_player.pause()
      return

    if state == QMediaPlayer.PlaybackState.PausedState:
      self.media_player.play()
      return

    if (
        self.playlist_view.playing_index is not None
        and 0 <= self.playlist_view.playing_index < len(tracks)
    ):
      target = self.playlist_view.playing_index
    elif self.playlist_view.selected_indices:
      target = min(self.playlist_view.selected_indices)
    else:
      target = 0

    self.play_track_at_index(target)

  def on_prev_track(self):
    tracks = self.playlist_view.tracks
    if not tracks:
      return

    if self.media_player.position() > 3000:
      self.media_player.setPosition(0)
      idx = self.playlist_view.playing_index
      if idx is not None and 0 <= idx < len(tracks):
        self.analyzer_thread.configure(tracks[idx]["path"], start_sec=0.0)
      return

    current = self.playlist_view.playing_index
    if current is None:
      prev_idx = 0
    elif self.is_shuffle and len(tracks) > 1:
      candidates = [i for i in range(len(tracks)) if i != current]
      prev_idx = random.choice(candidates)
    else:
      prev_idx = (current - 1) % len(tracks)

    self.play_track_at_index(prev_idx)

  def on_next_track(self, auto_advance: bool = False):
    tracks = self.playlist_view.tracks
    if not tracks:
      return

    current = self.playlist_view.playing_index

    if (
        auto_advance
        and self.is_repeat
        and current is not None
        and 0 <= current < len(tracks)
    ):
      self.play_track_at_index(current)
      return

    if current is None:
      next_idx = 0
    elif self.is_shuffle and len(tracks) > 1:
      candidates = [i for i in range(len(tracks)) if i != current]
      next_idx = random.choice(candidates)
    else:
      if current + 1 < len(tracks):
        next_idx = current + 1
      elif not auto_advance:
        next_idx = 0
      else:
        self.media_player.stop()
        return

    self.play_track_at_index(next_idx)

  def on_playback_state_changed(self, state: QMediaPlayer.PlaybackState):
    self.is_playing = state == QMediaPlayer.PlaybackState.PlayingState
    self.sync_play_pause_sprite()
    if state == QMediaPlayer.PlaybackState.StoppedState:
      self.set_progress_thumb_x(self.prog_min_x)
      self._update_timer_display(0)

  def on_media_status_changed(self, status: QMediaPlayer.MediaStatus):
    if status in (
        QMediaPlayer.MediaStatus.LoadedMedia,
        QMediaPlayer.MediaStatus.BufferedMedia,
    ):
      seek_pos = self._pending_seek_pos_ms
      resume_play = self._pending_resume_play
      if seek_pos is not None:
        self._pending_seek_pos_ms = None
        self.media_player.setPosition(seek_pos)
        self._update_timer_display(seek_pos)
      if resume_play:
        self._pending_resume_play = False
        self.media_player.play()
      return

    if status == QMediaPlayer.MediaStatus.EndOfMedia:
      current = self.playlist_view.playing_index
      if (
          self.is_repeat
          and current is not None
          and 0 <= current < len(self.playlist_view.tracks)
      ):
        self.media_player.setPosition(0)
        self.analyzer_thread.configure(
            self.playlist_view.tracks[current]["path"], start_sec=0.0
        )
        self.media_player.play()
      else:
        self.on_next_track(auto_advance=True)

  def on_player_duration_changed(self, duration_ms: int):
    self.track_duration_ms = max(0, duration_ms)
    idx = self.playlist_view.playing_index
    if idx is not None and duration_ms > 0:
      self.playlist_view.update_track_duration(idx, duration_ms / 1000.0)

  def on_player_position_changed(self, position_ms: int):
    if self.is_scrubbing_progress:
      return
    self._update_timer_display(position_ms)
    if self.track_duration_ms > 0:
      ratio = max(0.0, min(1.0, position_ms / self.track_duration_ms))
      span = self.prog_max_x - self.prog_min_x
      target_x = self.prog_min_x + int(round(ratio * span))
      self.set_progress_thumb_x(target_x)
    else:
      self.set_progress_thumb_x(self.prog_min_x)

  def set_repeat_enabled(self, enabled: bool):
    self.is_repeat = bool(enabled)
    if self.is_repeat and self.is_shuffle:
      self.is_shuffle = False
      self.btn_shuffle.set_active(False)
    self.btn_repeat.set_active(self.is_repeat)

  def set_shuffle_enabled(self, enabled: bool):
    self.is_shuffle = bool(enabled)
    if self.is_shuffle and self.is_repeat:
      self.is_repeat = False
      self.btn_repeat.set_active(False)
    self.btn_shuffle.set_active(self.is_shuffle)

  def toggle_repeat(self):
    self.set_repeat_enabled(not self.is_repeat)

  def toggle_shuffle(self):
    self.set_shuffle_enabled(not self.is_shuffle)

  def on_add_songs(self):
    files, _ = QFileDialog.getOpenFileNames(
        self,
        "Додати музику",
        self.base_dir,
        "Аудіофайли (*.mp3 *.wav *.ogg *.flac *.m4a *.opus *.aac);;Усі файли (*)",
    )
    if files:
      self.playlist_view.add_tracks(files, auto_sort_alpha=True)
      self.save_playlist()

  def on_remove_songs(self):
    playing_was_removed = self.playlist_view.remove_selected_tracks()
    self.save_playlist()
    if playing_was_removed:
      self._stop_active_playback_and_reset_ui()

  def _draw_sprite_timer(self, painter: QPainter):
    t = self.current_time_str
    if len(t) != 5 or t[2] != ":":
      t = "00:00"

    chars = [t[0], t[1], ":", t[3], t[4]]
    for ch, (tx, ty) in zip(chars, self.TIMER_POSITIONS):
      px = self.digit_sprites.get(ch)
      if px is not None and not px.isNull():
        painter.drawPixmap(tx, ty, px)

  def _draw_visualizer(self, painter: QPainter):
    if self.vis_block_px.isNull():
      return

    for col_idx, x_pos in enumerate(self.VIS_COLS_X):
      active_count = int(round(self.vis_levels[col_idx]))
      active_count = max(0, min(len(self.VIS_ROWS_Y_BOTTOM_UP), active_count))
      for row_idx in range(active_count):
        y_pos = self.VIS_ROWS_Y_BOTTOM_UP[row_idx]
        painter.drawPixmap(x_pos, y_pos, self.vis_block_px)

  def paintEvent(self, event):
    painter = QPainter(self)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
    painter.resetTransform()

    painter.drawPixmap(0, 0, self.bg_pixmap)
    self._draw_sprite_timer(painter)
    self._draw_visualizer(painter)

    if not self.vol_bar_px.isNull():
      painter.drawPixmap(264, 131, self.vol_bar_px)

  def mousePressEvent(self, event):
    if event.button() == Qt.MouseButton.LeftButton:
      local_pos = event.position().toPoint()

      if self.search_bar.hasFocus() and not self.search_bar.geometry().contains(
          local_pos
      ):
        self.search_bar.clearFocus()

      if self.titlebar_rect.contains(local_pos):
        self.is_dragging = True
        global_pos = event.globalPosition().toPoint()
        self.drag_offset = QPoint(
            global_pos.x() - self.x(), global_pos.y() - self.y()
        )
        event.accept()
        return

      if self.prog_track_click_rect.contains(local_pos):
        self.is_scrubbing_progress = True
        centered_x = local_pos.x() - (self.prog_thumb_w // 2)
        self.set_progress_thumb_x(centered_x)
        if self.track_duration_ms > 0:
          ratio = self._thumb_x_to_ratio(self.prog_thumb.x())
          self._update_timer_display(int(ratio * self.track_duration_ms))
        event.accept()
        return

      if self.vol_track_click_rect.contains(local_pos):
        self.is_scrubbing_volume = True
        centered_x = local_pos.x() - (self.vol_thumb_w // 2)
        self.set_volume_thumb_x(centered_x)
        event.accept()
        return

    super().mousePressEvent(event)

  def mouseMoveEvent(self, event):
    if self.is_dragging and (event.buttons() & Qt.MouseButton.LeftButton):
      global_pos = event.globalPosition().toPoint()
      self.move(
          int(global_pos.x() - self.drag_offset.x()),
          int(global_pos.y() - self.drag_offset.y()),
      )
      event.accept()
      return

    if self.is_scrubbing_progress and (
        event.buttons() & Qt.MouseButton.LeftButton
    ):
      local_x = int(event.position().x())
      centered_x = local_x - (self.prog_thumb_w // 2)
      self.set_progress_thumb_x(centered_x)
      if self.track_duration_ms > 0:
        ratio = self._thumb_x_to_ratio(self.prog_thumb.x())
        self._update_timer_display(int(ratio * self.track_duration_ms))
      event.accept()
      return

    if self.is_scrubbing_volume and (event.buttons() & Qt.MouseButton.LeftButton):
      local_x = int(event.position().x())
      centered_x = local_x - (self.vol_thumb_w // 2)
      self.set_volume_thumb_x(centered_x)
      event.accept()
      return

    super().mouseMoveEvent(event)

  def mouseReleaseEvent(self, event):
    if event.button() == Qt.MouseButton.LeftButton:
      if self.is_dragging:
        self.is_dragging = False
        event.accept()
        return
      if self.is_scrubbing_progress:
        self._seek_to_thumb_x(self.prog_thumb.x())
        event.accept()
        return
      if self.is_scrubbing_volume:
        self.is_scrubbing_volume = False
        event.accept()
        return
    super().mouseReleaseEvent(event)

  def keyPressEvent(self, event):
    if self.search_bar.hasFocus():
      super().keyPressEvent(event)
      return

    if event.key() == Qt.Key.Key_Escape:
      if self.is_options_open:
        self.set_options_open(False)
      event.accept()
      return
    elif event.key() == Qt.Key.Key_Space:
      self.toggle_play_pause()
    elif event.key() == Qt.Key.Key_Delete:
      self.on_remove_songs()

  def closeEvent(self, event):
    self.save_playlist()
    self.analyzer_thread.stop_stream()
    self.analyzer_thread.wait(500)
    super().closeEvent(event)


if __name__ == "__main__":
  app = QApplication(sys.argv)
  player = YucPlayer()
  player.show()
  sys.exit(app.exec())
