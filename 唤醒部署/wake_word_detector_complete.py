
"""
完整独立的唤醒词检测脚本
包含所有必要的模型定义，无需外部依赖
"""

import torch
import librosa
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import sounddevice as sd
import logging
from collections import deque
import time
import sys

# 使用 rich 库
try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.progress import BarColumn, Progress
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
except ImportError:
    print("正在安装 rich...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.progress import BarColumn, Progress
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== STFT Implementation ====================
class STFT(nn.Module):
    """使用 torchaudio 替代原来的 tools.torch_stft"""
    def __init__(self, filter_length=512, hop_length=256):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        
    def transform(self, input_waveform):
        """
        Args:
            input_waveform: [batch, time]
        Returns:
            magnitude: [batch, time_frames, freq_bins, 2]
        """
        # 使用 torch 内置的 stft
        spec = torch.stft(
            input_waveform,
            n_fft=self.filter_length,
            hop_length=self.hop_length,
            win_length=self.filter_length,
            window=torch.hann_window(self.filter_length).to(input_waveform.device),
            return_complex=False,
            center=True
        )
        # spec shape: [batch, freq, time, 2]
        # 转换为 [batch, time, freq, 2]
        spec = spec.permute(0, 2, 1, 3)
        return spec


# ==================== Model Components ====================
class Fbank(nn.Module):
    def __init__(self, sample_rate=16000, filter_length=512, hop_length=256, n_mels=64):
        super(Fbank, self).__init__()
        self.stft = STFT(filter_length, hop_length)
        self.alpha = nn.Parameter(torch.FloatTensor(1, 257))
        nn.init.constant_(self.alpha, 3)
        
        self.linear_to_mel_weight_matrix = torch.from_numpy(
            librosa.filters.mel(
                sr=sample_rate,
                n_fft=filter_length,
                n_mels=n_mels,
                fmin=20,
                fmax=8000,
                htk=True,
                norm=None
            ).T.astype(np.float32)
        )
    
    def forward(self, input_waveform):
        with torch.no_grad():
            spec = self.stft.transform(input_waveform)
            mag = (spec ** 2).sum(-1).sqrt()
        
        abs_mel = torch.matmul(mag, self.linear_to_mel_weight_matrix.to(input_waveform.device))
        abs_mel = abs_mel + 1e-6
        log_mel = abs_mel.log()
        log_mel[log_mel < -6] = -6
        return log_mel


class DSDilatedConv1d(nn.Module):
    """Dilated Depthwise-Separable Convolution"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1, bias=False):
        super(DSDilatedConv1d, self).__init__()
        self.receptive_fields = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            padding=0, dilation=dilation, stride=stride,
            groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv1d(
            in_channels, out_channels // 2,
            kernel_size=1, padding=0, dilation=1, bias=bias
        )

    def forward(self, inputs):
        outputs = self.conv(inputs)
        outputs = self.pointwise(outputs)
        return outputs


class TCNBlock(nn.Module):
    def __init__(self, in_channels, res_channels, kernel_size, dilation, causal):
        super(TCNBlock, self).__init__()
        self.in_channels = in_channels
        self.res_channels = res_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.causal = causal
        self.receptive_fields = dilation * (kernel_size - 1)
        self.half_receptive_fields = self.receptive_fields // 2
        
        self.conv1 = DSDilatedConv1d(
            in_channels=in_channels,
            out_channels=res_channels,
            kernel_size=kernel_size,
            dilation=dilation
        )
        self.prelu1 = nn.PReLU(res_channels // 2)
        self.conv2 = nn.Conv1d(
            in_channels=res_channels // 2,
            out_channels=res_channels,
            kernel_size=1
        )
        self.prelu2 = nn.PReLU(res_channels)

    def forward(self, xs, xs_lens=None, cnn_cache=None, is_last_cache=False):
        if cnn_cache is None:
            cnn_cache = torch.zeros(
                [xs.shape[0], xs.shape[1], self.receptive_fields],
                dtype=xs.dtype, device=xs.device
            )
        
        inputs = torch.cat((cnn_cache, xs), dim=-1)
        
        if xs_lens is None or is_last_cache:
            new_cache = inputs[:, :, -self.receptive_fields:]
        else:
            new_cache = []
            for i, xs_len in enumerate(xs_lens):
                c = inputs[i:i+1, :, xs_len:xs_len+self.receptive_fields]
                new_cache.append(c)
            new_cache = torch.cat(new_cache, axis=0)

        outputs1 = self.prelu1(self.conv1(inputs))
        outputs2 = self.conv2(outputs1)
        inputs = inputs[:, :, self.receptive_fields:]
        
        if self.in_channels == self.res_channels:
            res_out = self.prelu2(outputs2 + inputs)
        else:
            res_out = self.prelu2(outputs2)
        
        return res_out, new_cache.detach(), [outputs1, outputs2, res_out]


class TCNStack(nn.Module):
    def __init__(self, in_channels, stack_num, res_channels, kernel_size, causal):
        super(TCNStack, self).__init__()
        assert causal is True
        self.in_channels = in_channels
        self.stack_num = stack_num
        self.res_channels = res_channels
        self.kernel_size = kernel_size
        self.causal = causal
        self.res_blocks = self.stack_tcn_blocks()
        self.receptive_fields = self.calculate_receptive_fields()
        self.res_blocks = nn.Sequential(*self.res_blocks)

    def calculate_receptive_fields(self):
        receptive_fields = 0
        for block in self.res_blocks:
            receptive_fields += block.receptive_fields
        return receptive_fields

    def build_dilations(self):
        dilations = []
        for l in range(0, self.stack_num):
            dilations.append(2**l)
        return dilations

    def stack_tcn_blocks(self):
        dilations = self.build_dilations()
        res_blocks = nn.ModuleList()

        res_blocks.append(
            TCNBlock(self.in_channels, self.res_channels,
                    self.kernel_size, dilations[0], self.causal)
        )
        for dilation in dilations[1:]:
            res_blocks.append(
                TCNBlock(self.res_channels, self.res_channels,
                        self.kernel_size, dilation, self.causal)
            )
        return res_blocks

    def forward(self, xs, xs_lens=None, cnn_caches=None, is_last_cache=False):
        new_caches = []
        out_list_for_loss = []
        for block, cnn_cache in zip(self.res_blocks, cnn_caches):
            xs, new_cache, out_l = block(xs, xs_lens, cnn_cache, is_last_cache=is_last_cache)
            new_caches.append(new_cache)
            out_list_for_loss += out_l
        return xs, new_caches, out_list_for_loss


class MDTCSML(nn.Module):
    """Multi-scale Depthwise Temporal Convolution for Wake Word Detection"""
    def __init__(self, stack_num, stack_size, in_channels, res_channels, kernel_size, causal, shift=256):
        super(MDTCSML, self).__init__()
        self.fbank = Fbank(sample_rate=16000, filter_length=512, hop_length=shift, n_mels=64)
        self.stack_num = stack_num
        self.stack_size = stack_size
        self.kernel_size = kernel_size
        self.causal = causal
        self.shift = shift
        
        self.preprocessor = TCNBlock(
            in_channels, res_channels, kernel_size, dilation=1, causal=causal
        )
        self.prelu = nn.PReLU(res_channels)
        self.blocks = nn.ModuleList()
        self.receptive_fields = []
        self.receptive_fields.append(self.preprocessor.receptive_fields)
        
        for _ in range(stack_num):
            self.blocks.append(
                TCNStack(res_channels, stack_size, res_channels, kernel_size, causal)
            )
            self.receptive_fields.append(self.blocks[-1].receptive_fields)
        
        self.class_out = torch.nn.Linear(res_channels, 2)

    def forward(self, wav, kw_target=None, ckw_target=None, real_frames=None, 
                label_frames=None, ckw_len=None, clean_speech=None, hidden=None, custom_in=None):
        if hidden is None:
            hidden = [None for _ in range(self.stack_size * self.stack_num + 1)]
        else:
            if hidden[0].shape[0] >= wav.shape[0]:
                b = wav.size(0)
                h_l = []
                for h in hidden:
                    h_l.append(h[:b])
                hidden = h_l
            else:
                hidden = [None for _ in range(self.stack_size * self.stack_num + 1)]
        
        xs = self.fbank(wav)
        b, t, f = xs.size()
        
        if random.random() < 0.5 and self.training:
            is_last_cache = True
        else:
            is_last_cache = False

        outputs = xs.transpose(1, 2)
        outputs_list = []
        outputs_list_for_loss = []
        outputs_cache_list = []
        
        outputs, new_cache, o_l_1 = self.preprocessor(outputs, real_frames, hidden[0], is_last_cache=is_last_cache)
        outputs_list_for_loss += o_l_1
        outputs = self.prelu(outputs)
        outputs_list_for_loss.append(outputs)
        outputs_pre = outputs
        outputs_cache_list.append(new_cache)
        
        for i in range(len(self.blocks)):
            outputs, new_caches, o_l_tmp = self.blocks[i](
                outputs, real_frames, 
                hidden[1+i*self.stack_size: 1+(i+1)*self.stack_size],
                is_last_cache=is_last_cache
            )
            outputs_list_for_loss += o_l_tmp
            outputs_list.append(outputs)
            outputs_cache_list += new_caches

        outputs = sum(outputs_list)
        outputs = outputs.transpose(1, 2)
        logist = self.class_out(outputs)

        # 推理模式下不需要 loss
        return logist, None, outputs_cache_list, None, None, None, None


# ==================== Wake Word Detector ====================
class WakeWordDetector:
    def __init__(self, model_path, sample_rate=16000, chunk_size=1024, 
                 window_duration=1.0, threshold=0.5, use_rich=True):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.threshold = threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_rich = use_rich
        
        self.window_size = int(sample_rate * window_duration)
        self.audio_buffer = deque(maxlen=self.window_size)
        
        self.model = self.load_model(model_path)
        self.running = False
        
        # Rich 相关
        self.console = Console()
        self.log_lines = deque(maxlen=15)
        self.current_volume = 0.0
        self.current_confidence = 0.0
        
        logger.info(f"Wake word detector initialized")
        logger.info(f"Device: {self.device}")
        logger.info(f"Sample rate: {sample_rate}Hz")
        logger.info(f"Window duration: {window_duration}s")
        logger.info(f"Threshold: {threshold}")
        logger.info(f"Available audio devices:")
        print(sd.query_devices())
    
    def load_model(self, model_path):
        logger.info(f"Loading model from: {model_path}")
        
        model = MDTCSML(
            stack_num=4, stack_size=4, in_channels=64,
            res_channels=128, kernel_size=7, causal=True
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            step = checkpoint.get('step', 'unknown')
            logger.info(f"Loaded checkpoint from step: {step}")
        else:
            state_dict = checkpoint
        
        # 处理 DDP 的 'module.' 前缀
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        model.to(self.device)
        model.eval()
        
        logger.info("Model loaded successfully!")
        return model
    
    def add_log(self, message):
        """添加日志到显示队列"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_lines.append(f"[{timestamp}] {message}")
    
    def generate_display(self):
        """生成Rich显示布局"""
        # 创建音量条
        volume_percent = min(self.current_volume * 500, 100)
        bar_filled = int(volume_percent / 2)  # 50个字符
        volume_bar = "█" * bar_filled + "░" * (50 - bar_filled)
        
        # 创建表格
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style="cyan", width=15)
        table.add_column(style="white")
        
        table.add_row("音量", f"[{volume_bar}] {self.current_volume:.4f}")
        table.add_row("检测阈值", f"{self.threshold}")
        table.add_row("当前置信度", f"{self.current_confidence:.3f}")
        
        # 创建日志面板
        log_text = "\n".join(list(self.log_lines)[-12:])  # 最多显示12行
        logs_panel = Panel(
            log_text if log_text else "等待日志...",
            title="[bold cyan]日志输出[/bold cyan]",
            border_style="blue"
        )
        
        # 组合布局
        layout = Table.grid(padding=1)
        layout.add_column()
        
        layout.add_row(
            Panel(
                table,
                title="[bold green]🎤 唤醒词实时检测[/bold green]",
                border_style="green"
            )
        )
        layout.add_row(logs_panel)
        layout.add_row("[dim]按 Ctrl+C 退出[/dim]")
        
        return layout
    
    def preprocess_audio(self, audio_data):
        audio_array = np.array(audio_data, dtype=np.float32)
        audio_tensor = torch.FloatTensor(audio_array).unsqueeze(0)
        audio_tensor = audio_tensor.to(self.device)
        return audio_tensor
    
    def predict(self, audio_features):
        with torch.no_grad():
            logist, *_ = self.model(audio_features)
            probs = torch.softmax(logist, dim=-1)
            
            # 获取唤醒词的概率（跳过背景类0）
            wake_word_prob = torch.amax(probs[:, :, 1:], dim=(1, 2))
            confidence = float(wake_word_prob.cpu().item())
            
        return confidence
    
    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"\n⚠️ Audio status: {status}")
        
        audio_chunk = indata[:, 0]
        self.current_volume = np.abs(audio_chunk).mean()
        
        self.audio_buffer.extend(audio_chunk)
        
        if len(self.audio_buffer) >= self.window_size:
            audio_window = list(self.audio_buffer)
            
            try:
                features = self.preprocess_audio(audio_window)
                confidence = self.predict(features)
                self.current_confidence = confidence
                
                if confidence > self.threshold:
                    # 检测到唤醒词时换行显示
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"\n[{timestamp}] 🎯 WAKE WORD DETECTED! (Confidence: {confidence:.3f})")
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def run(self):
        """Main loop to run wake word detection"""
        if self.use_rich:
            self._run_with_rich()
        else:
            self._run_without_rich()
    
    def _run_with_rich(self):
        """使用简洁的静态界面"""
        print("\n" + "="*60)
        print("           🎤 唤醒词实时检测")
        print("="*60)
        print(f"检测阈值: {self.threshold}")
        print("="*60)
        print()
        
        try:
            self.running = True
            
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                blocksize=self.chunk_size,
                callback=self.audio_callback
            ):
                while self.running:
                    # 只更新音量条这一行
                    volume_percent = min(self.current_volume * 500, 100)
                    bar_filled = int(volume_percent / 2)
                    volume_bar = "█" * bar_filled + "░" * (50 - bar_filled)
                    
                    print(f"\r音量: [{volume_bar}] {self.current_volume:.4f} | 置信度: {self.current_confidence:.3f}", 
                          end='', flush=True)
                    time.sleep(0.05)
                    
        except KeyboardInterrupt:
            print("\n\n停止检测...")
        finally:
            self.running = False
    
    def _run_without_rich(self):
        """不使用Rich的普通模式"""
        try:
            self.running = True
            logger.info("Starting audio stream - listening for wake word...")
            logger.info("Press Ctrl+C to stop...")
            
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                blocksize=self.chunk_size,
                callback=self.audio_callback
            ):
                while self.running:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            logger.info("\nStopping wake word detection...")
            self.running = False
        except Exception as e:
            logger.error(f"Error: {e}")
            self.running = False


def main():
    MODEL_PATH = r"E:\语音\唤醒部署\model-272000--92.1627806854248.pickle"
    SAMPLE_RATE = 16000
    CHUNK_SIZE = 1024
    WINDOW_DURATION = 1.0
    THRESHOLD = 0.5
    
    import os
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found: {MODEL_PATH}")
        return
    
    detector = WakeWordDetector(
        model_path=MODEL_PATH,
        sample_rate=SAMPLE_RATE,
        chunk_size=CHUNK_SIZE,
        window_duration=WINDOW_DURATION,
        threshold=THRESHOLD
    )
    
    detector.run()


if __name__ == "__main__":
    main()
