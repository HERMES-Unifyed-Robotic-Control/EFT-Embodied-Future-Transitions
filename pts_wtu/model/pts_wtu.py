"""PredictiveWTU: history+lang → U → visual/state(/action) decoders (plan §9, §49)."""
from __future__ import annotations

from torch import nn

from pts_wtu.encoders.history_state import HistoryStateEncoder
from pts_wtu.encoders.history_visual import HistoryVisualEncoder
from pts_wtu.encoders.language import FrozenLanguageEncoder
from pts_wtu.model.action_head import ActionHead
from pts_wtu.model.context_fusion import ContextFusion
from pts_wtu.model.state_decoder import StateTransitionDecoder
from pts_wtu.model.u_predictor import TemporalQueryUPredictor
from pts_wtu.model.visual_decoder import VisualTransitionDecoder


class PredictiveWTU(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get("model", cfg)
        self.enc_v = HistoryVisualEncoder(
            d_model=int(m.get("d_context", 512)),
            n_layers=int(m.get("hist_visual_layers", 4)),
            n_heads=int(m.get("hist_visual_heads", 8)),
            n_views=int(m.get("n_views", 2)),
        )
        self.enc_s = HistoryStateEncoder(d_model=int(m.get("d_context", 512)))
        self.enc_l = FrozenLanguageEncoder(d_model=int(m.get("d_context", 512)))
        self.fuse = ContextFusion()
        self.u_pred = TemporalQueryUPredictor(
            d_model=int(m.get("d_u", 512)),
            K=int(m.get("K", 4)),
            n_layers=int(m.get("u_cross_layers", 2)),
            n_heads=int(m.get("u_cross_heads", 8)),
        )
        self.dec_v = VisualTransitionDecoder(d_u=int(m.get("d_u", 512)))
        self.dec_s = StateTransitionDecoder(
            d_u=int(m.get("d_u", 512)),
            hidden=int(m.get("state_decoder_hidden", 1024)),
            n_steps=int(m.get("state_samples_per_interval", 4)),
        )
        self.dec_a = ActionHead(d_u=int(m.get("d_u", 512)))

    def encode_u(self, batch: dict):
        """Predict U from history + language only."""
        raise NotImplementedError("PredictiveWTU.encode_u")

    def forward(self, batch: dict, stage: str = "a") -> dict:
        raise NotImplementedError("PredictiveWTU.forward")
