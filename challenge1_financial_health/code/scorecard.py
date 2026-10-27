"""Scorecard WoE + logistic regression, tự viết để không phụ thuộc thư viện ngoài sklearn."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

EPS = 0.5


def _bins_numeric(x, y, max_bins=6, min_share=0.05):
    """Chia theo phân vị rồi gộp bin liền kề cho tới khi WoE đơn điệu."""
    xs = x.dropna()
    if xs.nunique() <= max_bins:
        edges = np.unique(xs)
        edges = np.concatenate([[-np.inf], (edges[:-1] + edges[1:]) / 2, [np.inf]])
    else:
        q = np.unique(np.quantile(xs, np.linspace(0, 1, max_bins + 1)))
        edges = np.concatenate([[-np.inf], q[1:-1], [np.inf]])
    while len(edges) > 2:
        b = pd.cut(x, edges)
        t = pd.DataFrame({"b": b, "y": y}).dropna().groupby("b", observed=False).y.agg(["sum", "count"])
        rate = (t["sum"] + EPS) / (t["count"] + 2 * EPS)
        small = (t["count"] < min_share * len(xs)).to_numpy()
        diffs = np.diff(rate.to_numpy())
        mono = (diffs >= 0).all() or (diffs <= 0).all()
        if mono and not small.any():
            break
        # gộp cặp bin có tỷ lệ gần nhau nhất (hoặc bin quá nhỏ)
        i = int(np.argmin(np.abs(diffs))) if not small.any() else max(0, min(int(np.argmax(small)), len(diffs) - 1))
        edges = np.delete(edges, i + 1)
    return edges


class WoEScorecard:
    def __init__(self, numeric, categorical, max_bins=6, C=0.5, max_corr=0.6, edges=None):
        self.numeric, self.categorical, self.max_bins, self.C = numeric, categorical, max_bins, C
        self.max_corr = max_corr
        self.fixed_edges = edges or {}      # ngưỡng nghiệp vụ cố định cho biến hiếm, ví dụ {"n_due_total": [-inf, .5, 1.5, inf]}

    def _woe_table(self, codes, y):
        t = pd.DataFrame({"c": codes, "y": y}).groupby("c").y.agg(bad="sum", n="count")
        t["good"] = t.n - t.bad
        gb, gg = t.bad.sum(), t.good.sum()
        t["woe"] = np.log(((t.good + EPS) / (gg + EPS)) / ((t.bad + EPS) / (gb + EPS)))
        t["iv"] = ((t.good + EPS) / (gg + EPS) - (t.bad + EPS) / (gb + EPS)) * t.woe
        t["bad_rate"] = t.bad / t.n
        return t

    def _codes(self, df, col):
        if col in self.numeric:
            return pd.cut(df[col], self.edges_[col]).astype(str).where(df[col].notna(), "Thiếu")
        v = df[col].fillna("Thiếu").astype(str)
        return v.where(v.isin(self.keep_[col]), "Khác")

    def fit(self, df, y):
        y = pd.Series(np.asarray(y), index=df.index)
        self.edges_, self.keep_, self.tables_ = {}, {}, {}
        for col in self.numeric:
            self.edges_[col] = _bins_numeric(df[col], y, self.max_bins)
        for col in self.categorical:
            vc = df[col].fillna("Thiếu").astype(str).value_counts(normalize=True)
            self.keep_[col] = set(vc[vc >= 0.001].index)
        for col in self.numeric + self.categorical:
            self.tables_[col] = self._woe_table(self._codes(df, col), y)
        self.iv_ = pd.Series({c: t.iv.sum() for c, t in self.tables_.items()}).sort_values(ascending=False)
        self.features_ = [c for c in self.iv_.index if self.iv_[c] >= 0.02]
        # bỏ biến tương quan cao (|r| > max_corr trên WoE) với một biến có IV lớn hơn
        X = self.transform(df)
        corr = X.corr().abs()
        keep = []
        for col in self.features_:
            if all(corr.loc[col, k] <= self.max_corr for k in keep):
                keep.append(col)
        self.dropped_corr_ = [c for c in self.features_ if c not in keep]
        self.features_ = keep
        X = self.transform(df)
        self.lr_ = LogisticRegression(C=self.C, max_iter=1000).fit(X, y)
        # biến có hệ số sai dấu (WoE cao phải làm giảm xác suất xấu) thì bỏ và fit lại
        while (self.lr_.coef_[0] > 0).any():
            drop = self.features_[int(np.argmax(self.lr_.coef_[0]))]
            self.features_.remove(drop)
            X = self.transform(df)
            self.lr_ = LogisticRegression(C=self.C, max_iter=1000).fit(X, y)
        return self

    def fit_fixed(self, df, y, features, l2=1e-3):
        """Giữ đúng danh sách `features` (do người phân tích chọn theo từng trụ) và ước lượng trọng số
        bằng hồi quy logistic có ràng buộc: trọng số của -WoE không âm, nên mọi biến đều tác động đúng chiều."""
        from scipy.optimize import minimize
        y = pd.Series(np.asarray(y), index=df.index)
        self.edges_, self.keep_, self.tables_ = {}, {}, {}
        for col in self.numeric:
            self.edges_[col] = (np.array(self.fixed_edges[col], dtype=float) if col in self.fixed_edges
                                else _bins_numeric(df[col], y, self.max_bins))
        for col in self.categorical:
            vc = df[col].fillna("Thiếu").astype(str).value_counts(normalize=True)
            self.keep_[col] = set(vc[vc >= 0.001].index)
        for col in self.numeric + self.categorical:
            self.tables_[col] = self._woe_table(self._codes(df, col), y)
        self.iv_ = pd.Series({c: t.iv.sum() for c, t in self.tables_.items()}).sort_values(ascending=False)
        self.features_ = list(features)
        X = self.transform(df).to_numpy()
        yv = y.to_numpy().astype(float)

        def nll(w):
            z = w[0] - X @ w[1:]
            p = 1 / (1 + np.exp(-z))
            ll = -(yv * np.log(p + 1e-12) + (1 - yv) * np.log(1 - p + 1e-12)).mean() + l2 * (w[1:] ** 2).sum()
            g_z = p - yv
            grad = np.concatenate([[g_z.mean()], -(X * g_z[:, None]).mean(axis=0) + 2 * l2 * w[1:]])
            return ll, grad

        w0 = np.concatenate([[np.log(yv.mean() / (1 - yv.mean()))], np.full(X.shape[1], 0.5)])
        r = minimize(nll, w0, jac=True, method="L-BFGS-B", bounds=[(None, None)] + [(0, None)] * X.shape[1])

        class _LR:  # giao diện giống sklearn để các hàm khác dùng lại
            pass
        self.lr_ = _LR()
        self.lr_.intercept_ = np.array([r.x[0]])
        self.lr_.coef_ = np.array([-r.x[1:]])
        self.lr_.predict_proba = lambda Xd: np.c_[1 - 1 / (1 + np.exp(-(r.x[0] - np.asarray(Xd) @ r.x[1:]))),
                                                   1 / (1 + np.exp(-(r.x[0] - np.asarray(Xd) @ r.x[1:])))]
        return self

    def transform(self, df):
        out = {}
        for col in self.features_:
            woe = self.tables_[col].woe
            out[col] = self._codes(df, col).map(woe).fillna(0.0)
        return pd.DataFrame(out, index=df.index)

    def predict_pd(self, df):
        return self.lr_.predict_proba(self.transform(df))[:, 1]

    def points_table(self, factor, offset_per_var):
        """Điểm cộng/trừ cho từng bin: points = -(coef * woe) * factor + offset chia đều."""
        rows = []
        for col, coef in zip(self.features_, self.lr_.coef_[0]):
            t = self.tables_[col]
            for b, r in t.iterrows():
                rows.append(dict(bien=col, khoang=b, so_khach=int(r.n), ty_le_xau=r.bad_rate,
                                 woe=r.woe, diem=-coef * r.woe * factor + offset_per_var))
        return pd.DataFrame(rows)


def ks(y, s):
    d = pd.DataFrame({"y": y, "s": s}).sort_values("s")
    cb = (d.y == 1).cumsum() / max((d.y == 1).sum(), 1)
    cg = (d.y == 0).cumsum() / max((d.y == 0).sum(), 1)
    return float((cb - cg).abs().max())


def psi(a, b, bins=10):
    q = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    q[0], q[-1] = -np.inf, np.inf
    pa = np.histogram(a, q)[0] / len(a) + 1e-6
    pb = np.histogram(b, q)[0] / len(b) + 1e-6
    return float(((pa - pb) * np.log(pa / pb)).sum())


def metrics(y, score_good):
    """score_good: điểm càng cao càng tốt."""
    auc = roc_auc_score(y, -score_good)
    return dict(AUC=auc, Gini=2 * auc - 1, KS=ks(y, -score_good))
