import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from talib import WMA
from dcor import distance_correlation as dcor
from joblib import Parallel, delayed, Memory, cpu_count
from numpy_ext import rolling_apply
import numpy as np
from fml.utils import get_name


def rank(df):
    return df.rank(axis=1, pct=True)


def scale(df):
    return df.div(df.abs().sum(axis=1), axis=0)


def log(df):
    return np.log1p(df)


def sign(df):
    return np.sign(df)


def power(df, exp):
    return df.pow(exp)


def ts_lag(df: pd.DataFrame, t: int = 1) -> pd.DataFrame:
    return df.shift(t)


def ts_delta(df, period=1):
    return df.diff(period)


def ts_sum(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    return df.rolling(window).sum()


def ts_mean(df, window=10):
    return df.rolling(window).mean()


def ts_weighted_mean(df, period=10):
    return (df.apply(lambda x: WMA(x, timeperiod=period)))


def ts_std(df, window=10):
    return (df
            .rolling(window)
            .std())


def ts_rank(df, window=10):
    return (df
            .rolling(window)
            .apply(lambda x: x.rank().iloc[-1]))


def ts_product(df, window=10):
    return (df
            .rolling(window)
            .apply(np.prod))


def ts_min(df, window=10):
    return df.rolling(window).min()


def ts_max(df, window=10):
    return df.rolling(window).max()


def ts_argmax(df, window=10):
    return df.rolling(window).apply(np.argmax).add(1)


def ts_argmin(df, window=10):
    return (df.rolling(window)
            .apply(np.argmin)
            .add(1))


def ts_corr(x, y, window):
    res = Parallel(n_jobs=cpu_count())(delayed(rolling_apply)(dcor, window, x[col], y[col]) for col in x)
    res = pd.DataFrame.from_dict(dict(zip(x.columns, res)))
    res.index = x.index
    res.columns.name = 'ticker'
    return res

# def ts_corr(x, y, window=10):
    # res = x.rolling(window).corr(y)
    # return res

def ts_cov(x, y, window=10):
    return x.rolling(window).cov(y)


def get_mutual_info_score(returns, alpha, n=100000):
    df = pd.DataFrame({'y': returns, 'alpha': alpha}).dropna().sample(n=n)
    return mutual_info_regression(y=df.y, X=df[['alpha']])[0]


def alpha001(c, r):
    c[r < 0] = ts_std(r, 20)
    return (rank(ts_argmax(power(c, 2), 5)).mul(-.5)
            .stack().swaplevel())


def alpha002(o, c, v):
    s1 = rank(ts_delta(log(v), 2))
    s2 = rank((c / o) - 1)
    alpha = -ts_corr(s1, s2, 6)
    res = alpha.stack('ticker').swaplevel().replace([-np.inf, np.inf], np.nan)
    return res


def alpha003(o, v):
    return (-ts_corr(rank(o), rank(v), 10)
            .stack('ticker')
            .swaplevel()
            .replace([-np.inf, np.inf], np.nan))


def alpha004(l):
    return (-ts_rank(rank(l), 9)
            .stack('ticker')
            .swaplevel())


def alpha005(o, vwap, c):
    return (rank(o.sub(ts_mean(vwap, 10)))
            .mul(rank(c.sub(vwap)).mul(-1).abs())
            .stack('ticker')
            .swaplevel())


def alpha006(o, v):
    return (-ts_corr(o, v, 10)
            .stack('ticker')
            .swaplevel())


def alpha007(c, v, adv20):
    delta7 = ts_delta(c, 7)
    return (-ts_rank(abs(delta7), 60)
            .mul(sign(delta7))
            .where(adv20 < v, -1)
            .stack('ticker')
            .swaplevel())


def alpha008(o, r):
    return (-(rank(((ts_sum(o, 5) * ts_sum(r, 5)) -
                    ts_lag((ts_sum(o, 5) * ts_sum(r, 5)), 10))))
            .stack('ticker')
            .swaplevel())


def alpha009(c):
    close_diff = ts_delta(c, 1)
    alpha = close_diff.where(ts_min(close_diff, 5) > 0,
                             close_diff.where(ts_max(close_diff, 5) < 0,
                                              -close_diff))
    return (alpha
            .stack('ticker')
            .swaplevel())


def alpha010(c):
    close_diff = ts_delta(c, 1)
    alpha = close_diff.where(ts_min(close_diff, 4) > 0,
                             close_diff.where(ts_min(close_diff, 4) > 0,
                                              -close_diff))

    return (rank(alpha)
            .stack('ticker')
            .swaplevel())


def alpha011(c, vwap, v):
    return (rank(ts_max(vwap.sub(c), 3))
            .add(rank(ts_min(vwap.sub(c), 3)))
            .mul(rank(ts_delta(v, 3)))
            .stack('ticker')
            .swaplevel())


def alpha012(v, c):
    return (sign(ts_delta(v, 1)).mul(-ts_delta(c, 1))
            .stack('ticker')
            .swaplevel())


def alpha013(c, v):
    return (-rank(ts_cov(rank(c), rank(v), 5))
            .stack('ticker')
            .swaplevel())


def alpha014(o, v, r):
    alpha = -rank(ts_delta(r, 3)).mul(ts_corr(o, v, 10)
                                      .replace([-np.inf,
                                                np.inf],
                                               np.nan))
    return (alpha
            .stack('ticker')
            .swaplevel())


def alpha015(h, v):
    alpha = (-ts_sum(rank(ts_corr(rank(h), rank(v), 3)
                          .replace([-np.inf, np.inf], np.nan)), 3))
    return (alpha
            .stack('ticker')
            .swaplevel())


def alpha016(h, v):
    return (-rank(ts_cov(rank(h), rank(v), 5))
            .stack('ticker')
            .swaplevel())


def alpha017(c, v):
    adv20 = ts_mean(v, 20)
    return (-rank(ts_rank(c, 10))
            .mul(rank(ts_delta(ts_delta(c, 1), 1)))
            .mul(rank(ts_rank(v.div(adv20), 5)))
            .stack('ticker')
            .swaplevel())


def alpha018(o, c):
    return (-rank(ts_std(c.sub(o).abs(), 5)
                  .add(c.sub(o))
                  .add(ts_corr(c, o, 10)
                       .replace([-np.inf,
                                 np.inf],
                                np.nan)))
            .stack('ticker')
            .swaplevel())


def alpha019(c, r):
    return (-sign(ts_delta(c, 7) + ts_delta(c, 7))
            .mul(1 + rank(1 + ts_sum(r, 250)))
            .stack('ticker')
            .swaplevel())


def alpha020(o, h, l, c):
    return (rank(o - ts_lag(h, 1))
            .mul(rank(o - ts_lag(c, 1)))
            .mul(rank(o - ts_lag(l, 1)))
            .mul(-1)
            .stack('ticker')
            .swaplevel())


def alpha021(c, v):
    sma2 = ts_mean(c, 2)
    sma8 = ts_mean(c, 8)
    std8 = ts_std(c, 8)

    cond_1 = sma8.add(std8) < sma2
    cond_2 = sma8.add(std8) > sma2
    cond_3 = v.div(ts_mean(v, 20)) < 1

    val = np.ones_like(c)
    alpha = pd.DataFrame(np.select(condlist=[cond_1, cond_2, cond_3],
                                   choicelist=[-1, 1, -1], default=1),
                         index=c.index,
                         columns=c.columns)

    return (alpha
            .stack('ticker')
            .swaplevel())


def alpha022(h, c, v):
    return (ts_delta(ts_corr(h, v, 5)
                     .replace([-np.inf,
                               np.inf],
                              np.nan), 5)
            .mul(rank(ts_std(c, 20)))
            .mul(-1)
            .stack('ticker')
            .swaplevel())


def alpha023(h, c):
    return (ts_delta(h, 2)
            .mul(-1)
            .where(ts_mean(h, 20) < h, 0)
            .stack('ticker')
            .swaplevel())


def alpha024(c):
    cond = ts_delta(ts_mean(c, 100), 100) / ts_lag(c, 100) <= 0.05

    return (c.sub(ts_min(c, 100)).mul(-1).where(cond, -ts_delta(c, 3))
            .stack('ticker')
            .swaplevel())


def alpha025(h, c, r, vwap, adv20):
    return (rank(-r.mul(adv20)
                 .mul(vwap)
                 .mul(h.sub(c)))
            .stack('ticker')
            .swaplevel())


def alpha026(h, v):
    return (ts_max(ts_corr(ts_rank(v, 5),
                           ts_rank(h, 5), 5)
                   .replace([-np.inf, np.inf], np.nan), 3)
            .mul(-1)
            .stack('ticker')
            .swaplevel())


def alpha027(v, vwap):
    cond = rank(ts_mean(ts_corr(rank(v),
                                rank(vwap), 6), 2))
    alpha = cond.notnull().astype(float)
    return (alpha.where(cond <= 0.5, -alpha)
            .stack('ticker')
            .swaplevel())


def alpha028(h, l, c, v, adv20):
    return (scale(ts_corr(adv20, l, 5)
                  .replace([-np.inf, np.inf], 0)
                  .add(h.add(l).div(2).sub(c)))
            .stack('ticker')
            .swaplevel())


def alpha029(c, r):
    return (ts_min(rank(rank(scale(log(ts_sum(rank(rank(-rank(ts_delta((c - 1), 5)))), 2))))), 5)
            .add(ts_rank(ts_lag((-1 * r), 6), 5))
            .stack('ticker')
            .swaplevel())


def alpha030(c, v):
    close_diff = ts_delta(c, 1)
    return (rank(sign(close_diff)
                 .add(sign(ts_lag(close_diff, 1)))
                 .add(sign(ts_lag(close_diff, 2))))
            .mul(-1).add(1)
            .mul(ts_sum(v, 5))
            .div(ts_sum(v, 20))
            .stack('ticker')
            .swaplevel())


def alpha031(l, c, adv20):
    return (rank(rank(rank(ts_weighted_mean(rank(rank(ts_delta(c, 10))).mul(-1), 10))))
            .add(rank(ts_delta(c, 3).mul(-1)))
            .add(sign(scale(ts_corr(adv20, l, 12)
                            .replace([-np.inf, np.inf],
                                     np.nan))))
            .stack('ticker')
            .swaplevel())


def alpha032(c, vwap):
    return (scale(ts_mean(c, 7).sub(c))
            .add(20 * scale(ts_corr(vwap,
                                    ts_lag(c, 5), 230)))
            .stack('ticker')
            .swaplevel())


def alpha033(o, c):
    return (rank(o.div(c).mul(-1).add(1).mul(-1))
            .stack('ticker')
            .swaplevel())


def alpha034(c, r):
    return (rank(rank(ts_std(r, 2).div(ts_std(r, 5))
                      .replace([-np.inf, np.inf],
                               np.nan))
                 .mul(-1)
                 .sub(rank(ts_delta(c, 1)))
                 .add(2))
            .stack('ticker')
            .swaplevel())


def alpha035(h, l, c, v, r):
    return (ts_rank(v, 32)
            .mul(1 - ts_rank(c.add(h).sub(l), 16))
            .mul(1 - ts_rank(r, 32))
            .stack('ticker')
            .swaplevel())


def alpha036(o, c, v, r, adv20, vwap):
    return (rank(ts_corr(c.sub(o), ts_lag(v, 1), 15)).mul(2.21)
            .add(rank(o.sub(c)).mul(.7))
            .add(rank(ts_rank(ts_lag(-r, 6), 5)).mul(0.73))
            .add(rank(abs(ts_corr(vwap, adv20, 6))))
            .add(rank(ts_mean(c, 200).sub(o).mul(c.sub(o))).mul(0.6))
            .stack('ticker')
            .swaplevel())


def alpha037(o, c):
    return (rank(ts_corr(ts_lag(o.sub(c), 1), c, 200))
            .add(rank(o.sub(c)))
            .stack('ticker')
            .swaplevel())


def alpha038(o, c):
    return (rank(ts_rank(o, 10))
            .mul(rank(c.div(o).replace([-np.inf, np.inf], np.nan)))
            .mul(-1)
            .stack('ticker')
            .swaplevel())


def alpha039(c, v, r, adv20):
    return (rank(ts_delta(c, 7).mul(rank(ts_weighted_mean(v.div(adv20), 9)).mul(-1).add(1))).mul(-1)
            .mul(rank(ts_mean(r, 250).add(1)))
            .stack('ticker')
            .swaplevel())


def alpha040(h, v):
    return (rank(ts_std(h, 10))
            .mul(ts_corr(h, v, 10))
            .mul(-1)
            .stack('ticker')
            .swaplevel())


def alpha041(h, l, vwap):
    return (power(h.mul(l), 0.5)
            .sub(vwap)
            .stack('ticker')
            .swaplevel())


def alpha042(c, vwap):
    return (rank(vwap.sub(c))
            .div(rank(vwap.add(c)))
            .stack('ticker')
            .swaplevel())


def alpha043(c, v, adv20):
    return (ts_rank(v.div(adv20), 20)
            .mul(ts_rank(ts_delta(c, 7).mul(-1), 8))
            .stack('ticker')
            .swaplevel())


def alpha044(h, v):
    return (ts_corr(h, rank(v), 5)
            .replace([-np.inf, np.inf], np.nan)
            .mul(-1)
            .stack('ticker')
            .swaplevel())


def alpha045(c, v):
    return (rank(ts_mean(ts_lag(c, 5), 20))
            .mul(ts_corr(c, v, 2)
                 .replace([-np.inf, np.inf], np.nan))
            .mul(rank(ts_corr(ts_sum(c, 5),
                              ts_sum(c, 20), 2)))
            .mul(-1)
            .stack('ticker')
            .swaplevel())


def alpha046(c):
    cond = ts_lag(ts_delta(c, 10), 10).div(10).sub(ts_delta(c, 10).div(10))
    alpha = pd.DataFrame(-np.ones_like(cond),
                         index=c.index,
                         columns=c.columns)
    alpha[cond.isnull()] = np.nan
    return (cond.where(cond > 0.25,
                       -alpha.where(cond < 0,
                                    -ts_delta(c, 1)))
            .stack('ticker')
            .swaplevel())


def alpha047(h, c, v, vwap, adv20):
    return (rank(c.pow(-1)).mul(v).div(adv20)
            .mul(h.mul(rank(h.sub(c))
                       .div(ts_mean(h, 5)))
                 .sub(rank(ts_delta(vwap, 5))))
            .stack('ticker')
            .swaplevel())


def alpha049(c):
    cond = (ts_delta(ts_lag(c, 10), 10).div(10)
            .sub(ts_delta(c, 10).div(10)) >= -0.1 * c)
    return (-ts_delta(c, 1)
            .where(cond, 1)
            .stack('ticker')
            .swaplevel())


def alpha050(v, vwap):
    return (ts_max(rank(ts_corr(rank(v),
                                rank(vwap), 5)), 5)
            .mul(-1)
            .stack('ticker')
            .swaplevel())


def alpha051(c):
    cond = (ts_delta(ts_lag(c, 10), 10).div(10)
            .sub(ts_delta(c, 10).div(10)) >= -0.05 * c)
    return (-ts_delta(c, 1)
            .where(cond, 1)
            .stack('ticker')
            .swaplevel())



def alpha052(l, v, r):
    return (ts_delta(ts_min(l, 5), 5)
            .mul(rank(ts_sum(r, 240)
                      .sub(ts_sum(r, 20))
                      .div(220)))
            .mul(ts_rank(v, 5))
            .stack('ticker')
            .swaplevel())


def alpha053(h, l, c):
    inner = (c.sub(l)).add(1e-6)
    return (ts_delta(h.sub(c)
                     .mul(-1).add(1)
                     .div(c.sub(l)
                          .add(1e-6)), 9)
            .mul(-1)
            .stack('ticker')
            .swaplevel())


def alpha054(o, h, l, c):
    return (l.sub(c).mul(o.pow(5)).mul(-1)
            .div(l.sub(h).replace(0, -0.0001).mul(c ** 5))
            .stack('ticker')
            .swaplevel())


def alpha055(h, l, c, v):
    return (ts_corr(rank(c.sub(ts_min(l, 12))
                         .div(ts_max(h, 12).sub(ts_min(l, 12))
                              .replace(0, 1e-6))),
                    rank(v), 6)
            .replace([-np.inf, np.inf], np.nan)
            .mul(-1)
            .stack('ticker')
            .swaplevel())


def alpha057(c, vwap):
    return (c.sub(vwap.add(1e-5))
            .div(ts_weighted_mean(rank(ts_argmax(c, 30)))).mul(-1)
            .stack('ticker')
            .swaplevel())


def alpha060(l, h, c, v):
    return (scale(rank(c.mul(2).sub(l).sub(h)
                       .div(h.sub(l).replace(0, 1e-5))
                       .mul(v))).mul(2)
            .sub(scale(rank(ts_argmax(c, 10)))).mul(-1)
            .stack('ticker')
            .swaplevel())


def alpha061(v, vwap):
    return (rank(vwap.sub(ts_min(vwap, 16)))
            .lt(rank(ts_corr(vwap, ts_mean(v, 180), 18)))
            .astype(int)
            .stack('ticker')
            .swaplevel())


def alpha062(o, h, l, vwap, adv20):
    return (rank(ts_corr(vwap, ts_sum(adv20, 22), 9))
            .lt(rank(
        rank(o).mul(2))
                .lt(rank(h.add(l).div(2))
                    .add(rank(h))))
            .mul(-1)
            .stack('ticker')
            .swaplevel())


def alpha064(o, h, l, v, vwap):
    w = 0.178404
    return (rank(ts_corr(ts_sum(o.mul(w).add(l.mul(1 - w)), 12),
                         ts_sum(ts_mean(v, 120), 12), 16))
            .lt(rank(ts_delta(h.add(l).div(2).mul(w)
                              .add(vwap.mul(1 - w)), 3)))
            .mul(-1)
            .stack('ticker')
            .swaplevel())


def alpha065(o, v, vwap):
    w = 0.00817205
    return (rank(ts_corr(o.mul(w).add(vwap.mul(1 - w)),
                         ts_mean(ts_mean(v, 60), 9), 6))
            .lt(rank(o.sub(ts_min(o, 13))))
            .mul(-1)
            .stack('ticker')
            .swaplevel())


def alpha066(o, l, h, vwap):
    w = 0.96633
    return (rank(ts_weighted_mean(ts_delta(vwap, 4), 7))
            .add(ts_rank(ts_weighted_mean(l.mul(w).add(l.mul(1 - w))
                                          .sub(vwap)
                                          .div(o.sub(h.add(l).div(2)).add(1e-3)), 11), 7))
            .mul(-1)
            .stack('ticker')
            .swaplevel())


def alpha068(h, l, c, v):
    w = 0.518371
    return (ts_rank(ts_corr(rank(h), rank(ts_mean(v, 15)), 9), 14)
            .lt(rank(ts_delta(c.mul(w).add(l.mul(1 - w)), 1)))
            .mul(-1)
            .stack('ticker')
            .swaplevel())


def alpha071(o, l, c, v, vwap):
    s1 = (ts_rank(ts_weighted_mean(ts_corr(ts_rank(c, 3),
                                           ts_rank(ts_mean(v, 180), 12), 18), 4), 16))
    s2 = (ts_rank(ts_weighted_mean(rank(l.add(o).
                                        sub(vwap.mul(2)))
                                   .pow(2), 16), 4))
    return (s1.where(s1 > s2, s2)
            .stack('ticker')
            .swaplevel())


def alpha072(h, l, v, vwap):
    return (rank(ts_weighted_mean(ts_corr(h.add(l).div(2), ts_mean(v, 40), 9), 10))
            .div(rank(ts_weighted_mean(ts_corr(ts_rank(vwap, 3), ts_rank(v, 18), 6), 2)))
            .stack('ticker')
            .swaplevel())


def alpha073(o, l, vwap):
    w = 0.147155
    s1 = rank(ts_weighted_mean(ts_delta(vwap, 5), 3))
    s2 = (ts_rank(ts_weighted_mean(ts_delta(o.mul(w).add(l.mul(1 - w)), 2)
                                   .div(o.mul(w).add(l.mul(1 - w)).mul(-1)), 3), 16))

    return (s1.where(s1 > s2, s2)
            .mul(-1)
            .stack('ticker')
            .swaplevel())


def alpha074(h, c, v, vwap):
    w = 0.0261661
    return (rank(ts_corr(c, ts_mean(ts_mean(v, 30), 37), 15))
            .lt(rank(ts_corr(rank(h.mul(w).add(vwap.mul(1 - w))), rank(v), 11)))
            .mul(-1)
            .stack('ticker')
            .swaplevel())


def alpha075(l, v, vwap):
    return (rank(ts_corr(vwap, v, 4))
            .lt(rank(ts_corr(rank(l), rank(ts_mean(v, 50)), 12)))
            .astype(int)
            .stack('ticker')
            .swaplevel())


def alpha077(l, h, v, vwap):
    s1 = rank(ts_weighted_mean(h.add(l).div(2).sub(vwap), 20))
    s2 = rank(ts_weighted_mean(ts_corr(h.add(l).div(2), ts_mean(v, 40), 3), 5))
    return (s1.where(s1 < s2, s2)
            .stack('ticker')
            .swaplevel())


def alpha078(l, v, vwap):
    w = 0.352233
    return (rank(ts_corr(ts_sum((l.mul(w).add(vwap.mul(1 - w))), 19),
                         ts_sum(ts_mean(v, 40), 19), 6))
            .pow(rank(ts_corr(rank(vwap), rank(v), 5)))
            .stack('ticker')
            .swaplevel())


def alpha081(v, vwap):
    return (rank(log(ts_product(rank(rank(ts_corr(vwap,
                                                  ts_sum(ts_mean(v, 10), 50), 8))
                                     .pow(4)), 15)))
            .lt(rank(ts_corr(rank(vwap), rank(v), 5)))
            .mul(-1)
            .stack('ticker')
            .swaplevel())


def alpha083(h, l, c, v, vwap):
    s = h.sub(l).div(ts_mean(c, 5))

    return (rank(rank(ts_lag(s, 2))
                 .mul(rank(rank(v)))
                 .div(s).div(vwap.sub(c).add(1e-3)))
            .stack('ticker')
            .swaplevel()
            .replace((np.inf, -np.inf), np.nan))


def alpha084(c, vwap):
    return (rank(power(ts_rank(vwap.sub(ts_max(vwap, 15)), 20),
                       ts_delta(c, 6)))
            .stack('ticker')
            .swaplevel())


def alpha085(h, l, c, v):
    w = 0.876703
    return (rank(ts_corr(h.mul(w).add(c.mul(1 - w)), ts_mean(v, 30), 10))
            .pow(rank(ts_corr(ts_rank(h.add(l).div(2), 4),
                              ts_rank(v, 10), 7)))
            .stack('ticker')
            .swaplevel())


def alpha086(c, v, vwap):
    return (ts_rank(ts_corr(c, ts_mean(ts_mean(v, 20), 15), 6), 20)
            .lt(rank(c.sub(vwap)))
            .mul(-1)
            .stack('ticker')
            .swaplevel())


def alpha088(o, h, l, c, v):
    s1 = (rank(ts_weighted_mean(rank(o)
                                .add(rank(l))
                                .sub(rank(h))
                                .add(rank(c)), 8)))
    s2 = ts_rank(ts_weighted_mean(ts_corr(ts_rank(c, 8),
                                          ts_rank(ts_mean(v, 60), 20), 8), 6), 2)

    return (s1.where(s1 < s2, s2)
            .stack('ticker')
            .swaplevel())


def alpha092(o, h, l, c, v):
    p1 = ts_rank(ts_weighted_mean(h.add(l).div(2).add(c).lt(l.add(o)), 15), 18)
    p2 = ts_rank(ts_weighted_mean(ts_corr(rank(l), rank(ts_mean(v, 30)), 7), 6), 6)

    return (p1.where(p1 < p2, p2)
            .stack('ticker')
            .swaplevel())


def alpha094(v, vwap):
    return (rank(vwap.sub(ts_min(vwap, 11)))
            .pow(ts_rank(ts_corr(ts_rank(vwap, 20),
                                 ts_rank(ts_mean(v, 60), 4), 18), 2))
            .mul(-1)
            .stack('ticker')
            .swaplevel())


def alpha095(o, h, l, v):
    return (rank(o.sub(ts_min(o, 12)))
            .lt(ts_rank(rank(ts_corr(ts_mean(h.add(l).div(2), 19),
                                     ts_sum(ts_mean(v, 40), 19), 13).pow(5)), 12))
            .astype(int)
            .stack('ticker')
            .swaplevel())


def alpha096(c, v, vwap):
    s1 = ts_rank(ts_weighted_mean(ts_corr(rank(vwap), rank(v), 10), 4), 8)
    s2 = ts_rank(ts_weighted_mean(ts_argmax(ts_corr(ts_rank(c, 7),
                                                    ts_rank(ts_mean(v, 60), 10), 10), 12), 14), 13)
    return (s1.where(s1 > s2, s2)
            .mul(-1)
            .stack('ticker')
            .swaplevel())


def alpha098(o, v, vwap):
    adv5 = ts_mean(v, 5)
    adv15 = ts_mean(v, 15)
    return (rank(ts_weighted_mean(ts_corr(vwap, ts_mean(adv5, 26), 4), 7))
            .sub(rank(ts_weighted_mean(ts_rank(ts_argmin(ts_corr(rank(o),
                                                                 rank(adv15), 20), 8), 6))))
            .stack('ticker')
            .swaplevel())


def alpha099(h, l, v):
    return ((rank(ts_corr(ts_sum((h.add(l).div(2)), 19),
                          ts_sum(ts_mean(v, 60), 19), 8))
             .lt(rank(ts_corr(l, v, 6)))
             .mul(-1))
            .stack('ticker')
            .swaplevel())


def alpha101(o, h, l, c):
    return (c.sub(o).div(h.sub(l).add(1e-3))
            .stack('ticker')
            .swaplevel())

class Alphas101:

    def __init__(self):
        memory = Memory(f"store/{get_name()}", verbose=0)
        self.fit = memory.cache(self.fit)
        self.transform = memory.cache(self.transform)

    def fit(self, data):
        return self

    def transform(self, data):

        ohlcv = ['open', 'high', 'low', 'close', 'volume']
        adv20 = data.groupby('ticker').rolling(20).volume.mean().reset_index(0, drop=True)
        data = data.assign(adv20=adv20)
        data = data.join(data.groupby('date')[ohlcv].rank(axis=1, pct=True), rsuffix='_rank')
        data.dropna(inplace=True)

        o = data.open.unstack('ticker')
        h = data.high.unstack('ticker')
        l = data.low.unstack('ticker')
        c = data.close.unstack('ticker')
        v = data.volume.unstack('ticker')
        vwap = o.add(h).add(l).add(c).div(4)
        adv20 = v.rolling(20).mean()
        r = data.close.unstack('ticker').pct_change()

        alphas = []
        alphas.append(alpha001(c, r).rename('alpha001'))
        alphas.append(alpha002(o, c, v).rename('alpha002'))
        alphas.append(alpha003(o, v).rename('alpha003'))
        alphas.append(alpha004(l).rename('alpha004'))
        alphas.append(alpha005(o, vwap, c).rename('alpha005'))
        alphas.append(alpha006(o, v).rename('alpha006'))
        alphas.append(alpha007(c, v, adv20).rename('alpha007'))
        alphas.append(alpha008(o, r).rename('alpha008'))
        alphas.append(alpha009(c).rename('alpha009'))
        alphas.append(alpha010(c).rename('alpha010'))
        alphas.append(alpha011(c, vwap, v).rename('alpha011'))
        alphas.append(alpha012(v, c).rename('alpha012'))
        alphas.append(alpha013(c, v).rename('alpha013'))
        alphas.append(alpha014(o, v, r).rename('alpha014'))
        alphas.append(alpha015(h, v).rename('alpha015'))
        alphas.append(alpha016(h, v).rename('alpha016'))
        alphas.append(alpha017(c, v).rename('alpha017'))
        alphas.append(alpha018(o, c).rename('alpha018'))
        alphas.append(alpha019(c, r).rename('alpha019'))
        alphas.append(alpha020(o, h, l, c).rename('alpha020'))
        alphas.append(alpha021(c, v).rename('alpha021'))
        alphas.append(alpha022(h, c, v).rename('alpha022'))
        alphas.append(alpha023(h, c).rename('alpha023'))
        alphas.append(alpha024(c).rename('alpha024'))
        alphas.append(alpha025(h, c, r, vwap, adv20).rename('alpha025'))
        alphas.append(alpha026(h, v).rename('alpha026'))
        alphas.append(alpha027(v, vwap).rename('alpha027'))
        alphas.append(alpha028(h, l, c, v, adv20).rename('alpha028'))
        alphas.append(alpha029(c, r).rename('alpha029'))
        alphas.append(alpha030(c, v).rename('alpha030'))
        # alphas.append(alpha031(l, c, adv20).rename('alpha031')) # Produces all nans
        alphas.append(alpha032(c, vwap).rename('alpha032'))
        alphas.append(alpha033(o, c).rename('alpha033'))
        alphas.append(alpha034(c, r).rename('alpha034'))
        alphas.append(alpha035(h, l, c, v, r).rename('alpha035'))
        alphas.append(alpha036(o, c, v, r, adv20, vwap).rename('alpha036'))
        alphas.append(alpha037(o, c).rename('alpha037'))
        alphas.append(alpha038(o, c).rename('alpha038'))
        alphas.append(alpha039(c, v, r, adv20).rename('alpha039'))
        alphas.append(alpha040(h, v).rename('alpha040'))
        alphas.append(alpha041(h, l, vwap).rename('alpha041'))
        alphas.append(alpha042(c, vwap).rename('alpha042'))
        alphas.append(alpha043(c, v, adv20).rename('alpha043'))
        alphas.append(alpha044(h, v).rename('alpha044'))
        alphas.append(alpha045(c, v).rename('alpha045'))
        alphas.append(alpha046(c).rename('alpha046'))
        alphas.append(alpha047(h, c, v, vwap, adv20).rename('alpha047'))
        # alphas.append(alpha048(h, c, vwap, adv20).rename('alpha048'))  # No implementation
        alphas.append(alpha049(c).rename('alpha049'))
        alphas.append(alpha050(v, vwap).rename('alpha050'))
        alphas.append(alpha051(c).rename('alpha051'))
        alphas.append(alpha052(l, v, r).rename('alpha052'))
        alphas.append(alpha053(h, l, c).rename('alpha053'))
        alphas.append(alpha054(o, h, l, c).rename('alpha054'))
        alphas.append(alpha055(h, l, c, v).rename('alpha055'))
        # alphas.append(alpha056(h, l, c).rename('alpha056'))  # No implementation
        # alphas.append(alpha057(h, l, c).rename('alpha057'))  # No implementation
        # alphas.append(alpha058(h, l, c).rename('alpha057'))  # No implementation
        # alphas.append(alpha059(h, l, c).rename('alpha059'))  # No implementation
        alphas.append(alpha060(l, h, c, v).rename('alpha060'))
        alphas.append(alpha061(v, vwap).rename('alpha061'))
        alphas.append(alpha062(o, h, l, vwap, adv20).rename('alpha062'))
        # alphas.append(alpha063(o, h, l, vwap, adv20).rename('alpha063'))  # No implementation
        alphas.append(alpha064(o, h, l, v, vwap).rename('alpha064'))
        alphas.append(alpha065(o, v, vwap).rename('alpha065'))
        alphas.append(alpha066(o, l, h, vwap).rename('alpha066'))
        # alphas.append(alpha067(l, h, vwap).rename('alpha067'))
        alphas.append(alpha068(h, l, c, v).rename('alpha068'))
        # alphas.append(alpha069(h, c, v).rename('alpha069'))
        # alphas.append(alpha070(h, c, v).rename('alpha070'))
        alphas.append(alpha071(o, l, c, v, vwap).rename('alpha071'))
        alphas.append(alpha072(h, l, v, vwap).rename('alpha072'))
        alphas.append(alpha073(o, l, vwap).rename('alpha073'))
        alphas.append(alpha074(h, c, v, vwap).rename('alpha074'))
        alphas.append(alpha075(l, v, vwap).rename('alpha075'))
        # alphas.append(alpha076(l, v, vwap).rename('alpha076'))
        alphas.append(alpha077(l, h, v, vwap).rename('alpha077'))
        alphas.append(alpha078(l, v, vwap).rename('alpha078'))
        # alphas.append(alpha079(l, v, vwap).rename('alpha079'))
        # alphas.append(alpha080(l, v, vwap).rename('alpha080'))
        alphas.append(alpha081(v, vwap).rename('alpha081'))
        # alphas.append(alpha082(v, vwap).rename('alpha082'))
        alphas.append(alpha083(h, l, c, v, vwap).rename('alpha083'))
        alphas.append(alpha084(c, vwap).rename('alpha084'))
        alphas.append(alpha085(h, l, c, v).rename('alpha085'))
        alphas.append(alpha086(c, v, vwap).rename('alpha086'))
        # alphas.append(alpha087(c, v, vwap).rename('alpha087'))
        alphas.append(alpha088(o, h, l, c, v).rename('alpha088'))
        # alphas.append(alpha089(o, h, l, c, v).rename('alpha089'))
        # alphas.append(alpha090(o, h, l, c, v).rename('alpha090'))
        # alphas.append(alpha091(o, h, l, c, v).rename('alpha091'))
        alphas.append(alpha092(o, h, l, c, v).rename('alpha092'))
        # alphas.append(alpha093(o, l, c, v).rename('alpha093'))
        alphas.append(alpha094(v, vwap).rename('alpha094'))
        alphas.append(alpha095(o, h, l, v).rename('alpha095'))
        alphas.append(alpha096(c, v, vwap).rename('alpha096'))
        # alphas.append(alpha097(c, v, vwap).rename('alpha097'))
        alphas.append(alpha098(o, v, vwap).rename('alpha098'))
        alphas.append(alpha099(h, l, v).rename('alpha099'))
        # alphas.append(alpha100(l, v).rename('alpha100'))
        alphas.append(alpha101(o, h, l, c).rename('alpha101'))

        features = pd.concat(alphas, axis=1)
        features = features.reorder_levels(order=[1, 0])
        features = features.sort_index()
        return features

