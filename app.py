"""
Aplicación de Optimización Numérica
=====================================
Métodos: Descenso de Gradiente, Gradiente Conjugado (Polak-Ribière), Newton
Búsqueda de línea: Condiciones de Wolfe (primera y segunda)
"""

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import numpy as np
import sympy as sp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Optimización Numérica | Wolfe",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* ── global ── */
    [data-testid="stAppViewContainer"] { background: #0d1117; }
    [data-testid="stSidebar"] { background: #0a0f1a; border-right: 1px solid #1e2d40; }
    [data-testid="stSidebar"] * { color: #c9d1d9 !important; }

    /* ── header ── */
    .app-header {
        background: linear-gradient(135deg, #1a1f35 0%, #0d1117 100%);
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 28px;
        text-align: center;
    }
    .app-title {
        font-size: 2.4em;
        font-weight: 800;
        background: linear-gradient(135deg, #7c6af5, #4fc3f7, #81ecec);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0; line-height: 1.2;
    }
    .app-sub {
        color: #8b949e;
        font-size: 1em;
        margin-top: 8px;
    }

    /* ── cards ── */
    .card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px 22px;
        margin-bottom: 14px;
        transition: border-color 0.2s;
    }
    .card:hover { border-color: #7c6af5; }

    .card-title {
        font-size: 0.75em;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #7c6af5;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .card-value {
        font-size: 1.5em;
        font-weight: 700;
        color: #e6edf3;
        font-family: 'Courier New', monospace;
    }
    .card-sub {
        font-size: 0.82em;
        color: #8b949e;
        margin-top: 4px;
    }

    /* ── badges ── */
    .badge-ok {
        display: inline-block;
        background: #0d2818;
        border: 1px solid #238636;
        color: #3fb950;
        border-radius: 20px;
        padding: 3px 14px;
        font-size: 0.8em;
        font-weight: 600;
    }
    .badge-fail {
        display: inline-block;
        background: #2d1215;
        border: 1px solid #da3633;
        color: #f85149;
        border-radius: 20px;
        padding: 3px 14px;
        font-size: 0.8em;
        font-weight: 600;
    }

    /* ── method header ── */
    .method-header {
        display: flex; align-items: center; gap: 10px;
        margin-bottom: 16px;
    }
    .method-icon {
        font-size: 1.8em;
    }
    .method-name {
        font-size: 1.2em;
        font-weight: 700;
        color: #e6edf3;
    }

    /* ── section labels ── */
    .section-label {
        font-size: 1.05em;
        font-weight: 700;
        color: #7c6af5;
        border-bottom: 1px solid #21262d;
        padding-bottom: 6px;
        margin: 24px 0 14px 0;
    }

    /* ── info boxes ── */
    .info-box {
        background: #0d2818;
        border: 1px solid #238636;
        border-radius: 8px;
        padding: 12px 16px;
        color: #3fb950;
        font-size: 0.9em;
        margin: 12px 0;
    }

    /* ── run button ── */
    div.stButton > button {
        background: linear-gradient(135deg, #7c6af5, #4fc3f7) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 1.05em !important;
        padding: 12px 0 !important;
        width: 100% !important;
        letter-spacing: 0.5px;
        cursor: pointer;
    }
    div.stButton > button:hover { opacity: 0.88 !important; }

    /* ── code ── */
    code {
        background: #21262d;
        padding: 2px 6px;
        border-radius: 4px;
        font-family: 'Courier New', monospace;
        font-size: 0.88em;
        color: #79c0ff;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCTION PARSING (SymPy)
# ═══════════════════════════════════════════════════════════════════════════════

def parse_and_build(expr_str: str, n_vars: int):
    """
    Parsea la función ingresada como string y retorna callables numpy
    para f, grad_f y hess_f. Las variables deben ser x1, x2, ..., xn.
    """
    var_names = [f'x{i+1}' for i in range(n_vars)]
    sym_vars  = sp.symbols(' '.join(var_names))
    if n_vars == 1:
        sym_vars = (sym_vars,)

    local_dict = {name: var for name, var in zip(var_names, sym_vars)}

    try:
        expr = sp.sympify(expr_str, locals=local_dict)
    except Exception as e:
        raise ValueError(f"No se pudo interpretar la función: {e}")

    grad_exprs = [sp.diff(expr, v) for v in sym_vars]
    hess_exprs = sp.hessian(expr, sym_vars)

    _f  = sp.lambdify(sym_vars, expr,       modules='numpy')
    _gf = sp.lambdify(sym_vars, grad_exprs, modules='numpy')
    _hf = sp.lambdify(sym_vars, hess_exprs, modules='numpy')

    def f(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        val = _f(*x)
        return float(np.real(np.atleast_1d(val)[0] if hasattr(val, '__len__') else val))

    def grad_f(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        g = _gf(*x)
        g = [float(np.real(np.atleast_1d(gi)[0])) if hasattr(gi, '__len__') else float(np.real(gi)) for gi in g]
        return np.array(g, dtype=float)

    def hess_f(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        H = _hf(*x)
        H = np.array(H, dtype=complex)
        return np.real(H).astype(float)

    return f, grad_f, hess_f, sym_vars, expr


# ═══════════════════════════════════════════════════════════════════════════════
# BÚSQUEDA DE LÍNEA — CONDICIONES DE WOLFE
# ═══════════════════════════════════════════════════════════════════════════════

def _zoom(f, grad_f, x, d, alpha_lo, alpha_hi, f0, g0, c1, c2, strong):
    """Fase zoom del algoritmo de Wolfe (Nocedal & Wright, Alg. 3.6)."""
    for _ in range(60):
        alpha = (alpha_lo + alpha_hi) / 2.0
        x_new = x + alpha * d
        f_new = f(x_new)

        if f_new > f0 + c1 * alpha * g0 or f_new >= f(x + alpha_lo * d):
            alpha_hi = alpha
        else:
            g_new = np.dot(grad_f(x_new), d)
            cond2 = abs(g_new) <= c2 * abs(g0) if strong else g_new >= c2 * g0
            if cond2:
                return alpha
            if g_new * (alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo
            alpha_lo = alpha

    return (alpha_lo + alpha_hi) / 2.0


def wolfe_line_search(f, grad_f, x, d, c1=1e-4, c2=0.9,
                      strong=True, alpha_max=1.0):
    """
    Búsqueda de línea con condiciones de Wolfe.
    - Primera condición  (Armijo):   f(x+αd) ≤ f(x) + c₁·α·∇f(x)ᵀd
    - Segunda condición (curvatura): |∇f(x+αd)ᵀd| ≤ c₂·|∇f(x)ᵀd|  (strong)
                                  o  ∇f(x+αd)ᵀd ≥ c₂·∇f(x)ᵀd      (weak)
    """
    alpha_prev = 0.0
    alpha = min(1.0, alpha_max)
    f0 = f(x)
    g0 = np.dot(grad_f(x), d)

    if g0 >= 0:
        return 1e-10  # dirección no es de descenso

    for i in range(100):
        x_new  = x + alpha * d
        f_new  = f(x_new)

        if f_new > f0 + c1 * alpha * g0 or (i > 0 and f_new >= f(x + alpha_prev * d)):
            return _zoom(f, grad_f, x, d, alpha_prev, alpha, f0, g0, c1, c2, strong)

        g_new = np.dot(grad_f(x_new), d)

        cond2 = abs(g_new) <= c2 * abs(g0) if strong else g_new >= c2 * g0
        if cond2:
            return alpha

        if g_new >= 0:
            return _zoom(f, grad_f, x, d, alpha, alpha_prev, f0, g0, c1, c2, strong)

        alpha_prev = alpha
        alpha = min(2.0 * alpha, alpha_max)

    return alpha


# ═══════════════════════════════════════════════════════════════════════════════
# MÉTODOS DE OPTIMIZACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def gradient_descent(f, grad_f, x0, max_iter, tol, c1, c2, strong):
    """Descenso de Gradiente con búsqueda de línea de Wolfe."""
    x    = x0.copy()
    hist = []

    for i in range(max_iter + 1):
        g      = grad_f(x)
        norm_g = np.linalg.norm(g)
        hist.append({'iter': i, 'x': x.copy(), 'f': f(x), 'grad_norm': norm_g})

        if norm_g <= tol:
            break

        d     = -g
        alpha = wolfe_line_search(f, grad_f, x, d, c1, c2, strong)
        x     = x + alpha * d

    

    return x, hist, hist[-1]['grad_norm'] <= tol


def conjugate_gradient(f, grad_f, x0, max_iter, tol, c1, c2, strong):
    # CG necesita c2 < 0.5 en funciones no cuadráticas
    # con c2=0.9 el line search acepta pasos de mala calidad
    c2 = min(c2, 0.4)

    x    = x0.copy()
    g    = grad_f(x)
    d    = -g.copy()
    hist = []

    for i in range(max_iter):
        norm_g = np.linalg.norm(g)
        hist.append({'iter': i, 'x': x.copy(), 'f': f(x), 'grad_norm': norm_g})

        if norm_g <= tol:
            break

        # Si d dejó de ser dirección de descenso, reiniciar
        if np.dot(g, d) >= 0:
            d = -g.copy()

        alpha = wolfe_line_search(f, grad_f, x, d, c1, c2, strong)

        # Si el paso es insignificante, forzar reinicio
        if alpha < 1e-14:
            d     = -g.copy()
            alpha = wolfe_line_search(f, grad_f, x, d, c1, c2, strong)

        x_new = x + alpha * d
        g_new = grad_f(x_new)

        gg     = np.dot(g, g)
        gg_new = np.dot(g_new, g_new)

        if gg < 1e-30 or gg_new < 1e-30:
            beta = 0.0
        else:
            # Polak-Ribière+
            beta = max(0.0, np.dot(g_new, g_new - g) / gg)

            # Criterio de reinicio de Powell:
            # si los gradientes no son suficientemente ortogonales → reiniciar
            if abs(np.dot(g_new, g)) / gg_new >= 0.1:
                beta = 0.0

        d = -g_new + beta * d
        x = x_new
        g = g_new

    return x, hist, hist[-1]['grad_norm'] <= tol

def newton_method(f, grad_f, hess_f, x0, max_iter, tol, c1, c2, strong):
    """
    Método de Newton con búsqueda de línea de Wolfe.
    Usa modificación de la Hessiana (Cholesky con perturbación)
    para garantizar dirección de descenso.
    """
    x    = x0.copy()
    hist = []

    for i in range(max_iter + 1):
        g      = grad_f(x)
        norm_g = np.linalg.norm(g)
        hist.append({'iter': i, 'x': x.copy(), 'f': f(x), 'grad_norm': norm_g})

        if norm_g <= tol:
            break

        H = hess_f(x)

        # Modificación de Hessiana si no es definida positiva
        beta_reg = 1e-6
        for _ in range(50):
            try:
                eigs = np.linalg.eigvalsh(H)
                if np.all(eigs > 1e-12):
                    d = np.linalg.solve(H, -g)
                    break
                H = H + beta_reg * np.eye(len(x))
                beta_reg *= 10
            except np.linalg.LinAlgError:
                H = H + beta_reg * np.eye(len(x))
                beta_reg *= 10
        else:
            d = -g  # fallback

        # Verificar que es dirección de descenso
        if np.dot(g, d) >= 0:
            d = -g

        alpha = wolfe_line_search(f, grad_f, x, d, c1, c2, strong)
        x = x + alpha * d

    

    return x, hist, hist[-1]['grad_norm'] <= tol


# ═══════════════════════════════════════════════════════════════════════════════
# GRÁFICOS
# ═══════════════════════════════════════════════════════════════════════════════

COLORS = {
    'Gradiente':           '#7c6af5',
    'Gradiente Conjugado': '#4fc3f7',
    'Newton':              '#f7b731',
}

DARK = dict(
    paper_bgcolor='#0d1117',
    plot_bgcolor='#161b22',
    font=dict(color='#8b949e', size=13),
    xaxis=dict(gridcolor='#21262d', linecolor='#30363d', title_font=dict(color='#c9d1d9')),
    yaxis=dict(gridcolor='#21262d', linecolor='#30363d', title_font=dict(color='#c9d1d9')),
    legend=dict(bgcolor='#1c2128', bordercolor='#30363d', borderwidth=1,
                font=dict(color='#e6edf3')),
)


def plot_convergence(histories: dict) -> go.Figure:
    """Curvas de convergencia: norma del gradiente y f(x) vs iteraciones."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "‖∇f(x)‖ vs Iteraciones  (escala log)",
            "f(x) vs Iteraciones"
        ],
        horizontal_spacing=0.10
    )

    for method, history in histories.items():
        c    = COLORS.get(method, '#ffffff')
        its  = [h['iter']     for h in history]
        gnms = [h['grad_norm'] for h in history]
        fvls = [h['f']         for h in history]

        kw = dict(line=dict(color=c, width=2.5), mode='lines+markers',
                  marker=dict(size=4, color=c))

        fig.add_trace(go.Scatter(x=its, y=gnms, name=method, **kw), row=1, col=1)
        fig.add_trace(go.Scatter(x=its, y=fvls, name=method,
                                  showlegend=False,
                                  line=dict(color=c, width=2.5, dash='dot'),
                                  mode='lines+markers', marker=dict(size=4, color=c)),
                      row=1, col=2)

    fig.update_yaxes(type='log', row=1, col=1)
    fig.update_layout(height=400, **DARK,
                      margin=dict(t=50, b=30, l=40, r=20))
    fig.update_annotations(font_color='#c9d1d9')
    fig.update_xaxes(title_text="Iteración", row=1, col=1)
    fig.update_xaxes(title_text="Iteración", row=1, col=2)
    fig.update_yaxes(title_text="‖∇f(x)‖", row=1, col=1)
    fig.update_yaxes(title_text="f(x)", row=1, col=2)
    return fig


def plot_contour(f, histories: dict, x_star: np.ndarray) -> go.Figure:
    """Trayectorias sobre el mapa de contorno (solo para n=2)."""
    all_x = np.vstack([h['x'] for hist in histories.values() for h in hist])
    cx, cy = np.mean(all_x, axis=0)
    pad    = max(2.5, np.ptp(all_x, axis=0).max() * 0.6)

    x1r = np.linspace(cx - pad, cx + pad, 100)
    x2r = np.linspace(cy - pad, cy + pad, 100)
    X1, X2 = np.meshgrid(x1r, x2r)
    Z = np.vectorize(lambda a, b: f(np.array([a, b])))(X1, X2)

    fig = go.Figure()

    fig.add_trace(go.Contour(
        x=x1r, y=x2r, z=Z,
        colorscale='Plasma',
        showscale=True,
        contours=dict(coloring='heatmap', showlabels=True,
                      labelfont=dict(size=10, color='white')),
        opacity=0.55,
        colorbar=dict(tickfont=dict(color='#8b949e'))
    ))

    for method, history in histories.items():
        c  = COLORS.get(method, '#fff')
        xs = [h['x'][0] for h in history]
        ys = [h['x'][1] for h in history]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, name=method, mode='lines+markers',
            line=dict(color=c, width=2.2),
            marker=dict(size=5, color=c, line=dict(color='#0d1117', width=0.5))
        ))

    fig.add_trace(go.Scatter(
        x=[x_star[0]], y=[x_star[1]], name='x* (mínimo)',
        mode='markers',
        marker=dict(symbol='star', size=18, color='#f85149',
                    line=dict(color='white', width=1))
    ))

    fig.update_layout(
        title=dict(text='Trayectoria en el Espacio de Búsqueda',
                   font=dict(color='#c9d1d9')),
        xaxis_title='x₁', yaxis_title='x₂',
        height=440, **DARK,
        margin=dict(t=50, b=30, l=40, r=120),
    )
    fig.update_layout(
        legend=dict(
            bgcolor='#1c2128',
            bordercolor='#30363d',
            borderwidth=1,
            font=dict(color='#e6edf3'),
            orientation='h',
            yanchor='bottom',
            y=-0.2,
            xanchor='center',
            x=0.5
        )
    )
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# VALOR AGREGADO — OPTIMIZACIÓN RESTRINGIDA + KKT
# ═══════════════════════════════════════════════════════════════════════════════

from scipy.optimize import minimize as scipy_minimize

def build_2d_fns(expr_str):
    """Parsea función de 2 variables usando x, y."""
    xs, ys = sp.symbols('x y')
    local  = {'x': xs, 'y': ys}
    expr   = sp.sympify(expr_str, locals=local)
    _f  = sp.lambdify((xs, ys), expr,                   modules='numpy')
    _gx = sp.lambdify((xs, ys), sp.diff(expr, xs),      modules='numpy')
    _gy = sp.lambdify((xs, ys), sp.diff(expr, ys),      modules='numpy')
    def fn(pt):
        return float(np.real(_f(pt[0], pt[1])))
    def gn(pt):
        return np.array([float(np.real(_gx(pt[0], pt[1]))),
                         float(np.real(_gy(pt[0], pt[1])))], dtype=float)
    return fn, gn, expr


def solve_kkt(f_str, eq_strs, ineq_strs, x0):
    """
    Resuelve optimización restringida en 2D y calcula condiciones KKT.
    Restricciones de igualdad:    g(x,y) = 0
    Restricciones de desigualdad: h(x,y) ≤ 0
    """
    f_fn, f_gn, f_expr = build_2d_fns(f_str)

    eq_data   = [dict(zip(('fn','gn','expr'), build_2d_fns(s)))
                 for s in eq_strs if s.strip()]
    ineq_data = [dict(zip(('fn','gn','expr'), build_2d_fns(s)))
                 for s in ineq_strs if s.strip()]

    # scipy 'ineq' requiere fun(x) >= 0, entonces negamos h(x) ≤ 0
    constraints = (
        [{'type': 'eq',   'fun': d['fn'], 'jac': d['gn']} for d in eq_data] +
        [{'type': 'ineq', 'fun': lambda pt, d=d: -d['fn'](pt),
                          'jac': lambda pt, d=d: -d['gn'](pt)} for d in ineq_data]
    )

    res = scipy_minimize(f_fn, x0, method='SLSQP', jac=f_gn,
                         constraints=constraints, tol=1e-10,
                         options={'maxiter': 2000, 'ftol': 1e-10})
    x_star = res.x

    # Multiplicadores: resuelve sistema KKT de estacionariedad
    # ∇f(x*) + Σ λᵢ∇gᵢ(x*) + Σ μⱼ∇hⱼ(x*) = 0  →  A·m = -∇f
    all_grads = [d['gn'](x_star) for d in eq_data + ineq_data]
    mults = []
    if all_grads:
        A = np.column_stack(all_grads)
        m, _, _, _ = np.linalg.lstsq(A, -f_gn(x_star), rcond=None)
        mults = m.tolist()

    EPS = 1e-4
    n_eq = len(eq_data)

    # ── Verificación de condiciones KKT ──────────────────────────────────────
    # 1. Estacionariedad
    residual = f_gn(x_star).copy()
    for i, d in enumerate(eq_data + ineq_data):
        if i < len(mults):
            residual += mults[i] * d['gn'](x_star)
    stat_norm = np.linalg.norm(residual)

    # 2. Factibilidad primal — igualdad
    eq_vals  = [d['fn'](x_star)  for d in eq_data]

    # 3. Factibilidad primal — desigualdad
    ineq_vals = [d['fn'](x_star) for d in ineq_data]

    # 4. Factibilidad dual (μⱼ ≥ 0)
    mu_vals = [mults[n_eq + i] for i in range(len(ineq_data))] if mults else []

    # 5. Holgura complementaria (μⱼ · hⱼ(x*) = 0)
    comp_vals = [abs(mu_vals[i] * ineq_vals[i]) for i in range(len(ineq_data))]

    return {
        'x_star':    x_star,
        'f_star':    f_fn(x_star),
        'success':   res.success,
        'message':   res.message,
        'f_fn':      f_fn,
        'f_gn':      f_gn,
        'f_expr':    f_expr,
        'eq_data':   eq_data,
        'ineq_data': ineq_data,
        'mults':     mults,
        'n_eq':      n_eq,
        'kkt': {
            'stationarity':     (stat_norm      < EPS,  stat_norm),
            'eq_feasibility':   [(abs(v)         < EPS,  v)  for v in eq_vals],
            'ineq_feasibility': [(v              <= EPS, v)  for v in ineq_vals],
            'dual_feasibility': [(v              >= -EPS,v)  for v in mu_vals],
            'comp_slackness':   [(v              < EPS,  v)  for v in comp_vals],
        }
    }


def plot_kkt_2d(data):
    """Contorno de f + curvas de restricción + vectores gradiente en x*."""
    x_star, f_fn = data['x_star'], data['f_fn']
    pad = max(2.5, np.abs(x_star).max() * 1.5 + 1.5)
    xr  = np.linspace(x_star[0]-pad, x_star[0]+pad, 130)
    yr  = np.linspace(x_star[1]-pad, x_star[1]+pad, 130)
    X, Y = np.meshgrid(xr, yr)
    Z = np.vectorize(lambda a, b: f_fn([a, b]))(X, Y)

    fig = go.Figure()

    # Contorno de f
    fig.add_trace(go.Contour(
        x=xr, y=yr, z=Z, colorscale='Blues', opacity=0.50,
        showscale=True,
        contours=dict(coloring='heatmap', showlabels=True,
                      labelfont=dict(size=9, color='white')),
        colorbar=dict(x=1.13, thickness=13, tickfont=dict(color='#8b949e'))
    ))

    EQ_COLORS   = ['#f7b731', '#a29bfe', '#fd79a8']
    INEQ_COLORS = ['#00b894', '#e17055']

    # Restricciones de igualdad — curva g(x,y)=0
    for i, d in enumerate(data['eq_data']):
        Gc = np.vectorize(lambda a, b, d=d: d['fn']([a, b]))(X, Y)
        c  = EQ_COLORS[i % len(EQ_COLORS)]
        fig.add_trace(go.Contour(
            x=xr, y=yr, z=Gc, showscale=False,
            contours=dict(start=0, end=0, size=1e-6, coloring='lines',
                          showlabels=True, labelfont=dict(size=10, color=c)),
            line=dict(color=c, width=3),
            name=f'g{i+1}(x,y) = 0'
        ))

    # Restricciones de desigualdad — frontera h(x,y)=0
    for i, d in enumerate(data['ineq_data']):
        Hc = np.vectorize(lambda a, b, d=d: d['fn']([a, b]))(X, Y)
        c  = INEQ_COLORS[i % len(INEQ_COLORS)]
        fig.add_trace(go.Contour(
            x=xr, y=yr, z=Hc, showscale=False,
            contours=dict(start=0, end=0, size=1e-6, coloring='lines'),
            line=dict(color=c, width=2.5, dash='dash'),
            name=f'h{i+1}(x,y) = 0  (frontera)'
        ))

    # Vectores gradiente en x*
    scale = pad * 0.28
    mults, n_eq = data['mults'], data['n_eq']

    def add_arrow(vec, color, label):
        norm = np.linalg.norm(vec)
        if norm < 1e-10: return
        uv = vec / norm * scale
        fig.add_annotation(
            x=x_star[0]+uv[0], y=x_star[1]+uv[1],
            ax=x_star[0], ay=x_star[1],
            xref='x', yref='y', axref='x', ayref='y',
            arrowhead=3, arrowsize=1.5, arrowwidth=2.5,
            arrowcolor=color,
            text=f'<b>{label}</b>',
            font=dict(color=color, size=12)
        )

    add_arrow(data['f_gn'](x_star), '#74b9ff', '∇f')
    for i, d in enumerate(data['eq_data']):
        if i < len(mults):
            add_arrow(mults[i] * d['gn'](x_star),
                      EQ_COLORS[i % len(EQ_COLORS)], f'λ{i+1}∇g{i+1}')
    for i, d in enumerate(data['ineq_data']):
        idx = n_eq + i
        if idx < len(mults):
            add_arrow(mults[idx] * d['gn'](x_star),
                      INEQ_COLORS[i % len(INEQ_COLORS)], f'μ{i+1}∇h{i+1}')

    # Punto óptimo
    fig.add_trace(go.Scatter(
        x=[x_star[0]], y=[x_star[1]], mode='markers',
        name=f'x* = ({x_star[0]:.4f}, {x_star[1]:.4f})',
        marker=dict(symbol='star', size=20, color='#f85149',
                    line=dict(color='white', width=1.5))
    ))

    fig.update_layout(
        title=dict(text='Región factible, restricciones y punto óptimo',
                   font=dict(color='#c9d1d9')),
        xaxis_title='x', yaxis_title='y',
        height=480, **DARK,
        margin=dict(t=50, b=80, l=40, r=130)
    )
    fig.update_layout(legend=dict(
        bgcolor='#1c2128', bordercolor='#30363d', borderwidth=1,
        font=dict(color='#e6edf3'), orientation='h',
        yanchor='bottom', y=-0.28, xanchor='center', x=0.5
    ))
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# INTERFAZ PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

# ── Encabezado ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="app-title">📉 Optimización Numérica</div>
    <div class="app-sub">
        Descenso de Gradiente · Gradiente Conjugado (Polak-Ribière) · Método de Newton<br>
        <span style="color:#7c6af5; font-weight:600;">Búsqueda de línea con condiciones de Wolfe (1ª y 2ª)</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📉 Calculadora Principal", "🔒 Restricciones & KKT"])

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Parámetros de Entrada")
    st.markdown("---")

    n_vars = st.number_input("🔢 Número de variables", min_value=1, max_value=50, value=2, step=1)
    n_vars = int(n_vars)

    st.markdown("**🧮 Métodos a ejecutar**")
    use_gd = st.checkbox("Descenso de Gradiente",   value=True)
    use_cg = st.checkbox("Gradiente Conjugado",      value=True)
    use_nt = st.checkbox("Método de Newton",         value=True)

    st.markdown("---")

    st.markdown("**📐 Función objetivo**")
    default_fn = "(x1 - 2)**2 + (x2 - 3)**2" if n_vars >= 2 else "(x1 - 2)**2"
    func_str = st.text_area(
        f"f(x₁{',...,x'+str(n_vars) if n_vars>1 else ''})",
        value=default_fn, height=80,
        help="Use x1, x2, ..., xn como variables.\nEjemplo: 100*(x2-x1**2)**2 + (1-x1)**2"
    )

    st.markdown("**🎯 Punto de partida x₀**")
    x0_str = st.text_input("Valores separados por coma",
                            value=', '.join(['0.0'] * n_vars))

    st.markdown("---")

    st.markdown("**🔄 Parámetros de iteración**")
    max_iter = st.slider("Máx. iteraciones", 10, 2000, 300, 10)
    tol = st.select_slider(
        "Tolerancia de convergencia",
        options=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
        value=1e-6,
        format_func=lambda x: f"{x:.0e}"
    )

    st.markdown("**🔍 Parámetros de Wolfe**")
    wolfe_type = st.radio(
        "Condición de Wolfe",
        ["Solo 1ª condición (Armijo)", "1ª y 2ª condición (Wolfe fuerte)"],
        index=1,
        help="La 2ª condición (curvatura) garantiza pasos de calidad"
    )
    strong_wolfe = (wolfe_type == "1ª y 2ª condición (Wolfe fuerte)")

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        c1 = st.number_input("c₁ (Armijo)", value=1e-4,
                              min_value=1e-8, max_value=0.4, format="%.1e",
                              help="Suficiente decrecimiento. Típico: 1e-4")
    with col_c2:
        c2 = st.number_input("c₂ (curvatura)", value=0.9,
                              min_value=0.01, max_value=0.9999, format="%.2f",
                              help="c₂ > c₁. Newton: 0.9 | GC: 0.1")

    st.markdown("---")
    run_btn = st.button("🚀 Ejecutar optimización")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CALCULADORA PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    _ok = True

    # ── Página de bienvenida ──────────────────────────────────────────────────
    if not run_btn:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="card">
                <div class="method-header">
                    <span class="method-icon">⬇️</span>
                    <span class="method-name">Descenso de Gradiente</span>
                </div>
                <div class="card-sub">
                    Método de <strong>primer orden</strong>. En cada iteración avanza
                    en la dirección de mayor descenso <em>-∇f(x)</em>.<br><br>
                    Convergencia <strong>lineal</strong> en funciones fuertemente convexas.
                    Simple y robusto, pero puede ser lento cerca del mínimo.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="card">
                <div class="method-header">
                    <span class="method-icon">🔀</span>
                    <span class="method-name">Gradiente Conjugado</span>
                </div>
                <div class="card-sub">
                    Combina el gradiente actual con la dirección anterior mediante
                    el parámetro β <em>(Polak-Ribière)</em>.<br><br>
                    Convergencia <strong>superlineal</strong>. Mucho más eficiente que
                    gradiente puro, especialmente en problemas cuadráticos.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="card">
                <div class="method-header">
                    <span class="method-icon">⚡</span>
                    <span class="method-name">Método de Newton</span>
                </div>
                <div class="card-sub">
                    Usa información de <strong>segundo orden</strong> (Hessiana).
                    La dirección es <em>d = -H⁻¹∇f(x)</em>.<br><br>
                    Convergencia <strong>cuadrática</strong> cerca del mínimo.
                    Regularización automática cuando H no es definida positiva.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-label">📖 Ejemplos de funciones para probar</div>', unsafe_allow_html=True)

        ex_col1, ex_col2 = st.columns(2)
        with ex_col1:
            st.markdown("""
            <div class="card">
                <div class="card-title">2 variables</div>
                <code>(x1 - 2)**2 + (x2 - 3)**2</code>  → Cuadrática convexa<br><br>
                <code>100*(x2 - x1**2)**2 + (1 - x1)**2</code> → Rosenbrock<br><br>
                <code>(x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2</code> → Himmelblau
            </div>
            """, unsafe_allow_html=True)
        with ex_col2:
            st.markdown("""
            <div class="card">
                <div class="card-title">3+ variables</div>
                <code>x1**2 + 2*x2**2 + 3*x3**2 - x1*x2 + x2*x3</code><br><br>
                <code>(x1-1)**2 + (x2-2)**2 + (x3+1)**2 + (x4-3)**2</code><br><br>
                <code>sp.exp(x1) + x1**2 + x2**2 + x1*x2</code>
            </div>
            """, unsafe_allow_html=True)

        _ok = False

    # ── Validar inputs ────────────────────────────────────────────────────────
    if _ok:
        try:
            x0_vals = [float(v.strip()) for v in x0_str.split(',')]
            if len(x0_vals) != n_vars:
                st.error(f"❌ El punto de partida debe tener exactamente **{n_vars}** valor(es), separados por coma.")
                _ok = False
            else:
                x0 = np.array(x0_vals, dtype=float)
        except ValueError:
            st.error("❌ El punto de partida contiene valores inválidos. Use números separados por coma.")
            _ok = False

    if _ok and not (use_gd or use_cg or use_nt):
        st.warning("⚠️ Selecciona al menos un método en la barra lateral.")
        _ok = False

    if _ok and not (0 < c1 < c2 < 1):
        st.error("❌ Los parámetros de Wolfe deben satisfacer **0 < c₁ < c₂ < 1**.")
        _ok = False

    # ── Parsear función ───────────────────────────────────────────────────────
    if _ok:
        with st.spinner("⚙️ Calculando gradiente y Hessiana simbólicos..."):
            try:
                f, grad_f, hess_f, sym_vars, expr = parse_and_build(func_str, n_vars)
                _ = f(x0); _ = grad_f(x0)
            except Exception as e:
                st.error(f"❌ Error al procesar la función: {e}")
                _ok = False

    if _ok:
        # Mostrar función parseada en LaTeX
        latex_expr = sp.latex(expr)
        st.markdown("""
<div class="card" style="text-align:center; margin-bottom:4px;">
    <div class="card-title">Función objetivo reconocida</div>
</div>
""", unsafe_allow_html=True)
        st.latex(f"f(x) = {latex_expr}")
        st.markdown(f"""
<div class="card" style="text-align:center; margin-top:4px; margin-bottom:24px;">
    <div class="card-sub">Variables: {', '.join([str(v) for v in sym_vars])} &nbsp;|&nbsp;
    Punto de partida: x₀ = ({', '.join([str(v) for v in x0])})</div>
</div>
""", unsafe_allow_html=True)

        # ── Ejecutar métodos ──────────────────────────────────────────────────
        results   = {}
        histories = {}
        method_icons = {'Gradiente': '⬇️', 'Gradiente Conjugado': '🔀', 'Newton': '⚡'}

        with st.spinner("🚀 Ejecutando métodos de optimización..."):
            if use_gd:
                try:
                    x_opt, hist, conv = gradient_descent(
                        f, grad_f, x0, max_iter, tol, c1, c2, strong_wolfe)
                    results['Gradiente']   = {'x': x_opt, 'f': f(x_opt),
                                               'iters': len(hist)-1, 'converged': conv,
                                               'grad_norm': hist[-1]['grad_norm']}
                    histories['Gradiente'] = hist
                except Exception as e:
                    st.warning(f"⚠️ Descenso de Gradiente falló: {e}")

            if use_cg:
                try:
                    x_opt, hist, conv = conjugate_gradient(
                        f, grad_f, x0, max_iter, tol, c1, c2, strong_wolfe)
                    results['Gradiente Conjugado']   = {'x': x_opt, 'f': f(x_opt),
                                                         'iters': len(hist)-1, 'converged': conv,
                                                         'grad_norm': hist[-1]['grad_norm']}
                    histories['Gradiente Conjugado'] = hist
                except Exception as e:
                    st.warning(f"⚠️ Gradiente Conjugado falló: {e}")

            if use_nt:
                try:
                    x_opt, hist, conv = newton_method(
                        f, grad_f, hess_f, x0, max_iter, tol, c1, c2, strong_wolfe)
                    results['Newton']   = {'x': x_opt, 'f': f(x_opt),
                                            'iters': len(hist)-1, 'converged': conv,
                                            'grad_norm': hist[-1]['grad_norm']}
                    histories['Newton'] = hist
                except Exception as e:
                    st.warning(f"⚠️ Newton falló: {e}")

        if not results:
            st.error("❌ Ningún método pudo ejecutarse. Revisa la función y los parámetros.")
        else:
            # ── Resultados ────────────────────────────────────────────────────
            st.markdown('<div class="section-label">📊 Resultados</div>', unsafe_allow_html=True)

            cols = st.columns(len(results))
            for col, (method, res) in zip(cols, results.items()):
                with col:
                    icon  = method_icons.get(method, '🔧')
                    x_str = ', '.join([f'{v:.6f}' for v in res['x']])
                    st.markdown(f"### {icon} {method}")
                    if res['converged']:
                        st.success("✅ Convergió")
                    else:
                        st.error("⚠️ No convergió")
                    st.metric("f(x*) — valor óptimo",   f"{res['f']:.8f}")
                    st.metric("Iteraciones realizadas",  res['iters'])
                    st.metric("Error final ‖∇f(x*)‖",   f"{res['grad_norm']:.3e}")
                    st.caption(f"**Punto mínimo x*:** ({x_str})")
                    st.caption(f"**Criterio de parada:** {'‖∇f‖ ≤ ' + f'{tol:.0e}' if res['converged'] else 'Máx. iteraciones'}")
                    st.divider()

            # ── Gráfico de convergencia ───────────────────────────────────────
            st.markdown('<div class="section-label">📈 Gráfico de Convergencia</div>', unsafe_allow_html=True)
            st.plotly_chart(plot_convergence(histories), use_container_width=True)

            # ── Trayectoria 2D ────────────────────────────────────────────────
            if n_vars == 2:
                st.markdown('<div class="section-label">🗺️ Trayectoria en el Espacio de Búsqueda</div>',
                            unsafe_allow_html=True)
                best_method = min(results, key=lambda k: results[k]['f'])
                try:
                    contour_fig = plot_contour(f, histories, results[best_method]['x'])
                    st.plotly_chart(contour_fig, use_container_width=True)
                except Exception as e:
                    st.info(f"No se pudo generar el contorno: {e}")

            # ── Tabla de iteraciones ──────────────────────────────────────────
            with st.expander("📋 Ver tabla de iteraciones detallada"):
                for method, history in histories.items():
                    st.markdown(f"**{method_icons.get(method,'')} {method}** — {len(history)-1} iteraciones")
                    step = max(1, len(history) // 60)
                    rows = []
                    for h in history[::step]:
                        row = {
                            'Iteración': h['iter'],
                            'f(x)':      f"{h['f']:.8f}",
                            '‖∇f(x)‖':   f"{h['grad_norm']:.4e}",
                        }
                        for j, xi in enumerate(h['x']):
                            row[f'x{j+1}'] = f"{xi:.7f}"
                        rows.append(row)
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=260)
                    st.markdown("<br>", unsafe_allow_html=True)

            # ── Comparación resumen ───────────────────────────────────────────
            if len(results) > 1:
                st.markdown('<div class="section-label">🏆 Comparación de Eficiencia</div>', unsafe_allow_html=True)
                best_f    = min(results, key=lambda k: results[k]['f'])
                fewest_it = min(results, key=lambda k: results[k]['iters'])
                comp_rows = []
                for method, res in results.items():
                    comp_rows.append({
                        'Método':      method,
                        'Convergió':   '✅ Sí' if res['converged'] else '❌ No',
                        'Iteraciones': res['iters'],
                        'f(x*)':       f"{res['f']:.8f}",
                        '‖∇f(x*)‖':   f"{res['grad_norm']:.3e}",
                        'Eficiencia':  '🏆 Más rápido' if method == fewest_it else ('🎯 Mejor valor' if method == best_f else '—')
                    })
                st.dataframe(pd.DataFrame(comp_rows).set_index('Método'), use_container_width=True)

            # ── Footer ────────────────────────────────────────────────────────
            st.markdown("---")
            st.markdown("""
<div style="text-align:center; color:#484f58; font-size:0.82em; padding: 10px 0;">
    Implementado con Python · NumPy · SymPy · Plotly · Streamlit<br>
    Condiciones de Wolfe según <em>Nocedal & Wright — Numerical Optimization, 2ª ed.</em>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — OPTIMIZACIÓN RESTRINGIDA + KKT
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
<div class="app-header" style="margin-top:10px;">
    <div class="app-title" style="font-size:1.9em;">🔒 Optimización Restringida</div>
    <div class="app-sub">
        Multiplicadores de Lagrange · Condiciones KKT<br>
        <span style="color:#7c6af5; font-weight:600;">
            min f(x,y) &nbsp;s.a.&nbsp; g(x,y)=0 &nbsp;/&nbsp; h(x,y)≤0
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

    with st.expander("📖 ¿Qué son las condiciones KKT?", expanded=False):
        st.markdown("""
Las **condiciones de Karush-Kuhn-Tucker (KKT)** son condiciones necesarias de optimalidad
para problemas de optimización con restricciones. Generalizan los multiplicadores de Lagrange
al caso con restricciones de desigualdad.

Para el problema: **min f(x)** s.a. **gᵢ(x)=0**, **hⱼ(x)≤0**

Un punto x* es óptimo local solo si existen multiplicadores λᵢ y μⱼ tales que:

| # | Condición | Expresión |
|---|-----------|-----------|
| 1 | **Estacionariedad** | ∇f(x*) + Σλᵢ∇gᵢ(x*) + Σμⱼ∇hⱼ(x*) = 0 |
| 2 | **Factibilidad primal (igualdad)** | gᵢ(x*) = 0 |
| 3 | **Factibilidad primal (desigualdad)** | hⱼ(x*) ≤ 0 |
| 4 | **Factibilidad dual** | μⱼ ≥ 0 |
| 5 | **Holgura complementaria** | μⱼ · hⱼ(x*) = 0 |

La condición 1 dice que los **gradientes deben ser paralelos** en el óptimo,
lo que se puede ver visualmente en el gráfico.
""")

    st.markdown('<div class="section-label">⚙️ Configuración del problema restringido</div>',
                unsafe_allow_html=True)

    kkt_col1, kkt_col2 = st.columns(2)

    with kkt_col1:
        kkt_f    = st.text_input("Función objetivo f(x, y)",
                                  value="x**2 + y**2",
                                  key="kkt_f",
                                  help="Variables: x e y")
        kkt_x0   = st.text_input("Punto inicial (x, y)",
                                  value="0.5, 0.5", key="kkt_x0")
        n_eq_kkt = st.number_input("Nº restricciones de igualdad   g(x,y) = 0",
                                    min_value=0, max_value=3, value=1, key="n_eq_kkt")
        eq_inputs = []
        for i in range(int(n_eq_kkt)):
            val = "x + y - 1" if i == 0 else ""
            eq_inputs.append(st.text_input(f"g{i+1}(x,y) = 0", value=val, key=f"eq_kkt_{i}"))

    with kkt_col2:
        st.markdown("""
    <div class="card" style="margin-top:4px;">
        <div class="card-title">💡 Ejemplos para probar</div>
        <div class="card-sub">
            <b>Clásico Lagrange:</b><br>
            f = x**2 + y**2 &nbsp;|&nbsp; g = x+y-1=0<br><br>
            <b>Con desigualdad:</b><br>
            f = (x-3)**2+(y-2)**2 &nbsp;|&nbsp; h = x**2+y**2-4 ≤ 0<br><br>
            <b>Producción (Cobb-Douglas):</b><br>
            f = -(x**0.5 * y**0.5) &nbsp;|&nbsp; g = 2*x+3*y-12=0
        </div>
    </div>
    """, unsafe_allow_html=True)
        n_ineq_kkt = st.number_input("Nº restricciones de desigualdad   h(x,y) ≤ 0",
                                      min_value=0, max_value=3, value=0, key="n_ineq_kkt")
        ineq_inputs = []
        for i in range(int(n_ineq_kkt)):
            val = "x**2 + y**2 - 4" if i == 0 else ""
            ineq_inputs.append(st.text_input(f"h{i+1}(x,y) ≤ 0", value=val,
                                              key=f"ineq_kkt_{i}"))

    kkt_run = st.button("🔒 Resolver y verificar KKT", key="kkt_run")

    if kkt_run:
        _kkt_ok = True
        kkt_x0_arr = None

        try:
            kkt_x0_arr = np.array([float(v.strip()) for v in kkt_x0.split(',')])
            if len(kkt_x0_arr) != 2:
                st.error("❌ El punto inicial debe tener exactamente 2 valores.")
                _kkt_ok = False
        except ValueError:
            st.error("❌ Punto inicial inválido.")
            _kkt_ok = False

        if _kkt_ok and not (eq_inputs or ineq_inputs):
            st.warning("⚠️ Agrega al menos una restricción para usar esta sección.")
            _kkt_ok = False

        if _kkt_ok:
            with st.spinner("Resolviendo y verificando condiciones KKT..."):
                try:
                    sol = solve_kkt(kkt_f, eq_inputs, ineq_inputs, kkt_x0_arr)
                except Exception as e:
                    st.error(f"❌ Error al resolver: {e}")
                    _kkt_ok = False

        if _kkt_ok:
            st.markdown('<div class="section-label">📊 Solución</div>', unsafe_allow_html=True)
            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric("x*", f"({sol['x_star'][0]:.6f},  {sol['x_star'][1]:.6f})")
            with r2:
                st.metric("f(x*)", f"{sol['f_star']:.8f}")
            with r3:
                if sol['success']:
                    st.success("✅ Solución encontrada")
                else:
                    st.warning(f"⚠️ {sol['message']}")

            if sol['mults']:
                st.markdown('<div class="section-label">🔢 Multiplicadores</div>',
                            unsafe_allow_html=True)
                mcols = st.columns(len(sol['mults']))
                for i, (col, val) in enumerate(zip(mcols, sol['mults'])):
                    label = f"λ{i+1}" if i < sol['n_eq'] else f"μ{i-sol['n_eq']+1}"
                    tipo  = "Igualdad" if i < sol['n_eq'] else "Desigualdad"
                    with col:
                        st.metric(f"{label}  ({tipo})", f"{val:.6f}")

            st.markdown('<div class="section-label">✅ Verificación de Condiciones KKT</div>',
                        unsafe_allow_html=True)

            kkt_res = sol['kkt']

            def kkt_row(ok, val, name, formula, ideal):
                icon = "✅" if ok else "❌"
                return {"Condición": f"{icon} {name}", "Fórmula": formula,
                        "Valor": f"{val:.2e}", "¿Se cumple?": "Sí" if ok else "No",
                        "Ideal": ideal}

            kkt_rows = []
            ok1, v1 = kkt_res['stationarity']
            kkt_rows.append(kkt_row(ok1, v1, "Estacionariedad", "‖∇f + Σλ∇g + Σμ∇h‖", "≈ 0"))
            for i, (ok, v) in enumerate(kkt_res['eq_feasibility']):
                kkt_rows.append(kkt_row(ok, v, f"Factib. primal g{i+1}", f"g{i+1}(x*) = 0", "= 0"))
            for i, (ok, v) in enumerate(kkt_res['ineq_feasibility']):
                kkt_rows.append(kkt_row(ok, v, f"Factib. primal h{i+1}", f"h{i+1}(x*) ≤ 0", "≤ 0"))
            for i, (ok, v) in enumerate(kkt_res['dual_feasibility']):
                kkt_rows.append(kkt_row(ok, v, f"Factib. dual μ{i+1}", f"μ{i+1} ≥ 0", "≥ 0"))
            for i, (ok, v) in enumerate(kkt_res['comp_slackness']):
                kkt_rows.append(kkt_row(ok, v, f"Holgura complement. h{i+1}", f"|μ{i+1}·h{i+1}(x*)| ≈ 0", "≈ 0"))

            st.dataframe(pd.DataFrame(kkt_rows).set_index("Condición"), use_container_width=True)

            all_ok = ok1 and all(o for o,_ in kkt_res['eq_feasibility']) \
                         and all(o for o,_ in kkt_res['ineq_feasibility']) \
                         and all(o for o,_ in kkt_res['dual_feasibility']) \
                         and all(o for o,_ in kkt_res['comp_slackness'])

            if all_ok:
                st.success("🏆 El punto x* satisface **todas** las condiciones KKT — es un candidato a óptimo.")
            else:
                st.warning("⚠️ Alguna condición KKT no se satisface. El solver puede no haber convergido.")

            st.markdown('<div class="section-label">🗺️ Visualización</div>', unsafe_allow_html=True)
            try:
                st.plotly_chart(plot_kkt_2d(sol), use_container_width=True)
                st.caption("Las flechas muestran los vectores gradiente escalados en x*. "
                           "La condición de estacionariedad exige que **∇f = −Σλᵢ∇gᵢ − Σμⱼ∇hⱼ** "
                           "(los vectores deben cancelarse entre sí).")
            except Exception as e:
                st.info(f"No se pudo generar el gráfico: {e}")
