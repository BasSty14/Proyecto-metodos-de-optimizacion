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

    for i in range(max_iter):
        g      = grad_f(x)
        norm_g = np.linalg.norm(g)
        hist.append({'iter': i, 'x': x.copy(), 'f': f(x), 'grad_norm': norm_g})

        if norm_g <= tol:
            break

        d     = -g
        alpha = wolfe_line_search(f, grad_f, x, d, c1, c2, strong)
        x     = x + alpha * d

    g = grad_f(x)
    hist.append({'iter': len(hist), 'x': x.copy(), 'f': f(x), 'grad_norm': np.linalg.norm(g)})

    return x, hist, hist[-1]['grad_norm'] <= tol


def conjugate_gradient(f, grad_f, x0, max_iter, tol, c1, c2, strong):
    """
    Gradiente Conjugado No Lineal — Fórmula Polak-Ribière con reinicio.
    Reinicia la dirección cada n pasos o si β < 0.
    """
    x    = x0.copy()
    g    = grad_f(x)
    d    = -g.copy()
    n    = len(x0)
    hist = []

    for i in range(max_iter):
        norm_g = np.linalg.norm(g)
        hist.append({'iter': i, 'x': x.copy(), 'f': f(x), 'grad_norm': norm_g})

        if norm_g <= tol:
            break

        alpha = wolfe_line_search(f, grad_f, x, d, c1, c2, strong)
        x_new = x + alpha * d
        g_new = grad_f(x_new)

        # Fórmula Polak-Ribière
        gg = np.dot(g, g)
        beta = max(0.0, np.dot(g_new, g_new - g) / gg) if gg > 1e-30 else 0.0

        # Reinicio periódico cada n iteraciones
        if (i + 1) % n == 0:
            beta = 0.0

        d = -g_new + beta * d
        x = x_new
        g = g_new

    g = grad_f(x)
    hist.append({'iter': len(hist), 'x': x.copy(), 'f': f(x), 'grad_norm': np.linalg.norm(g)})

    return x, hist, hist[-1]['grad_norm'] <= tol


def newton_method(f, grad_f, hess_f, x0, max_iter, tol, c1, c2, strong):
    """
    Método de Newton con búsqueda de línea de Wolfe.
    Usa modificación de la Hessiana (Cholesky con perturbación)
    para garantizar dirección de descenso.
    """
    x    = x0.copy()
    hist = []

    for i in range(max_iter):
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


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Parámetros de Entrada")
    st.markdown("---")

    # Variables
    n_vars = st.number_input("🔢 Número de variables", min_value=1, max_value=50, value=2, step=1)
    n_vars = int(n_vars)

    # Métodos
    st.markdown("**🧮 Métodos a ejecutar**")
    use_gd = st.checkbox("Descenso de Gradiente",   value=True)
    use_cg = st.checkbox("Gradiente Conjugado",      value=True)
    use_nt = st.checkbox("Método de Newton",         value=True)

    st.markdown("---")

    # Función
    st.markdown("**📐 Función objetivo**")
    default_fn = "(x1 - 2)**2 + (x2 - 3)**2" if n_vars >= 2 else "(x1 - 2)**2"
    func_str = st.text_area(
        f"f(x₁{',...,x'+str(n_vars) if n_vars>1 else ''})",
        value=default_fn, height=80,
        help="Use x1, x2, ..., xn como variables.\nEjemplo: 100*(x2-x1**2)**2 + (1-x1)**2"
    )

    # Punto de partida
    st.markdown("**🎯 Punto de partida x₀**")
    x0_str = st.text_input("Valores separados por coma",
                            value=', '.join(['0.0'] * n_vars))

    st.markdown("---")

    # Parámetros de iteración
    st.markdown("**🔄 Parámetros de iteración**")
    max_iter = st.slider("Máx. iteraciones", 10, 2000, 300, 10)
    tol = st.select_slider(
        "Tolerancia de convergencia",
        options=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
        value=1e-6,
        format_func=lambda x: f"{x:.0e}"
    )

    # Wolfe
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


# ── Página de bienvenida ──────────────────────────────────────────────────────
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

    st.stop()


# ── Validar inputs ────────────────────────────────────────────────────────────
try:
    x0_vals = [float(v.strip()) for v in x0_str.split(',')]
    if len(x0_vals) != n_vars:
        st.error(f"❌ El punto de partida debe tener exactamente **{n_vars}** valor(es), separados por coma.")
        st.stop()
    x0 = np.array(x0_vals, dtype=float)
except ValueError:
    st.error("❌ El punto de partida contiene valores inválidos. Use números separados por coma.")
    st.stop()

if not (use_gd or use_cg or use_nt):
    st.warning("⚠️ Selecciona al menos un método en la barra lateral.")
    st.stop()

if not (0 < c1 < c2 < 1):
    st.error("❌ Los parámetros de Wolfe deben satisfacer **0 < c₁ < c₂ < 1**.")
    st.stop()


# ── Parsear función ───────────────────────────────────────────────────────────
with st.spinner("⚙️ Calculando gradiente y Hessiana simbólicos..."):
    try:
        f, grad_f, hess_f, sym_vars, expr = parse_and_build(func_str, n_vars)
        _ = f(x0); _ = grad_f(x0)  # test
    except Exception as e:
        st.error(f"❌ Error al procesar la función: {e}")
        st.stop()

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


# ── Ejecutar métodos ──────────────────────────────────────────────────────────
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
    st.stop()


# ── Resultados ────────────────────────────────────────────────────────────────
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

        


# ── Gráfico de convergencia ───────────────────────────────────────────────────
st.markdown('<div class="section-label">📈 Gráfico de Convergencia</div>', unsafe_allow_html=True)
st.plotly_chart(plot_convergence(histories), use_container_width=True)


# ── Trayectoria 2D ────────────────────────────────────────────────────────────
if n_vars == 2:
    st.markdown('<div class="section-label">🗺️ Trayectoria en el Espacio de Búsqueda</div>',
                unsafe_allow_html=True)
    best_method = min(results, key=lambda k: results[k]['f'])
    try:
        contour_fig = plot_contour(f, histories, results[best_method]['x'])
        st.plotly_chart(contour_fig, use_container_width=True)
    except Exception as e:
        st.info(f"No se pudo generar el contorno: {e}")


# ── Tabla de iteraciones ──────────────────────────────────────────────────────
with st.expander("📋 Ver tabla de iteraciones detallada"):
    for method, history in histories.items():
        st.markdown(f"**{method_icons.get(method,'')} {method}** — {len(history)-1} iteraciones")

        step = max(1, len(history) // 60)  # máximo ~60 filas
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


# ── Comparación resumen ───────────────────────────────────────────────────────
if len(results) > 1:
    st.markdown('<div class="section-label">🏆 Comparación de Eficiencia</div>', unsafe_allow_html=True)

    best_f    = min(results, key=lambda k: results[k]['f'])
    fewest_it = min(results, key=lambda k: results[k]['iters'])

    comp_rows = []
    for method, res in results.items():
        comp_rows.append({
            'Método':       method,
            'Convergió':    '✅ Sí' if res['converged'] else '❌ No',
            'Iteraciones':  res['iters'],
            'f(x*)':        f"{res['f']:.8f}",
            '‖∇f(x*)‖':    f"{res['grad_norm']:.3e}",
            'Eficiencia':   '🏆 Más rápido' if method == fewest_it else ('🎯 Mejor valor' if method == best_f else '—')
        })

    st.dataframe(pd.DataFrame(comp_rows).set_index('Método'),
                 use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#484f58; font-size:0.82em; padding: 10px 0;">
    Implementado con Python · NumPy · SymPy · Plotly · Streamlit<br>
    Condiciones de Wolfe según <em>Nocedal & Wright — Numerical Optimization, 2ª ed.</em>
</div>
""", unsafe_allow_html=True)
