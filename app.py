# -------------------------------------------------------------
#  📊 DASHBOARD FINANCIERO AVANZADO
# -------------------------------------------------------------
import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime  # <-- AÑADIDO

# Configuración global
st.set_page_config(
    page_title="📊 Dashboard Financiero Avanzado",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

# Parámetros editables
Rf = 0.0435   # riesgo libre
Rm = 0.085    # retorno mercado
Tc0 = 0.21    # tasa impositiva por defecto

# Orden de sectores
SECTOR_RANK = {
    "Consumer Defensive": 1,
    "Consumer Cyclical": 2,
    "Healthcare": 3,
    "Technology": 4,
    "Financial Services": 5,
    "Industrials": 6,
    "Communication Services": 7,
    "Energy": 8,
    "Real Estate": 9,
    "Utilities": 10,
    "Basic Materials": 11,
    "Unknown": 99,
}

MAX_TICKERS_PER_CHART = 10

# =============================================================
# FUNCIONES AUXILIARES
# =============================================================
def safe_first(obj):
    if obj is None:
        return None
    if hasattr(obj, "dropna"):
        obj = obj.dropna()
    return obj.iloc[0] if hasattr(obj, "iloc") and not obj.empty else obj

def seek_row(df, keys):
    for k in keys:
        if k in df.index:
            return df.loc[k]
    return pd.Series([0], index=df.columns[:1])

def format_number(x, decimals=2, is_percent=False):
    if pd.isna(x):
        return "N/D"
    if is_percent:
        return f"{x*100:.{decimals}f}%"
    return f"{x:.{decimals}f}"

def calc_ke(beta):
    return Rf + beta * (Rm - Rf)

def calc_kd(interest, debt):
    return interest / debt if debt else 0

def calc_wacc(mcap, debt, ke, kd, t):
    total = (mcap or 0) + (debt or 0)
    return (mcap/total)*ke + (debt/total)*kd*(1-t) if total else None

def cagr4(fin, metric):
    if metric not in fin.index:
        return None
    v = fin.loc[metric].dropna().iloc[:4]
    return (v.iloc[0]/v.iloc[-1])**(1/(len(v)-1))-1 if len(v)>1 and v.iloc[-1] else None

def chunk_df(df, size=MAX_TICKERS_PER_CHART):
    if df.empty:
        return []
    return [df.iloc[i:i+size] for i in range(0, len(df), size)]

def auto_ylim(ax, values, pad=0.10):
    """Ajuste automático del eje Y."""
    if isinstance(values, pd.DataFrame):
        arr = values.to_numpy(dtype="float64")
    else:
        arr = np.asarray(values, dtype="float64")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    if vmax == vmin:
        ymin = vmin - abs(vmin)*pad - 1
        ymax = vmax + abs(vmax)*pad + 1
        ax.set_ylim(ymin, ymax)
        return
    if vmin >= 0:
        ymin = 0
        ymax = vmax * (1 + pad)
    elif vmax <= 0:
        ymax = 0
        ymin = vmin * (1 + pad)
    else:
        m = max(abs(vmin), abs(vmax)) * (1 + pad)
        ymin, ymax = -m, m
    ax.set_ylim(ymin, ymax)

def add_value_labels(ax, spacing=5, formatter=None):
    """Agrega valores a las barras de un gráfico."""
    if formatter is None:
        formatter = lambda x: f"{x:.2f}"
    
    # Para cada barra en el gráfico
    for rect in ax.patches:
        # Obtener la posición y altura de la barra
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2
        
        # Determinar si el valor es negativo para posicionar la etiqueta correctamente
        space = spacing
        if y_value < 0:
            space *= -1
        
        # Formatear el valor
        label = formatter(y_value)
        
        # Crear anotación
        ax.annotate(
            label,
            (x_value, y_value),
            xytext=(0, space),
            textcoords="offset points",
            ha='center',
            va='bottom' if y_value >= 0 else 'top',
            fontsize=8,
            alpha=0.7
        )

def obtener_datos_financieros(tk, Tc_def):
    try:
        # Configurar headers para evitar bloqueos
        tkr = yf.Ticker(tk)
        
        # Intentar con diferentes métodos si falla
        try:
            info = tkr.info
        except:
            # Intentar obtener datos básicos de manera alternativa
            hist = tkr.history(period="1d")
            if hist.empty:
                return None
            
            # Crear info básica manualmente
            info = {
                "shortName": tk,
                "longName": tk,
                "country": "N/D",
                "industry": "N/D",
                "sector": "Unknown",
                "currentPrice": hist["Close"].iloc[-1] if not hist.empty else 0,
                "marketCap": 0,
                "beta": 1.0,
                "trailingPE": None,
                "priceToBook": None,
                "dividendYield": None,
                "payoutRatio": None,
                "currentRatio": None,
                "quickRatio": None,
                "debtToEquity": None,
                "longTermDebtToEquity": None,
                "operatingMargins": None,
                "profitMargins": None,
                "returnOnAssets": None,
                "returnOnEquity": None,
                "sharesOutstanding": 0
            }
        
        # Obtener datos financieros con manejo de errores
        try:
            bs = tkr.balance_sheet
        except:
            bs = pd.DataFrame()
            
        try:
            fin = tkr.financials
        except:
            fin = pd.DataFrame()
            
        try:
            cf = tkr.cashflow
        except:
            cf = pd.DataFrame()
        
        # Datos básicos
        beta = info.get("beta", 1) or 1
        ke = calc_ke(beta)
        
        # Manejo robusto de datos del balance sheet
        debt = 0
        cash = 0
        equity = 0
        
        if not bs.empty:
            debt = safe_first(seek_row(bs, ["Total Debt", "Long Term Debt"])) or info.get("totalDebt", 0)
            cash = safe_first(seek_row(bs, [
                "Cash And Cash Equivalents",
                "Cash And Cash Equivalents At Carrying Value",
                "Cash Cash Equivalents And Short Term Investments",
            ])) or 0
            equity = safe_first(seek_row(bs, ["Common Stock Equity", "Total Stockholder Equity"]))
        else:
            debt = info.get("totalDebt", 0) or 0
            cash = info.get("totalCash", 0) or 0
            equity = info.get("totalEquity", 0) or 0

        # Manejo robusto de datos financieros
        interest = 0
        ebt = 0
        tax_exp = 0
        ebit = 0
        
        if not fin.empty:
            interest = safe_first(seek_row(fin, ["Interest Expense"])) or 0
            ebt = safe_first(seek_row(fin, ["Ebt", "EBT"])) or 0
            tax_exp = safe_first(seek_row(fin, ["Income Tax Expense"])) or 0
            ebit = safe_first(seek_row(fin, ["EBIT", "Operating Income",
                                           "Earnings Before Interest and Taxes"])) or 0

        kd = calc_kd(interest, debt)
        tax = tax_exp / ebt if ebt else Tc_def
        mcap = info.get("marketCap", 0) or 0
        wacc = calc_wacc(mcap, debt, ke, kd, tax)

        nopat = ebit * (1 - tax) if ebit is not None else None
        invested = (equity or 0) + ((debt or 0) - (cash or 0))
        roic = nopat / invested if (nopat is not None and invested and invested != 0) else None
        
        # CALCULAR CREACIÓN DE VALOR (WACC vs ROIC) en lugar de EVA
        creacion_valor = (roic - wacc) * 100 if all(v is not None for v in (roic, wacc)) else None

        price = info.get("currentPrice")
        fcf = 0
        if not cf.empty:
            fcf = safe_first(seek_row(cf, ["Free Cash Flow"])) or 0
        shares = info.get("sharesOutstanding") or 1
        pfcf = price / (fcf/shares) if (fcf and shares and fcf != 0) else None

        # Cálculo de ratios
        current_ratio = info.get("currentRatio")
        quick_ratio = info.get("quickRatio")
        debt_eq = info.get("debtToEquity")
        lt_debt_eq = info.get("longTermDebtToEquity")
        oper_margin = info.get("operatingMargins")
        profit_margin = info.get("profitMargins")
        roa = info.get("returnOnAssets")
        roe = info.get("returnOnEquity")
        
        # Dividendos
        div_yield = info.get("dividendYield")
        payout = info.get("payoutRatio")
        
        # Crecimiento
        revenue_growth = cagr4(fin, "Total Revenue") if not fin.empty else None
        eps_growth = cagr4(fin, "Net Income") if not fin.empty else None
        fcf_growth = cagr4(cf, "Free Cash Flow") or cagr4(cf, "Operating Cash Flow") if not cf.empty else None

        # Obtener datos para la nueva sección de estructura de capital
        # Obtener datos históricos de balance sheet para los últimos 4 años
        balance_sheet_4y = pd.DataFrame()
        if not bs.empty:
            balance_sheet_4y = bs.iloc[:, :4] if len(bs.columns) >= 4 else bs
        
        # Extraer datos de estructura de capital
        try:
            total_assets = seek_row(balance_sheet_4y, ["Total Assets"])
            total_liabilities = seek_row(balance_sheet_4y, ["Total Liabilities Net Minority Interest", "Total Liabilities"])
            total_equity = seek_row(balance_sheet_4y, ["Total Equity Gross Minority Interest", "Stockholders Equity"])
            total_debt = seek_row(balance_sheet_4y, ["Total Debt", "Long Term Debt"])
        except:
            total_assets = pd.Series([0] * min(len(balance_sheet_4y.columns), 4), index=balance_sheet_4y.columns[:4] if not balance_sheet_4y.empty else [])
            total_liabilities = pd.Series([0] * min(len(balance_sheet_4y.columns), 4), index=balance_sheet_4y.columns[:4] if not balance_sheet_4y.empty else [])
            total_equity = pd.Series([0] * min(len(balance_sheet_4y.columns), 4), index=balance_sheet_4y.columns[:4] if not balance_sheet_4y.empty else [])
            total_debt = pd.Series([0] * min(len(balance_sheet_4y.columns), 4), index=balance_sheet_4y.columns[:4] if not balance_sheet_4y.empty else [])
        
        # Obtener datos para ratios de sostenibilidad de deuda
        income_statement_4y = pd.DataFrame()
        cash_flow_4y = pd.DataFrame()
        
        if not fin.empty:
            income_statement_4y = fin.iloc[:, :4] if len(fin.columns) >= 4 else fin
        
        if not cf.empty:
            cash_flow_4y = cf.iloc[:, :4] if len(cf.columns) >= 4 else cf
        
        try:
            ebitda = seek_row(income_statement_4y, ["EBITDA"])
            interest_expense = seek_row(income_statement_4y, ["Interest Expense"])
            operating_cash_flow = seek_row(cash_flow_4y, ["Operating Cash Flow"])
            capital_expenditure = seek_row(cash_flow_4y, ["Capital Expenditure"])
        except:
            ebitda = pd.Series([0] * min(len(income_statement_4y.columns), 4), index=income_statement_4y.columns[:4] if not income_statement_4y.empty else [])
            interest_expense = pd.Series([0] * min(len(income_statement_4y.columns), 4), index=income_statement_4y.columns[:4] if not income_statement_4y.empty else [])
            operating_cash_flow = pd.Series([0] * min(len(cash_flow_4y.columns), 4), index=cash_flow_4y.columns[:4] if not cash_flow_4y.empty else [])
            capital_expenditure = pd.Series([0] * min(len(cash_flow_4y.columns), 4), index=cash_flow_4y.columns[:4] if not cash_flow_4y.empty else [])
        
        # Calcular ratios de sostenibilidad de deuda para el último año
        debt_to_equity = None
        interest_coverage = None
        debt_to_ebitda = None
        cash_flow_to_debt = None
        leverage_ratio = None
        free_cash_flow = None
        fcf_to_debt = None
        
        if len(total_debt) > 0 and len(total_equity) > 0 and total_equity.iloc[0] != 0:
            debt_to_equity = total_debt.iloc[0] / total_equity.iloc[0]
            
        if len(ebitda) > 0 and len(interest_expense) > 0 and interest_expense.iloc[0] != 0:
            interest_coverage = ebitda.iloc[0] / abs(interest_expense.iloc[0])
            
        if len(total_debt) > 0 and len(ebitda) > 0 and ebitda.iloc[0] != 0:
            debt_to_ebitda = total_debt.iloc[0] / ebitda.iloc[0]
            
        if len(operating_cash_flow) > 0 and len(total_debt) > 0 and total_debt.iloc[0] != 0:
            cash_flow_to_debt = operating_cash_flow.iloc[0] / total_debt.iloc[0]
            
        if len(total_debt) > 0 and len(total_assets) > 0 and total_assets.iloc[0] != 0:
            leverage_ratio = total_debt.iloc[0] / total_assets.iloc[0]
            
        # Calcular Free Cash Flow
        if len(operating_cash_flow) > 0 and len(capital_expenditure) > 0:
            free_cash_flow = operating_cash_flow.iloc[0] + capital_expenditure.iloc[0]  # CapEx es negativo
            if total_debt.iloc[0] != 0:
                fcf_to_debt = free_cash_flow / total_debt.iloc[0]

        # Obtener datos de ingresos (facturación)
        total_revenue = 0
        if not fin.empty:
            total_revenue = safe_first(seek_row(fin, ["Total Revenue", "Revenue"])) or 0

        return {
            "Ticker": tk,
            "Nombre": info.get("longName") or info.get("shortName") or info.get("displayName") or tk,
            "País": info.get("country") or info.get("countryCode") or "N/D",
            "Industria": info.get("industry") or info.get("industryKey") or info.get("industryDisp") or "N/D",
            "Sector": info.get("sector", "Unknown"),
            "Precio": price,
            "P/E": info.get("trailingPE"),
            "P/B": info.get("priceToBook"),
            "P/FCF": pfcf,
            "Dividend Yield %": div_yield,
            "Payout Ratio": payout,
            "ROA": roa,
            "ROE": roe,
            "Current Ratio": current_ratio,
            "Quick Ratio": quick_ratio,
            "Debt/Eq": debt_eq,
            "LtDebt/Eq": lt_debt_eq,
            "Oper Margin": oper_margin,
            "Profit Margin": profit_margin,
            "WACC": wacc,
            "ROIC": roic,
            "Creacion Valor (Wacc vs Roic)": creacion_valor,
            "Revenue Growth": revenue_growth,
            "EPS Growth": eps_growth,
            "FCF Growth": fcf_growth,
            "MarketCap": mcap,
            # Nuevos datos para la sección de estructura de capital
            "Total Assets": total_assets,
            "Total Liabilities": total_liabilities,
            "Total Equity": total_equity,
            "Total Debt": total_debt,
            "EBITDA": ebitda,
            "Interest Expense": interest_expense,
            "Operating Cash Flow": operating_cash_flow,
            "Capital Expenditure": capital_expenditure,
            "Debt to Equity": debt_to_equity,
            "Interest Coverage": interest_coverage,
            "Debt to EBITDA": debt_to_ebitda,
            "Cash Flow to Debt": cash_flow_to_debt,
            "Leverage Ratio": leverage_ratio,
            "FCF to Debt": fcf_to_debt,
            # Datos para la nueva sección de sostenibilidad de deuda
            "Total Revenue": total_revenue,
            "Cash And Cash Equivalents": cash,
        }
    except Exception as e:
        st.error(f"Error obteniendo datos para {tk}: {str(e)}")
        return None

# =============================================================
# INTERFAZ PRINCIPAL
# =============================================================
def main():
    st.title("📊 Dashboard de Análisis Financiero Avanzado")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")
        t_in = st.text_area("Tickers (separados por comas)", 
                          "AAPL, MSFT, GOOGL, AMZN, TSLA, JNJ, JPM, V, WMT, PG")
        max_t = st.slider("Máximo de tickers", 1, 100, 10)
        
        st.markdown("---")
        st.markdown("**Parámetros WACC**")
        global Rf, Rm, Tc0
        Rf = st.number_input("Tasa libre de riesgo (%)", 0.0, 20.0, 4.35)/100
        Rm = st.number_input("Retorno esperado del mercado (%)", 0.0, 30.0, 8.5)/100
        Tc0 = st.number_input("Tasa impositiva corporativa (%)", 0.0, 50.0, 21.0)/100
        
        st.markdown("---")
        st.markdown("**Nota:** Si algunos tickers fallan, intente:")
        st.markdown("1. Verificar que el ticker existe")
        st.markdown("2. Reducir el número de tickers")
        st.markdown("3. Esperar y reintentar")

    if st.button("🔍 Analizar Acciones", type="primary"):
        tickers = [t.strip().upper() for t in t_in.split(",") if t.strip()][:max_t]
        
        # Validar tickers
        valid_tickers = []
        for tk in tickers:
            if len(tk) > 0 and tk.isalpha():
                valid_tickers.append(tk)
            else:
                st.warning(f"Ticker inválido ignorado: {tk}")
        
        if not valid_tickers:
            st.error("No hay tickers válidos para analizar")
            return
            
        tickers = valid_tickers
        
        # Obtener datos
        datos = []
        errs = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        with st.spinner("Obteniendo datos financieros..."):
            for i, tk in enumerate(tickers):
                try:
                    status_text.text(f"⏳ Procesando {tk} ({i+1}/{len(tickers)})...")
                    data = obtener_datos_financieros(tk, Tc0)
                    if data:
                        datos.append(data)
                    else:
                        errs.append({"Ticker": tk, "Error": "No se pudieron obtener datos"})
                except Exception as e:
                    errs.append({"Ticker": tk, "Error": str(e)})
                progress_bar.progress((i + 1) / len(tickers))
                time.sleep(0.5)  # Reducido para mayor velocidad

        status_text.text("✅ Análisis completado!")
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()

        if not datos:
            st.error("No se pudieron obtener datos para los tickers proporcionados")
            if errs:
                st.table(pd.DataFrame(errs))
            return

        df = pd.DataFrame(datos)
        df["SectorRank"] = df["Sector"].map(SECTOR_RANK).fillna(99).astype(int)
        df = df.sort_values(["SectorRank", "Sector", "Ticker"])
        
        # Formatear valores para visualización
        df_disp = df.copy()
        
        # Columnas con 2 decimales
        for col in ["P/E", "P/B", "P/FCF", "Current Ratio", "Quick Ratio", "Debt/Eq", "LtDebt/Eq"]:
            df_disp[col] = df_disp[col].apply(lambda x: format_number(x, 2))
            
        # Porcentajes con 2 decimales
        for col in ["Dividend Yield %", "Payout Ratio", "ROA", "ROE", "Oper Margin", 
                   "Profit Margin", "WACC", "ROIC", "Revenue Growth", "EPS Growth", "FCF Growth"]:
            df_disp[col] = df_disp[col].apply(lambda x: format_number(x, 2, is_percent=True))
            
        # Creación de Valor con 2 decimales y porcentaje
        df_disp["Creacion Valor (Wacc vs Roic)"] = df_disp["Creacion Valor (Wacc vs Roic)"].apply(
            lambda x: format_number(x/100, 2, is_percent=True) if pd.notnull(x) else "N/D"
        )
            
        # Precio y MarketCap con 2 decimales
        df_disp["Precio"] = df_disp["Precio"].apply(lambda x: f"${float(x):,.2f}" if pd.notnull(x) else "N/D")
        df_disp["MarketCap"] = df_disp["MarketCap"].apply(lambda x: f"${float(x)/1e9:,.2f}B" if pd.notnull(x) else "N/D")
        
        # Formatear nuevos ratios de sostenibilidad de deuda
        for col in ["Debt to Equity", "Interest Coverage", "Debt to EBITDA", 
                   "Cash Flow to Debt", "Leverage Ratio", "FCF to Debt"]:
            df_disp[col] = df_disp[col].apply(lambda x: format_number(x, 2) if pd.notnull(x) else "N/D")
        
        # Asegurar que las columnas de texto no sean None
        for c in ["Nombre", "País", "Industria"]:
            df_disp[c] = df_disp[c].fillna("N/D").replace({None: "N/D", "": "N/D"})

        # =====================================================
        # SECCIÓN 1: RESUMEN GENERAL
        # =====================================================
        st.header("📋 Resumen General (agrupado por Sector)")
        
        # Mostrar tabla
        st.dataframe(
            df_disp[[
                "Ticker", "Nombre", "País", "Industria", "Sector",
                "Precio", "P/E", "P/B", "P/FCF",
                "Dividend Yield %", "Payout Ratio", "ROA", "ROE",
                "Current Ratio", "Debt/Eq", "Oper Margin", "Profit Margin",
                "WACC", "ROIC", "Creacion Valor (Wacc vs Roic)", "MarketCap"
            ]],
            use_container_width=True,
            height=500
        )

        if errs:
            st.subheader("🚫 Tickers con error")
            st.table(pd.DataFrame(errs))

        sectors_ordered = df["Sector"].unique()

        # =====================================================
        # SECCIÓN 2: ANÁLISIS DE VALORACIÓN
        # =====================================================
        st.header("💰 Análisis de Valoración (por Sector)")
        
        for sec in sectors_ordered:
            sec_df = df[df["Sector"] == sec]
            if sec_df.empty:
                continue
                
            with st.expander(f"Sector: {sec} ({len(sec_df)} empresas)", expanded=False):
                fig, ax = plt.subplots(figsize=(10, 4))
                val = sec_df[["Ticker", "P/E", "P/B", "P/FCF"]].set_index("Ticker").apply(pd.to_numeric, errors="coerce")
                val.plot(kind="bar", ax=ax, rot=45)
                ax.set_ylabel("Ratio")
                auto_ylim(ax, val)
                # AÑADIR VALORES A LAS BARRAS
                add_value_labels(ax)
                st.pyplot(fig)
                plt.close()

        # =============================================================
        # SECCIÓN 3: RENTABILIDAD Y EFICIENCIA
        # =============================================================
        st.header("📈 Rentabilidad y Eficiencia")

        tabs = st.tabs(["ROE vs ROA", "Márgenes", "WACC vs ROIC"])

        with tabs[0]:
            for sec in sectors_ordered:
                sec_df = df[df["Sector"] == sec]
                if sec_df.empty:
                    continue
                    
                with st.expander(f"Sector: {sec}", expanded=False):
                    fig, ax = plt.subplots(figsize=(10, 5))
                    rr = pd.DataFrame({
                        "ROE": (sec_df["ROE"]*100).values,
                        "ROA": (sec_df["ROA"]*100).values
                    }, index=sec_df["Ticker"])
                    rr.plot(kind="bar", ax=ax, rot=45)
                    ax.set_ylabel("%")
                    auto_ylim(ax, rr)
                    # AÑADIR VALORES A LAS BARRAS
                    add_value_labels(ax, formatter=lambda x: f"{x:.1f}%")
                    st.pyplot(fig)
                    plt.close()

        with tabs[1]:
            for sec in sectors_ordered:
                sec_df = df[df["Sector"] == sec]
                if sec_df.empty:
                    continue
                    
                with st.expander(f"Sector: {sec}", expanded=False):
                    fig, ax = plt.subplots(figsize=(10, 5))
                    mm = pd.DataFrame({
                        "Oper Margin": (sec_df["Oper Margin"]*100).values,
                        "Profit Margin": (sec_df["Profit Margin"]*100).values
                    }, index=sec_df["Ticker"])
                    mm.plot(kind="bar", ax=ax, rot=45)
                    ax.set_ylabel("%")
                    auto_ylim(ax, mm)
                    # AÑADIR VALORES A LAS BARRAS
                    add_value_labels(ax, formatter=lambda x: f"{x:.1f}%")
                    st.pyplot(fig)
                    plt.close()

        with tabs[2]:
            # WACC vs ROIC dividido por sectores
            for sec in sectors_ordered:
                sec_df = df[df["Sector"] == sec]
                if sec_df.empty:
                    continue
                    
                with st.expander(f"Sector: {sec} - WACC vs ROIC", expanded=False):
                    fig, ax = plt.subplots(figsize=(12, 6))
                    rw = pd.DataFrame({
                        "ROIC": (sec_df["ROIC"]*100).values,
                        "WACC": (sec_df["WACC"]*100).values
                    }, index=sec_df["Ticker"])
                    rw.plot(kind="bar", ax=ax, rot=45)
                    ax.set_ylabel("%")
                    ax.set_title(f"Creación de Valor: ROIC vs WACC - Sector {sec}")
                    auto_ylim(ax, rw)
                    
                    # Añadir línea de referencia en 0
                    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    
                    # Calcular y mostrar creación/destrucción de valor promedio del sector
                    valor_creado = (sec_df["ROIC"] - sec_df["WACC"]).mean() * 100
                    color = "green" if valor_creado > 0 else "red"
                    ax.axhline(y=valor_creado, color=color, linestyle='--', alpha=0.7, 
                              label=f'Valor creado promedio: {valor_creado:.2f}%')
                    ax.legend()
                    
                    # AÑADIR VALORES A LAS BARRAS
                    add_value_labels(ax, formatter=lambda x: f"{x:.1f}%")
                    
                    st.pyplot(fig)
                    plt.close()
                    
                    # Mostrar estadísticas del sector
                    st.markdown(f"**Estadísticas del Sector {sec}:**")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        avg_roic = sec_df["ROIC"].mean() * 100
                        avg_wacc = sec_df["WACC"].mean() * 100
                        st.metric("ROIC Promedio", f"{avg_roic:.2f}%")
                        st.metric("WACC Promedio", f"{avg_wacc:.2f}%")
                    
                    with col2:
                        median_roic = sec_df["ROIC"].median() * 100
                        median_wacc = sec_df["WACC"].median() * 100
                        st.metric("ROIC Mediano", f"{median_roic:.2f}%")
                        st.metric("WACC Mediano", f"{median_wacc:.2f}%")
                    
                    with col3:
                        empresas_creadoras = (sec_df["ROIC"] > sec_df["WACC"]).sum()
                        total_empresas = len(sec_df)
                        porcentaje_creadoras = (empresas_creadoras / total_empresas) * 100
                        st.metric("Empresas que crean valor", f"{empresas_creadoras}/{total_empresas} ({porcentaje_creadoras:.1f}%)")

        # =============================================================
        # SECCIÓN 4: ESTRUCTURA DE CAPITAL Y SOSTENIBILIDAD DE DEUDA
        # =============================================================
        st.header("🏦 Estructura de Capital y Sostenibilidad de Deuda")

        # Crear pestañas para la nueva sección
        tab_estructura, tab_sostenibilidad = st.tabs([
            "Evolución de Activos, Pasivos y Patrimonio Neto (por Sector)",
            "Ratios de Sostenibilidad de Deuda (por Sector)"
        ])

        with tab_estructura:
            st.subheader("Evolución de Activos, Pasivos y Patrimonio Neto (por Sector)")
            
            for sec in sectors_ordered:
                sec_df = df[df["Sector"] == sec]
                if sec_df.empty:
                    continue
                    
                with st.expander(f"Sector: {sec} - Estructura de Capital", expanded=False):
                    for i, chunk in enumerate(chunk_df(sec_df), 1):
                        st.caption(f"Bloque {i}")
                        
                        # Crear gráficos para cada empresa en el chunk
                        for _, empresa in chunk.iterrows():
                            st.markdown(f"**{empresa['Ticker']} - {empresa['Nombre']}**")
                            
                            # Obtener datos para los últimos 4 años
                            total_assets = empresa["Total Assets"]
                            total_liabilities = empresa["Total Liabilities"]
                            total_equity = empresa["Total Equity"]
                            
                            if total_assets.empty or total_liabilities.empty or total_equity.empty:
                                st.warning("Datos insuficientes para mostrar la estructura de capital")
                                continue
                            
                            # Crear gráfico de barras
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            # Preparar datos para el gráfico
                            years = []
                            for col in total_assets.index:
                                if hasattr(col, 'strftime'):
                                    years.append(col.strftime('%Y'))
                                else:
                                    years.append(str(col))
                            x_pos = np.arange(len(years))
                            width = 0.25
                            
                            # Graficar barras
                            bars1 = ax.bar(x_pos - width, total_assets.values / 1e6, width, label='Activos Totales', color='#0074D9')
                            bars2 = ax.bar(x_pos, total_liabilities.values / 1e6, width, label='Pasivos Totales', color='#FF4136')
                            bars3 = ax.bar(x_pos + width, total_equity.values / 1e6, width, label='Patrimonio Neto', color='#2ECC40')
                            
                            # Configurar el gráfico
                            ax.set_xlabel('Año')
                            ax.set_ylabel('Millones USD')
                            ax.set_title(f'Estructura de Capital - {empresa["Ticker"]}')
                            ax.set_xticks(x_pos)
                            ax.set_xticklabels(years)
                            ax.legend()
                            
                            # Añadir valores en las barras
                            for bars in [bars1, bars2, bars3]:
                                for bar in bars:
                                    height = bar.get_height()
                                    ax.annotate(f'{height:,.0f}',
                                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                                xytext=(0, 3),
                                                textcoords="offset points",
                                                ha='center', va='bottom', fontsize=8)
                            
                            st.pyplot(fig)
                            plt.close()

        with tab_sostenibilidad:
            st.subheader("Ratios de Sostenibilidad de Deuda (por Sector)")
            
            # Definir umbrales para los ratios
            umbrales = {
                'Debt to Equity': 1.0,
                'Interest Coverage': 3.0,
                'Debt to EBITDA': 3.0,
                'Cash Flow to Debt': 0.2,
                'FCF to Debt': 0.1
            }
            
            # Explicación detallada de los ratios y umbrales
            with st.expander("ℹ️ Explicación Detallada de los Ratios y Umbrales", expanded=False):
                st.markdown("""
                ### 📊 Ratios de Sostenibilidad de Deuda
                
                **¿Qué es un umbral?** Un **umbral** es un valor de referencia que se utiliza para evaluar si un ratio financiero 
                indica una situación saludable o problemática. Es como una línea divisoria entre lo aceptable y lo riesgoso.
                
                ---
                
                **1. Debt to Equity (Deuda/Patrimonio)**
                - **Fórmula**: Deuda Total / Patrimonio Neto
                - **Umbral**: < 1.0
                - **Explicación**: Indica cuánta deuda tiene la empresa en relación con su capital propio. 
                  Un valor menor a 1 significa que la empresa tiene más capital que deuda, lo que es favorable.
                - **Interpretación**: 
                  - ✅ < 1.0: Estructura de capital conservadora
                  - ⚠️ 1.0-2.0: Apalancamiento moderado
                  - ❌ > 2.0: Alto apalancamiento, mayor riesgo
                
                **2. Interest Coverage (Cobertura de Intereses)**
                - **Fórmula**: EBITDA / Gastos por Intereses
                - **Umbral**: > 3.0
                - **Explicación**: Mide la capacidad de la empresa para pagar sus intereses con sus ganancias operativas. 
                  Un valor mayor a 3 indica que genera suficientes ganancias para cubrir cómodamente sus obligaciones de intereses.
                - **Interpretación**:
                  - ✅ > 3.0: Cómoda capacidad de pago de intereses
                  - ⚠️ 1.5-3.0: Capacidad adecuada pero limitada
                  - ❌ < 1.5: Dificultad para pagar intereses
                
                **3. Debt to EBITDA (Deuda/EBITDA)**
                - **Fórmula**: Deuda Total / EBITDA
                - **Umbral**: < 3.0
                - **Explicación**: Indica cuántos años le tomaría a la empresa pagar toda su deuda usando su EBITDA actual. 
                  Menos de 3 años se considera manejable.
                - **Interpretación**:
                  - ✅ < 3.0: Deuda manejable en relación a ganancias
                  - ⚠️ 3.0-5.0: Nivel de deuda moderadamente alto
                  - ❌ > 5.0: Deuda excesiva en relación to ganancias
                
                **4. Cash Flow to Debt (Flujo de Caja/Deuda)**
                - **Fórmula**: Flujo de Caja Operativo / Deuda Total
                - **Umbral**: > 0.2 (20%)
                - **Explicación**: Mide qué porcentaje de la deuda podría pagarse con el flujo de caja operativo en un año. 
                  Más del 20% es saludable.
                - **Interpretación**:
                  - ✅ > 0.2: Alta capacidad de generar cash flow para pagar deuda
                  - ⚠️ 0.1-0.2: Capacidad moderada
                  - ❌ < 0.1: Baja capacidad para servir la deuda con cash flow operativo
                
                **5. FCF to Debt (Free Cash Flow/Deuda)**
                - **Fórmula**: Free Cash Flow / Deuda Total
                - **Umbral**: > 0.1 (10%)
                - **Explicación**: Similar al anterior pero usando el Free Cash Flow (flujo de caja libre), 
                  que es el dinero disponible después de inversiones en capital.
                - **Interpretación**:
                  - ✅ > 0.1: Buena generación de cash flow libre para pagar deuda
                  - ⚠️ 0.05-0.1: Capacidad moderada
                  - ❌ < 0.05: Baja generación de cash flow libre para servir la deuda
                
                ---
                
                **Nota**: Estos umbrales pueden variar según la industria. Algunos sectores como utilities o bienes raíces 
                suelen operar con niveles de deuda más altos de manera normal.
                """)
            
            for sec in sectors_ordered:
                sec_df = df[df["Sector"] == sec]
                if sec_df.empty:
                    continue
                    
                with st.expander(f"Sector: {sec} - Sostenibilidad de Deuda", expanded=False):
                    # Mostrar tabla de ratios
                    ratios_df = sec_df[["Ticker", "Debt to Equity", "Interest Coverage", 
                                      "Debt to EBITDA", "Cash Flow to Debt", "FCF to Debt"]].copy()
                    
                    # Calcular evaluación de sostenibilidad
                    evaluacion = []
                    for _, row in ratios_df.iterrows():
                        sostenible = all([
                            pd.notnull(row.get("Debt to Equity")) and row.get("Debt to Equity") < umbrales['Debt to Equity'],
                            pd.notnull(row.get("Interest Coverage")) and row.get("Interest Coverage") > umbrales['Interest Coverage'],
                            pd.notnull(row.get("Debt to EBITDA")) and row.get("Debt to EBITDA") < umbrales['Debt to EBITDA'],
                            pd.notnull(row.get("Cash Flow to Debt")) and row.get("Cash Flow to Debt") > umbrales['Cash Flow to Debt'],
                            pd.notnull(row.get("FCF to Debt")) and row.get("FCF to Debt") > umbrales['FCF to Debt']
                        ])
                        evaluacion.append("✅ Sostenible" if sostenible else "❌ No Sostenible")
                    
                    ratios_df["Evaluación"] = evaluacion
                    st.dataframe(ratios_df.set_index("Ticker"), use_container_width=True)
                    
                    # Gráfico de ratios comparativos
                    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                    axes = axes.flatten()
                    
                    ratios = ["Debt to Equity", "Interest Coverage", "Debt to EBITDA", 
                             "Cash Flow to Debt", "FCF to Debt"]
                    
                    for i, ratio in enumerate(ratios):
                        if i < len(axes):
                            ax = axes[i]
                            values = sec_df[ratio].values
                            tickers = sec_df["Ticker"].values
                            
                            # Filtrar valores nulos
                            mask = ~pd.isnull(values)
                            filtered_values = values[mask]
                            filtered_tickers = tickers[mask]
                            
                            if len(filtered_values) > 0:
                                bars = ax.bar(filtered_tickers, filtered_values)
                                ax.set_title(ratio)
                                ax.tick_params(axis='x', rotation=45)
                                
                                # Añadir línea de umbral
                                if ratio in umbrales:
                                    ax.axhline(y=umbrales[ratio], color='r', linestyle='--', alpha=0.7)
                                    ax.text(0.02, 0.98, f'Umbral: {umbrales[ratio]}', 
                                           transform=ax.transAxes, verticalalignment='top',
                                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                                
                                # AÑADIR VALORES A LAS BARRAS
                                for j, (ticker, value) in enumerate(zip(filtered_tickers, filtered_values)):
                                    ax.annotate(f'{value:.2f}',
                                                xy=(j, value),
                                                xytext=(0, 3),
                                                textcoords="offset points",
                                                ha='center', va='bottom', fontsize=8, alpha=0.7)
                    
                    # Ocultar subplots vacíos
                    for i in range(len(ratios), len(axes)):
                        axes[i].set_visible(False)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    # Resumen del sector
                    st.subheader(f"Resumen de Sostenibilidad - Sector {sec}")
                    total_empresas = len(sec_df)
                    sostenibles = sum(1 for eval in evaluacion if eval == "✅ Sostenible")
                    porcentaje_sostenible = (sostenibles / total_empresas) * 100 if total_empresas > 0 else 0
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Empresas con deuda sostenible", f"{sostenibles}/{total_empresas}")
                    with col2:
                        st.metric("Porcentaje", f"{porcentaje_sostenible:.1f}%")
                    
                    if porcentaje_sostenible > 70:
                        st.success("✅ Este sector muestra una buena salud financiera en general")
                    elif porcentaje_sostenible > 40:
                        st.warning("⚠️ Este sector muestra una salud financiera mixta")
                    else:
                        st.error("❌ Este sector muestra problemas de sostenibilidad de deuda")
                        
                    # Nota sobre variaciones sectoriales
                    if sec in ["Utilities", "Real Estate", "Energy"]:
                        st.info(f"💡 Nota: El sector {sec} típicamente opera con niveles de deuda más altos que otros sectores. " 
                               "Los umbrales estándar pueden ser demasiado conservadores para este sector.")

        # =============================================================
        # NUEVA SECCIÓN: SOSTENIBILIDAD DE DEUDA Y MÚSCULO FINANCIERO
        # =============================================================
        st.header("💪 Sostenibilidad de Deuda y Músculo Financiero")

        # Explicación de los conceptos
        with st.expander("📚 Explicación de los Conceptos Clave", expanded=False):
            st.markdown("""
            ### 📊 Análisis de Sostenibilidad de Deuda
            
            **Deuda Neta = Deuda Total - Efectivo Disponible**
            - Indica la obligación financiera real después de considerar la liquidez inmediata.
            - Una deuda neta negativa (más efectivo que deuda) indica fortaleza financiera.
            
            **Deuda Neta vs Facturación Anual**
            - Mide el tamaño de la deuda en relación a la capacidad generadora de ingresos.
            - Un ratio bajo (<10-15%) indica deuda manejable.
            - Un ratio alto (>30-40%) puede indicar sobreendeudamiento.
            
            **Músculo Financiero = Efectivo > Deuda Neta**
            - Cuando una empresa tiene suficiente efectivo para cubrir toda su deuda.
            - Proporciona resiliencia ante crisis económicas.
            """)

        # Crear pestañas para el análisis
        tab_deuda_neta, tab_facturacion = st.tabs([
            "Deuda Neta vs Posición de Caja",
            "Deuda vs Capacidad de Generación de Ingresos"
        ])

        with tab_deuda_neta:
            st.subheader("Análisis de Deuda Neta y Posición de Caja")
            
            # Calcular métricas para cada empresa
            deuda_metrics = []
            for _, empresa in df.iterrows():
                try:
                    # Obtener el ticker para yfinance
                    ticker = empresa["Ticker"]
                    tkr = yf.Ticker(ticker)
                    
                    # Obtener balance sheet más reciente
                    balance_sheet = tkr.balance_sheet
                    if balance_sheet.empty:
                        # Intentar usar los datos ya obtenidos
                        total_debt = empresa.get("Total Debt", pd.Series([0])).iloc[0] if hasattr(empresa.get("Total Debt"), 'iloc') else empresa.get("Total Debt", 0)
                        cash = empresa.get("Cash And Cash Equivalents", 0)
                    else:
                        # Obtener datos del último período disponible
                        latest_date = balance_sheet.columns[0]
                        
                        # Buscar deuda total
                        total_debt = 0
                        debt_keys = ["Total Debt", "Long Term Debt", "Total Liabilities Net Minority Interest"]
                        for key in debt_keys:
                            if key in balance_sheet.index:
                                total_debt = balance_sheet.loc[key, latest_date]
                                break
                        
                        # Buscar efectivo y equivalentes
                        cash = 0
                        cash_keys = [
                            "Cash And Cash Equivalents", 
                            "Cash And Cash Equivalents At Carrying Value",
                            "Cash Cash Equivalents And Short Term Investments"
                        ]
                        for key in cash_keys:
                            if key in balance_sheet.index:
                                cash = balance_sheet.loc[key, latest_date]
                                break
                    
                    # Asegurar que son números
                    total_debt = float(total_debt) if total_debt else 0
                    cash = float(cash) if cash else 0
                    
                    # Calcular deuda neta
                    net_debt = total_debt - cash
                    
                    # Determinar posición financiera
                    if cash > total_debt:
                        position = "✅ Posición Fuerte (Caja > Deuda)"
                        position_color = "green"
                    elif net_debt == 0:
                        position = "⚖️ Equilibrio Perfecto (Caja = Deuda)"
                        position_color = "blue"
                    else:
                        position = "⚠️ Deuda Neta Positiva"
                        position_color = "orange" if total_debt > 0 and (net_debt/total_debt < 0.5) else "red"
                    
                    deuda_metrics.append({
                        "Ticker": ticker,
                        "Deuda Total": total_debt,
                        "Caja": cash,
                        "Deuda Neta": net_debt,
                        "Posición": position,
                        "Color": position_color
                    })
                    
                except Exception as e:
                    # Usar datos existentes si están disponibles
                    total_debt = empresa.get("Total Debt", 0)
                    if hasattr(total_debt, 'iloc'):
                        total_debt = total_debt.iloc[0] if not total_debt.empty else 0
                    
                    cash = empresa.get("Cash And Cash Equivalents", 0)
                    
                    net_debt = float(total_debt) - float(cash)
                    
                    if cash > total_debt:
                        position = "✅ Posición Fuerte (Caja > Deuda)"
                        position_color = "green"
                    elif net_debt == 0:
                        position = "⚖️ Equilibrio Perfecto (Caja = Deuda)"
                        position_color = "blue"
                    else:
                        position = "⚠️ Deuda Neta Positiva"
                        position_color = "orange" if total_debt > 0 and (net_debt/total_debt < 0.5) else "red"
                    
                    deuda_metrics.append({
                        "Ticker": ticker,
                        "Deuda Total": total_debt,
                        "Caja": cash,
                        "Deuda Neta": net_debt,
                        "Posición": position,
                        "Color": position_color
                    })
            
            if deuda_metrics:
                deuda_df = pd.DataFrame(deuda_metrics)
                
                # Mostrar tabla resumen
                st.dataframe(
                    deuda_df[["Ticker", "Deuda Total", "Caja", "Deuda Neta", "Posición"]].style.apply(
                        lambda x: [f"background-color: {deuda_df.loc[x.name, 'Color']}; color: white" 
                                  if deuda_df.loc[x.name, "Color"] in ["red", "orange"] else "" 
                                  for _ in x], axis=1
                    ),
                    use_container_width=True,
                    height=400
                )
                
                # Gráfico de comparación
                fig, ax = plt.subplots(figsize=(12, 8))
                
                x = range(len(deuda_df))
                width = 0.35
                
                # Convertir a millones para mejor visualización
                deuda_total_m = deuda_df["Deuda Total"] / 1e6
                caja_m = deuda_df["Caja"] / 1e6
                
                bars1 = ax.bar(x, deuda_total_m, width, label='Deuda Total', color='#FF4136', alpha=0.8)
                bars2 = ax.bar(x, caja_m, width, label='Caja', color='#2ECC40', alpha=0.8)
                
                ax.set_xlabel('Empresas')
                ax.set_ylabel('Millones USD')
                ax.set_title('Comparación Deuda Total vs Caja')
                ax.set_xticks(x)
                ax.set_xticklabels(deuda_df["Ticker"], rotation=45, ha='right')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Añadir valores a las barras
                for i, (deuda_val, caja_val) in enumerate(zip(deuda_total_m, caja_m)):
                    if deuda_val > 0:
                        ax.text(i, deuda_val + max(deuda_total_m.max() * 0.01, 5), 
                               f'{deuda_val:,.0f}M', ha='center', va='bottom', fontsize=8)
                    if caja_val > 0:
                        ax.text(i, caja_val + max(caja_m.max() * 0.01, 5), 
                               f'{caja_val:,.0f}M', ha='center', va='bottom', fontsize=8)
                
                # Añadir línea para deuda neta cero
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                
                st.pyplot(fig)
                plt.close()
                
                # Análisis por sectores
                st.subheader("Análisis por Sectores")
                sector_analysis = deuda_df.merge(df[["Ticker", "Sector"]], on="Ticker")
                sector_stats = sector_analysis.groupby("Sector").agg({
                    "Deuda Total": "mean",
                    "Caja": "mean",
                    "Deuda Neta": "mean",
                    "Ticker": "count"
                }).rename(columns={"Ticker": "Empresas"}).round(0)
                
                # Formatear números grandes
                sector_stats["Deuda Total"] = sector_stats["Deuda Total"].apply(lambda x: f"${x/1e6:,.0f}M")
                sector_stats["Caja"] = sector_stats["Caja"].apply(lambda x: f"${x/1e6:,.0f}M")
                sector_stats["Deuda Neta"] = sector_stats["Deuda Neta"].apply(lambda x: f"${x/1e6:,.0f}M")
                
                st.dataframe(sector_stats, use_container_width=True)
                
            else:
                st.warning("No se pudieron calcular las métricas de deuda neta")

        with tab_facturacion:
            st.subheader("Deuda vs Capacidad de Generación de Ingresos")
            
            # Calcular métricas de deuda vs facturación
            facturacion_metrics = []
            for _, empresa in df.iterrows():
                try:
                    ticker = empresa["Ticker"]
                    
                    # Usar datos existentes
                    total_debt = empresa.get("Total Debt", 0)
                    if hasattr(total_debt, 'iloc'):
                        total_debt = total_debt.iloc[0] if not total_debt.empty else 0
                    
                    cash = empresa.get("Cash And Cash Equivalents", 0)
                    revenue = empresa.get("Total Revenue", 0)
                    
                    total_debt = float(total_debt) if total_debt else 0
                    cash = float(cash) if cash else 0
                    revenue = float(revenue) if revenue else 0
                    
                    net_debt = total_debt - cash
                    
                    # Calcular ratios
                    debt_to_revenue = total_debt / revenue if revenue and revenue != 0 else float('inf')
                    net_debt_to_revenue = net_debt / revenue if revenue and revenue != 0 else float('inf')
                    
                    # Evaluar sostenibilidad
                    if net_debt_to_revenue < 0:
                        evaluation = "✅ Excelente (Deuda Neta Negativa)"
                        eval_color = "green"
                    elif net_debt_to_revenue < 0.1:
                        evaluation = "✅ Muy Buena (<10%)"
                        eval_color = "green"
                    elif net_debt_to_revenue < 0.3:
                        evaluation = "⚠️ Aceptable (10-30%)"
                        eval_color = "orange"
                    elif net_debt_to_revenue == float('inf'):
                        evaluation = "❌ Sin datos de facturación"
                        eval_color = "gray"
                    else:
                        evaluation = "❌ Preocupante (>30%)"
                        eval_color = "red"
                    
                    facturacion_metrics.append({
                        "Ticker": ticker,
                        "Facturación Anual": revenue,
                        "Deuda Total": total_debt,
                        "Deuda Neta": net_debt,
                        "Deuda/Facturación": debt_to_revenue if debt_to_revenue != float('inf') else None,
                        "Deuda Neta/Facturación": net_debt_to_revenue if net_debt_to_revenue != float('inf') else None,
                        "Evaluación": evaluation,
                        "Color": eval_color
                    })
                    
                except Exception as e:
                    continue
            
            if facturacion_metrics:
                fact_df = pd.DataFrame(facturacion_metrics)
                
                # Mostrar tabla
                display_df = fact_df[["Ticker", "Facturación Anual", "Deuda Total", "Deuda Neta", 
                                    "Deuda/Facturación", "Deuda Neta/Facturación", "Evaluación"]].copy()
                
                # Formatear números grandes
                display_df["Facturación Anual"] = display_df["Facturación Anual"].apply(lambda x: f"${x/1e6:,.0f}M" if pd.notnull(x) and x > 0 else "N/D")
                display_df["Deuda Total"] = display_df["Deuda Total"].apply(lambda x: f"${x/1e6:,.0f}M" if pd.notnull(x) and x > 0 else "N/D")
                display_df["Deuda Neta"] = display_df["Deuda Neta"].apply(lambda x: f"${x/1e6:,.0f}M" if pd.notnull(x) else "N/D")
                display_df["Deuda/Facturación"] = display_df["Deuda/Facturación"].apply(lambda x: f"{x:.1%}" if pd.notnull(x) else "N/D")
                display_df["Deuda Neta/Facturación"] = display_df["Deuda Neta/Facturación"].apply(lambda x: f"{x:.1%}" if pd.notnull(x) else "N/D")
                
                st.dataframe(
                    display_df.style.apply(
                        lambda x: [f"background-color: {fact_df.loc[x.name, 'Color']}; color: white" 
                                  if fact_df.loc[x.name, "Color"] in ["red", "orange", "gray"] else "" 
                                  for _ in x], axis=1
                    ),
                    use_container_width=True,
                    height=400
                )
                
                # Gráfico de ratios (solo empresas con datos)
                valid_data = fact_df[fact_df["Deuda Neta/Facturación"].notna() & (fact_df["Deuda Neta/Facturación"] != float('inf'))]
                
                if len(valid_data) > 0:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Ratio Deuda/Facturación
                    colors = [valid_data.iloc[i]["Color"] for i in range(len(valid_data))]
                    
                    bars1 = ax1.bar(valid_data["Ticker"], valid_data["Deuda Neta/Facturación"], color=colors, alpha=0.8)
                    ax1.axhline(y=0.1, color='green', linestyle='--', alpha=0.7, label='Límite 10%')
                    ax1.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='Límite 30%')
                    ax1.set_title('Deuda Neta vs Facturación Anual')
                    ax1.set_ylabel('Ratio (Deuda Neta / Facturación)')
                    ax1.tick_params(axis='x', rotation=45)
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    # Añadir valores a las barras
                    for i, bar in enumerate(bars1):
                        height = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f'{height:.1%}', ha='center', va='bottom', fontsize=8)
                    
                    # Distribución de evaluaciones
                    eval_counts = fact_df["Evaluación"].value_counts()
                    colors_pie = ['green', 'lightgreen', 'orange', 'red', 'gray'][:len(eval_counts)]
                    ax2.pie(eval_counts.values, labels=eval_counts.index, autopct='%1.1f%%',
                           colors=colors_pie)
                    ax2.set_title('Distribución de Evaluaciones de Sostenibilidad')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    # Análisis detallado por empresa
                    st.subheader("Análisis Detallado por Empresa")
                    
                    selected_ticker = st.selectbox("Seleccione una empresa para análisis detallado:", 
                                                  fact_df["Ticker"].unique())
                    
                    if selected_ticker:
                        empresa_data = fact_df[fact_df["Ticker"] == selected_ticker].iloc[0]
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Facturación Anual", f"${empresa_data['Facturación Anual']/1e9:,.2f}B" 
                                     if empresa_data['Facturación Anual'] > 0 else "N/D")
                            st.metric("Deuda Total", f"${empresa_data['Deuda Total']/1e9:,.2f}B" 
                                     if empresa_data['Deuda Total'] > 0 else "N/D")
                            st.metric("Caja", f"${(empresa_data['Deuda Total'] - empresa_data['Deuda Neta'])/1e9:,.2f}B" 
                                     if empresa_data['Deuda Total'] > 0 else "N/D")
                        
                        with col2:
                            st.metric("Deuda Neta", f"${empresa_data['Deuda Neta']/1e9:,.2f}B" 
                                     if pd.notnull(empresa_data['Deuda Neta']) else "N/D")
                            st.metric("Deuda/Facturación", f"{empresa_data['Deuda/Facturación']:.2%}" 
                                     if pd.notnull(empresa_data['Deuda/Facturación']) else "N/D")
                            st.metric("Deuda Neta/Facturación", f"{empresa_data['Deuda Neta/Facturación']:.2%}" 
                                     if pd.notnull(empresa_data['Deuda Neta/Facturación']) else "N/D")
                        
                        with col3:
                            # Evaluación de riesgo
                            if empresa_data['Color'] == 'green':
                                if empresa_data['Deuda Neta'] < 0:
                                    st.success("✅ MÚSCULO FINANCIERO FUERTE")
                                    st.info("La empresa tiene más efectivo que deuda, lo que le proporciona gran resiliencia ante crisis.")
                                else:
                                    st.success("✅ DEUDA MANEJABLE")
                                    st.info("La deuda representa menos del 10% de la facturación anual, nivel muy conservador.")
                            elif empresa_data['Color'] == 'orange':
                                st.warning("⚠️ DEUDA MODERADA")
                                st.info("La deuda está en niveles aceptables pero debe monitorearse.")
                            elif empresa_data['Color'] == 'red':
                                st.error("❌ DEUDA ELEVADA")
                                st.info("La deuda supera el 30% de la facturación, lo que podría ser preocupante en crisis económicas.")
                            elif empresa_data['Color'] == 'gray':
                                st.warning("⚠️ DATOS INCOMPLETOS")
                                st.info("No se dispone de datos completos para evaluar la sostenibilidad.")
                            
                            # Recomendación basada en el análisis
                            if empresa_data['Color'] == 'green':
                                if empresa_data['Deuda Neta'] < 0:
                                    recommendation = "Excelente posición. Considerar inversión."
                                else:
                                    recommendation = "Buena posición. Deuda manejable."
                            elif empresa_data['Color'] == 'orange':
                                recommendation = "Posición aceptable. Monitorear tendencia."
                            elif empresa_data['Color'] == 'red':
                                recommendation = "Posición riesgosa. Evaluar cuidadosamente."
                            else:
                                recommendation = "Datos insuficientes para recomendación."
                            
                            st.metric("Recomendación", recommendation)
                else:
                    st.warning("No hay datos suficientes para generar gráficos")
            else:
                st.warning("No se pudieron calcular las métricas de facturación")

        # =====================================================
        # SECCIÓN 5: CRECIMIENTO
        # =====================================================
        st.header("🚀 Crecimiento (CAGR 3-4 años, por sector)")
        
        for sec in sectors_ordered:
            sec_df = df[df["Sector"] == sec]
            if sec_df.empty:
                continue
                
            with st.expander(f"Sector: {sec}", expanded=False):
                for i, chunk in enumerate(chunk_df(sec_df), 1):
                    st.caption(f"Bloque {i}")
                    fig, ax = plt.subplots(figsize=(12, 6))  # CORREGIDO: subplots no subforms
                    gdf = pd.DataFrame({
                        "Revenue Growth": (chunk["Revenue Growth"]*100).values,
                        "EPS Growth": (chunk["EPS Growth"]*100).values,
                        "FCF Growth": (chunk["FCF Growth"]*100).values
                    }, index=chunk["Ticker"])
                    gdf.plot(kind="bar", ax=ax, rot=45)
                    ax.axhline(0, color="black", linewidth=0.8)
                    ax.set_ylabel("%")
                    auto_ylim(ax, gdf)
                    # AÑADIR VALORES A LAS BARRAS
                    add_value_labels(ax, formatter=lambda x: f"{x:.1f}%")
                    st.pyplot(fig)
                    plt.close()

        # =====================================================
        # SECCIÓN 6: ANÁLISIS INDIVIDUAL
        # =====================================================
        st.header("🔍 Análisis por Empresa")
        pick = st.selectbox("Selecciona empresa", df_disp["Ticker"].unique())
        det_disp = df_disp[df_disp["Ticker"] == pick].iloc[0]
        det_raw = df[df["Ticker"] == pick].iloc[0]

        st.markdown(f"""
        **{det_raw['Nombre']}**  
        **Sector:** {det_raw['Sector']}  
        **País:** {det_raw['País']}  
        **Industria:** {det_raw['Industria']}
        """)

        cA, cB, cC = st.columns(3)
        with cA:
            st.metric("Precio", det_disp["Precio"])
            st.metric("P/E", det_disp["P/E"])
            st.metric("P/B", det_disp["P/B"])
            st.metric("P/FCF", det_disp["P/FCF"])
            
        with cB:
            st.metric("Market Cap", det_disp["MarketCap"])
            st.metric("ROIC", det_disp["ROIC"])
            st.metric("WACC", det_disp["WACC"])
            st.metric("Creación Valor", det_disp["Creacion Valor (Wacc vs Roic)"])
            
        with cC:
            st.metric("ROE", det_disp["ROE"])
            st.metric("Dividend Yield", det_disp["Dividend Yield %"])
            st.metric("Current Ratio", det_disp["Current Ratio"])
            st.metric("Debt/Eq", det_disp["Debt/Eq"])

        # Nueva sección de análisis individual de estructura de capital
        st.subheader("🏦 Análisis de Estructura de Capital y Sostenibilidad de Deuda")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Ratios de Sostenibilidad de Deuda**")
            ratios_data = {
                "Ratio": ["Debt to Equity", "Interest Coverage", "Debt to EBITDA", 
                         "Cash Flow to Debt", "FCF to Debt"],
                "Valor": [
                    det_disp["Debt to Equity"],
                    det_disp["Interest Coverage"],
                    det_disp["Debt to EBITDA"],
                    det_disp["Cash Flow to Debt"],
                    det_disp["FCF to Debt"]
                ],
                "Umbral": [
                    umbrales["Debt to Equity"],
                    umbrales["Interest Coverage"],
                    umbrales["Debt to EBITDA"],
                    umbrales["Cash Flow to Debt"],
                    umbrales["FCF to Debt"]
                ],
                "Evaluación": [
                    "✅" if pd.notnull(det_raw.get("Debt to Equity")) and det_raw.get("Debt to Equity") < umbrales["Debt to Equity"] else "❌",
                    "✅" if pd.notnull(det_raw.get("Interest Coverage")) and det_raw.get("Interest Coverage") > umbrales["Interest Coverage"] else "❌",
                    "✅" if pd.notnull(det_raw.get("Debt to EBITDA")) and det_raw.get("Debt to EBITDA") < umbrales["Debt to EBITDA"] else "❌",
                    "✅" if pd.notnull(det_raw.get("Cash Flow to Debt")) and det_raw.get("Cash Flow to Debt") > umbrales["Cash Flow to Debt"] else "❌",
                    "✅" if pd.notnull(det_raw.get("FCF to Debt")) and det_raw.get("FCF to Debt") > umbrales["FCF to Debt"] else "❌"
                ]
            }
            
            ratios_df = pd.DataFrame(ratios_data)
            st.dataframe(ratios_df, hide_index=True, use_container_width=True)
            
            # Evaluación general
            sostenible = all([
                pd.notnull(det_raw.get("Debt to Equity")) and det_raw.get("Debt to Equity") < umbrales["Debt to Equity"],
                pd.notnull(det_raw.get("Interest Coverage")) and det_raw.get("Interest Coverage") > umbrales["Interest Coverage"],
                pd.notnull(det_raw.get("Debt to EBITDA")) and det_raw.get("Debt to EBITDA") < umbrales["Debt to EBITDA"],
                pd.notnull(det_raw.get("Cash Flow to Debt")) and det_raw.get("Cash Flow to Debt") > umbrales["Cash Flow to Debt"],
                pd.notnull(det_raw.get("FCF to Debt")) and det_raw.get("FCF to Debt") > umbrales["FCF to Debt"]
            ])
            
            if sostenible:
                st.success("✅ La deuda de esta empresa es SOSTENIBLE según los ratios analizados")
            else:
                st.error("❌ La deuda de esta empresa podría NO SER SOSTENIBLE")
        
        with col2:
            st.markdown("**Evolución de la Estructura de Capital**")
            
            # Gráfico de estructura de capital
            total_assets = det_raw.get("Total Assets")
            if total_assets is not None and not total_assets.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                years = []
                for col in total_assets.index:
                    if hasattr(col, 'strftime'):
                        years.append(col.strftime('%Y'))
                    else:
                        years.append(str(col))
                x_pos = np.arange(len(years))
                width = 0.25
                
                bars1 = ax.bar(x_pos - width, total_assets.values / 1e6, width, label='Activos Totales', color='#0074D9')
                bars2 = ax.bar(x_pos, det_raw["Total Liabilities"].values / 1e6, width, label='Pasivos Totales', color='#FF4136')
                bars3 = ax.bar(x_pos + width, det_raw["Total Equity"].values / 1e6, width, label='Patrimonio Neto', color='#2ECC40')
                
                ax.set_xlabel('Año')
                ax.set_ylabel('Millones USD')
                ax.set_title(f'Estructura de Capital - {det_raw["Ticker"]}')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(years)
                ax.legend()
                
                st.pyplot(fig)
                plt.close()
            else:
                st.warning("Datos insuficientes para mostrar la estructura de capital")

        st.subheader("ROIC vs WACC")
        if pd.notnull(det_raw.get("ROIC")) and pd.notnull(det_raw.get("WACC")):
            fig, ax = plt.subplots(figsize=(5, 4))
            comp = pd.DataFrame({
                "ROIC": [det_raw["ROIC"]*100],
                "WACC": [det_raw["WACC"]*100]
            }, index=[pick])
            comp.plot(kind="bar", ax=ax, rot=0, legend=False, 
                     color=["green" if det_raw["ROIC"] > det_raw["WACC"] else "red", "gray"])
            ax.set_ylabel("%")
            auto_ylim(ax, comp)
            # AÑADIR VALORES A LAS BARRAS
            for i, (value, color) in enumerate(zip([det_raw["ROIC"]*100, det_raw["WACC"]*100], 
                                                 ["green" if det_raw["ROIC"] > det_raw["WACC"] else "red", "gray"])):
                ax.annotate(f'{value:.1f}%',
                           xy=(i, value),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=10, weight='bold', color=color)
            st.pyplot(fig)
            plt.close()
            
            if det_raw["ROIC"] > det_raw["WACC"]:
                st.success("✅ Crea valor (ROIC > WACC)")
            else:
                st.error("❌ Destruye valor (ROIC < WACC)")
        else:
            st.warning("Datos insuficientes para comparar ROIC/WACC")

if __name__ == "__main__":
    main()
